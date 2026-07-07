//! Hierarchical Navigable Small World (HNSW) index.
//!
//! HNSW is the approximate-nearest-neighbour index behind most production vector
//! databases. It builds a layered proximity graph: the bottom layer holds every
//! vector, and each higher layer holds an exponentially thinning sample. A
//! search greedily walks the sparse top layers to get *near* the query cheaply,
//! then explores the dense bottom layer to refine. That turns an `O(n)` scan
//! into something closer to `O(log n)` — while giving up only a little recall.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs" (2016).
//!
//! # Design
//!
//! The graph is stored as **index-based adjacency lists**: nodes are `u32`
//! indices into flat `Vec`s, and `links[node][layer]` is the list of neighbour
//! indices. This is the idiomatic way to build graphs in Rust — no
//! `Rc<RefCell<Node>>`, no lifetime gymnastics, and it keeps the vectors
//! contiguous for cache-friendly distance computation.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::payload::{Filter, Payload};
use crate::persist;
use crate::quant::ScalarQuantizer;
use crate::rng::SplitMix64;
use crate::store::VectorStore;
use crate::{Metric, SearchResult, VectorIndex};

/// Hard ceiling on layer count, so a pathological random draw can never blow up
/// per-node storage. Reaching even layer 16 is astronomically unlikely.
const MAX_LEVEL: usize = 32;

/// Magic bytes at the head of a saved index file.
const MAGIC: &[u8; 4] = b"VELO";

/// On-disk format version. Bump on any breaking change to the layout.
const FORMAT_VERSION: u32 = 2;

/// Construction and search parameters for an [`HnswIndex`].
///
/// The defaults (`m = 16`, `ef_construction = 200`, `ef_search = 64`) are the
/// commonly recommended starting point and give high recall on typical
/// embedding data. Larger values raise recall and cost; smaller values do the
/// reverse.
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Target number of neighbours per node on the upper layers. Layer 0 uses
    /// `2 * m`, since the base layer carries the most connections.
    pub m: usize,
    /// Size of the dynamic candidate list while *inserting*. Higher builds a
    /// better graph at higher cost.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list while *searching*. Higher trades
    /// latency for recall. Always effectively at least `k`.
    pub ef_search: usize,
    /// Seed for the layer-assignment PRNG, so an index built from the same
    /// inserts is bit-for-bit reproducible.
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5EED_1234_ABCD_0001,
        }
    }
}

/// A node paired with its distance to some query point. Ordered by distance
/// (with the node id as a tie-break) so it can drive the search heaps.
#[derive(Debug, Clone, Copy)]
struct Scored {
    dist: f32,
    node: u32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node && self.dist.to_bits() == other.dist.to_bits()
    }
}
impl Eq for Scored {}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.node.cmp(&other.node))
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// An approximate nearest-neighbour index backed by an HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    config: HnswConfig,
    /// Level-generation normaliser, `1 / ln(m)`.
    ml: f64,
    ids: Vec<u64>,
    /// Backing vector storage (full-precision or scalar-quantized).
    store: VectorStore,
    /// Optional metadata per node, indexed like `ids`. `None` = no payload.
    payloads: Vec<Option<Payload>>,
    /// `links[node][layer]` = neighbour node indices of `node` on `layer`.
    links: Vec<Vec<Vec<u32>>>,
    /// Top layer each node participates in.
    node_layer: Vec<usize>,
    entry: Option<u32>,
    max_layer: usize,
    rng: SplitMix64,
}

impl HnswIndex {
    /// Create an empty index over `dim`-dimensional vectors with default
    /// [`HnswConfig`].
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self::with_config(dim, metric, HnswConfig::default())
    }

    /// Create an empty index with explicit parameters.
    ///
    /// # Panics
    /// Panics if `dim` is zero or `config.m` is less than 2.
    pub fn with_config(dim: usize, metric: Metric, config: HnswConfig) -> Self {
        assert!(dim > 0, "dimension must be non-zero");
        assert!(config.m >= 2, "m must be at least 2");
        Self {
            dim,
            metric,
            config,
            ml: 1.0 / (config.m as f64).ln(),
            ids: Vec::new(),
            store: VectorStore::plain(dim),
            payloads: Vec::new(),
            links: Vec::new(),
            node_layer: Vec::new(),
            entry: None,
            max_layer: 0,
            rng: SplitMix64::new(config.seed),
        }
    }

    /// Override the search-time candidate-list size and return `self`, for
    /// fluent construction.
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.config.ef_search = ef_search;
        self
    }

    /// Set the search-time candidate-list size in place.
    ///
    /// Only affects queries, not the stored graph, so it can be swept freely to
    /// trade recall against latency on an already-built index.
    pub fn set_ef_search(&mut self, ef_search: usize) {
        self.config.ef_search = ef_search;
    }

    /// The dimensionality this index expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric this index searches with.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The active configuration.
    pub fn config(&self) -> HnswConfig {
        self.config
    }

    /// Distance from stored vector `i` to an external full-precision query.
    #[inline]
    fn dist_to_query(&self, i: usize, query: &[f32]) -> f32 {
        self.store.dist_to_query(self.metric, i, query)
    }

    /// Distance between two stored vectors.
    #[inline]
    fn dist_between(&self, i: usize, j: usize) -> f32 {
        self.store.dist_between(self.metric, i, j)
    }

    #[inline]
    fn payload(&self, node: usize) -> Option<&Payload> {
        self.payloads.get(node).and_then(Option::as_ref)
    }

    /// Whether vectors are stored in scalar-quantized (`u8`) form.
    pub fn is_quantized(&self) -> bool {
        self.store.is_quantized()
    }

    /// Compress the stored vectors in place with scalar quantization, cutting
    /// vector memory roughly 4x (`f32` -> `u8`).
    ///
    /// Trains the quantizer on the vectors already inserted, so call it after
    /// building the index. Searches afterwards run on the quantized codes and
    /// trade a little recall for the smaller footprint — measure it with the
    /// recall harness. Vectors added later are encoded with the same quantizer.
    /// A no-op if already quantized.
    pub fn quantize(&mut self) {
        if let VectorStore::Plain { dim, data } = &self.store {
            let dim = *dim;
            let quant = ScalarQuantizer::train(dim, data);
            let count = self.ids.len();
            let mut codes = vec![0u8; count * dim];
            for (row, chunk) in codes.chunks_exact_mut(dim).enumerate() {
                quant.encode_into(&data[row * dim..(row + 1) * dim], chunk);
            }
            self.store = VectorStore::quantized_from(dim, codes, quant);
        }
    }

    #[inline]
    fn max_conn(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m * 2
        } else {
            self.config.m
        }
    }

    /// Draw a random top layer for a new node from the exponential
    /// distribution `floor(-ln(U) * ml)`.
    fn random_level(&mut self) -> usize {
        let u = self.rng.next_f64_open();
        ((((-u.ln()) * self.ml).floor()) as usize).min(MAX_LEVEL)
    }

    /// Greedy/beam search within a single `layer`.
    ///
    /// Explores outward from `entry_points`, keeping the `ef` closest vectors to
    /// `query` seen so far, and returns them sorted nearest-first. This is the
    /// primitive both insertion and querying are built from.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        filter: Option<&Filter>,
    ) -> Vec<Scored> {
        let matches = |node: u32| filter.is_none_or(|f| f.matches(self.payload(node as usize)));

        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        // `candidates` is a min-heap (nearest first) of frontier nodes to expand.
        // The frontier is driven purely by distance so the graph stays fully
        // traversable; the `filter` only gates what lands in the result set `w`.
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        // `w` is a max-heap (farthest first) of the best `ef` *matching* results.
        let mut w: BinaryHeap<Scored> = BinaryHeap::new();

        for &ep in entry_points {
            let s = Scored {
                dist: self.dist_to_query(ep as usize, query),
                node: ep,
            };
            visited.insert(ep);
            candidates.push(Reverse(s));
            if matches(ep) {
                w.push(s);
            }
        }
        while w.len() > ef {
            w.pop();
        }

        while let Some(Reverse(c)) = candidates.pop() {
            // If the nearest unexpanded candidate is farther than the farthest
            // result and we already have enough results, we are done.
            if let Some(farthest) = w.peek() {
                if c.dist > farthest.dist && w.len() >= ef {
                    break;
                }
            }
            for &e in &self.links[c.node as usize][layer] {
                if visited.insert(e) {
                    let d = self.dist_to_query(e as usize, query);
                    let improves = w.len() < ef || w.peek().is_none_or(|f| d < f.dist);
                    if improves {
                        let s = Scored { dist: d, node: e };
                        candidates.push(Reverse(s));
                        if matches(e) {
                            w.push(s);
                            if w.len() > ef {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out = w.into_vec();
        out.sort_unstable();
        out
    }

    /// The HNSW neighbour-selection heuristic (Malkov & Yashunin, Alg. 4).
    ///
    /// Given `candidates` sorted nearest-first relative to the point being
    /// linked (their distance to it lives in `Scored::dist`), keep up to `m` of
    /// them, skipping any candidate that sits closer to an already-selected
    /// neighbour than to that point. This favours a spread of directions over a
    /// tight cluster, which is what keeps the graph navigable.
    fn select_neighbors(&self, candidates: &[Scored], m: usize) -> Vec<u32> {
        let mut result: Vec<u32> = Vec::with_capacity(m);
        for cand in candidates {
            if result.len() >= m {
                break;
            }
            let mut keep = true;
            for &r in &result {
                let d = self.dist_between(cand.node as usize, r as usize);
                if d < cand.dist {
                    keep = false;
                    break;
                }
            }
            if keep {
                result.push(cand.node);
            }
        }
        result
    }

    /// Re-run neighbour selection on `node`'s layer-`layer` connections after an
    /// insert pushed it over the connection cap.
    fn prune(&mut self, node: u32, layer: usize, m_max: usize) {
        let mut cands: Vec<Scored> = self.links[node as usize][layer]
            .iter()
            .map(|&x| Scored {
                dist: self.dist_between(node as usize, x as usize),
                node: x,
            })
            .collect();
        cands.sort_unstable();
        self.links[node as usize][layer] = self.select_neighbors(&cands, m_max);
    }
}

impl HnswIndex {
    /// Insert a vector together with a metadata [`Payload`], enabling filtered
    /// search over it via [`search_filtered`](Self::search_filtered).
    pub fn add_with_payload(&mut self, id: u64, vector: &[f32], payload: Payload) {
        self.insert(id, vector, Some(payload));
    }

    /// Nearest neighbours restricted to vectors whose payload matches `filter`.
    ///
    /// The graph is still traversed by distance for connectivity, but only
    /// matching vectors enter the result set. A very selective filter therefore
    /// explores more of the graph for the same `ef_search`; raise it with
    /// [`set_ef_search`](Self::set_ef_search) if recall on a rare filter matters.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: &Filter) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        if k == 0 {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };

        let mut ep = entry;
        for lc in (1..=self.max_layer).rev() {
            if let Some(nearest) = self.search_layer(query, &[ep], 1, lc, None).first() {
                ep = nearest.node;
            }
        }
        let ef = self.config.ef_search.max(k);
        self.search_layer(query, &[ep], ef, 0, Some(filter))
            .into_iter()
            .take(k)
            .map(|s| SearchResult {
                id: self.ids[s.node as usize],
                distance: s.dist,
            })
            .collect()
    }

    fn insert(&mut self, id: u64, vector: &[f32], payload: Option<Payload>) {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");

        let node = self.ids.len() as u32;
        self.ids.push(id);
        self.store.push(vector);
        self.payloads.push(payload);

        let level = self.random_level();
        self.node_layer.push(level);
        self.links.push(vec![Vec::new(); level + 1]);

        // First vector becomes the entry point and we are done.
        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.max_layer = level;
            return;
        };

        let max_layer = self.max_layer;
        let mut ep = entry;

        // Phase 1: greedily descend the layers above our insertion level,
        // narrowing to the closest single entry point.
        if max_layer > level {
            for lc in ((level + 1)..=max_layer).rev() {
                if let Some(nearest) = self.search_layer(vector, &[ep], 1, lc, None).first() {
                    ep = nearest.node;
                }
            }
        }

        // Phase 2: from our insertion level down to 0, find neighbours with a
        // full-width beam, wire them bidirectionally, and prune as needed.
        let start = level.min(max_layer);
        let mut entry_points = vec![ep];
        for lc in (0..=start).rev() {
            let found =
                self.search_layer(vector, &entry_points, self.config.ef_construction, lc, None);
            let m = self.max_conn(lc);
            let selected = self.select_neighbors(&found, m);

            self.links[node as usize][lc] = selected.clone();
            for &nb in &selected {
                self.links[nb as usize][lc].push(node);
                let m_max = self.max_conn(lc);
                if self.links[nb as usize][lc].len() > m_max {
                    self.prune(nb, lc, m_max);
                }
            }

            entry_points = found.iter().map(|s| s.node).collect();
            if entry_points.is_empty() {
                entry_points = vec![ep];
            }
        }

        // A taller node than any seen so far becomes the new entry point.
        if level > max_layer {
            self.entry = Some(node);
            self.max_layer = level;
        }
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: u64, vector: &[f32]) {
        self.insert(id, vector, None);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        if k == 0 {
            return Vec::new();
        }
        let Some(entry) = self.entry else {
            return Vec::new();
        };

        let mut ep = entry;
        for lc in (1..=self.max_layer).rev() {
            if let Some(nearest) = self.search_layer(query, &[ep], 1, lc, None).first() {
                ep = nearest.node;
            }
        }

        let ef = self.config.ef_search.max(k);
        self.search_layer(query, &[ep], ef, 0, None)
            .into_iter()
            .take(k)
            .map(|s| SearchResult {
                id: self.ids[s.node as usize],
                distance: s.dist,
            })
            .collect()
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

impl HnswIndex {
    /// Serialize the whole index — vectors, graph, and payloads — to `path` in
    /// velo's compact binary format.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut writer = BufWriter::new(File::create(path)?);
        self.write_to(&mut writer)?;
        writer.flush()
    }

    /// Load an index previously written by [`save`](Self::save).
    ///
    /// Returns an [`io::Error`] of kind `InvalidData` if the file is not a velo
    /// index or was written by an incompatible format version.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        Self::read_from(&mut reader)
    }

    fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(MAGIC)?;
        persist::write_u32(w, FORMAT_VERSION)?;
        persist::write_u8(w, metric_tag(self.metric))?;
        persist::write_u32(w, self.dim as u32)?;
        persist::write_u32(w, self.config.m as u32)?;
        persist::write_u32(w, self.config.ef_construction as u32)?;
        persist::write_u32(w, self.config.ef_search as u32)?;
        persist::write_u64(w, self.config.seed)?;
        persist::write_i64(w, self.entry.map_or(-1, i64::from))?;
        persist::write_u32(w, self.max_layer as u32)?;

        persist::write_u32(w, self.ids.len() as u32)?;
        for &id in &self.ids {
            persist::write_u64(w, id)?;
        }
        // Vector storage: tag 0 = full-precision f32, tag 1 = scalar-quantized.
        match &self.store {
            VectorStore::Plain { data, .. } => {
                persist::write_u8(w, 0)?;
                for &x in data {
                    persist::write_f32(w, x)?;
                }
            }
            VectorStore::Quantized { codes, quant, .. } => {
                persist::write_u8(w, 1)?;
                for &code in codes {
                    persist::write_u8(w, code)?;
                }
                for &m in quant.min() {
                    persist::write_f32(w, m)?;
                }
                for &s in quant.scale() {
                    persist::write_f32(w, s)?;
                }
            }
        }
        for &level in &self.node_layer {
            persist::write_u32(w, level as u32)?;
        }
        // Each node stores exactly `node_layer + 1` adjacency lists.
        for layers in &self.links {
            for neighbours in layers {
                persist::write_u32(w, neighbours.len() as u32)?;
                for &nb in neighbours {
                    persist::write_u32(w, nb)?;
                }
            }
        }
        for payload in &self.payloads {
            persist::write_payload(w, payload.as_ref())?;
        }
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a velo index",
            ));
        }
        let version = persist::read_u32(r)?;
        if version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported velo format version {version}"),
            ));
        }

        let metric = metric_from_tag(persist::read_u8(r)?)?;
        let dim = persist::read_u32(r)? as usize;
        let m = persist::read_u32(r)? as usize;
        let ef_construction = persist::read_u32(r)? as usize;
        let ef_search = persist::read_u32(r)? as usize;
        let seed = persist::read_u64(r)?;
        let entry_raw = persist::read_i64(r)?;
        let entry = (entry_raw >= 0).then_some(entry_raw as u32);
        let max_layer = persist::read_u32(r)? as usize;

        let count = persist::read_u32(r)? as usize;
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            ids.push(persist::read_u64(r)?);
        }
        let store = match persist::read_u8(r)? {
            0 => {
                let mut data = Vec::with_capacity(count * dim);
                for _ in 0..count * dim {
                    data.push(persist::read_f32(r)?);
                }
                VectorStore::plain_from(dim, data)
            }
            1 => {
                let mut codes = vec![0u8; count * dim];
                for code in &mut codes {
                    *code = persist::read_u8(r)?;
                }
                let mut min = Vec::with_capacity(dim);
                for _ in 0..dim {
                    min.push(persist::read_f32(r)?);
                }
                let mut scale = Vec::with_capacity(dim);
                for _ in 0..dim {
                    scale.push(persist::read_f32(r)?);
                }
                VectorStore::quantized_from(
                    dim,
                    codes,
                    ScalarQuantizer::from_parts(dim, min, scale),
                )
            }
            tag => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown vector store tag {tag}"),
                ))
            }
        };
        let mut node_layer = Vec::with_capacity(count);
        for _ in 0..count {
            node_layer.push(persist::read_u32(r)? as usize);
        }
        let mut links = Vec::with_capacity(count);
        for &level in &node_layer {
            let mut layers = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                let len = persist::read_u32(r)? as usize;
                let mut neighbours = Vec::with_capacity(len);
                for _ in 0..len {
                    neighbours.push(persist::read_u32(r)?);
                }
                layers.push(neighbours);
            }
            links.push(layers);
        }
        let mut payloads = Vec::with_capacity(count);
        for _ in 0..count {
            payloads.push(persist::read_payload(r)?);
        }

        Ok(Self {
            dim,
            metric,
            config: HnswConfig {
                m,
                ef_construction,
                ef_search,
                seed,
            },
            ml: 1.0 / (m as f64).ln(),
            ids,
            store,
            payloads,
            links,
            node_layer,
            entry,
            max_layer,
            rng: SplitMix64::new(seed),
        })
    }
}

fn metric_tag(metric: Metric) -> u8 {
    match metric {
        Metric::Cosine => 0,
        Metric::Dot => 1,
        Metric::L2 => 2,
    }
}

fn metric_from_tag(tag: u8) -> io::Result<Metric> {
    Ok(match tag {
        0 => Metric::Cosine,
        1 => Metric::Dot,
        2 => Metric::L2,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown metric tag {other}"),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FlatIndex;

    fn random_vec(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
        (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn finds_exact_neighbour_in_tiny_index() {
        let mut idx = HnswIndex::new(2, Metric::L2);
        idx.add(10, &[0.0, 0.0]);
        idx.add(20, &[1.0, 0.0]);
        idx.add(30, &[5.0, 5.0]);

        let hits = idx.search(&[0.9, 0.1], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 20);
    }

    #[test]
    fn empty_and_zero_k_return_nothing() {
        let idx = HnswIndex::new(4, Metric::Cosine);
        assert!(idx.is_empty());
        assert!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn results_are_sorted_by_distance() {
        let mut rng = SplitMix64::new(1);
        let mut idx = HnswIndex::new(8, Metric::L2);
        for id in 0..500 {
            idx.add(id, &random_vec(&mut rng, 8));
        }
        let hits = idx.search(&random_vec(&mut rng, 8), 10);
        assert_eq!(hits.len(), 10);
        for pair in hits.windows(2) {
            assert!(pair[0].distance <= pair[1].distance);
        }
    }

    #[test]
    fn recall_matches_brute_force() {
        // The real correctness gate: HNSW must recover almost all of the true
        // nearest neighbours that the exact index finds.
        let d = 16;
        let mut rng = SplitMix64::new(1234);

        let mut hnsw = HnswIndex::new(d, Metric::Cosine);
        let mut flat = FlatIndex::new(d, Metric::Cosine);
        for id in 0..2000u64 {
            let v = random_vec(&mut rng, d);
            hnsw.add(id, &v);
            flat.add(id, &v);
        }

        let k = 10;
        let queries = 200;
        let mut total = 0.0f32;
        for _ in 0..queries {
            let q = random_vec(&mut rng, d);
            let truth: HashSet<u64> = flat.search(&q, k).into_iter().map(|r| r.id).collect();
            let got = hnsw.search(&q, k);
            let hit = got.iter().filter(|r| truth.contains(&r.id)).count();
            total += hit as f32 / k as f32;
        }
        let recall = total / queries as f32;
        assert!(recall > 0.90, "recall too low: {recall:.3}");
    }

    #[test]
    fn filtered_search_returns_only_matching() {
        use crate::payload::{Filter, Payload, Value};

        let d = 8;
        let mut rng = SplitMix64::new(99);
        let mut idx = HnswIndex::new(d, Metric::Cosine);
        for id in 0..1000u64 {
            let v = random_vec(&mut rng, d);
            let mut p = Payload::new();
            let lang = if id % 2 == 0 { "en" } else { "fr" };
            p.insert("lang".into(), Value::Str(lang.into()));
            idx.add_with_payload(id, &v, p);
        }

        let query = random_vec(&mut rng, d);
        let filter = Filter::Eq("lang".into(), Value::Str("en".into()));
        let hits = idx.search_filtered(&query, 10, &filter);

        assert!(!hits.is_empty());
        for hit in &hits {
            assert_eq!(hit.id % 2, 0, "returned non-matching id {}", hit.id);
        }
    }

    #[test]
    fn save_load_roundtrip_preserves_results() {
        use crate::payload::{Filter, Payload, Value};

        let d = 12;
        let mut rng = SplitMix64::new(7);
        let mut idx = HnswIndex::new(d, Metric::Cosine);
        for id in 0..800u64 {
            let v = random_vec(&mut rng, d);
            if id % 3 == 0 {
                let mut p = Payload::new();
                p.insert("k".into(), Value::Int(id as i64));
                idx.add_with_payload(id, &v, p);
            } else {
                idx.add(id, &v);
            }
        }

        let path = std::env::temp_dir().join("velo_roundtrip_test.bin");
        idx.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), idx.len());

        // Identical queries must return identical results on the reloaded index.
        let mut qrng = SplitMix64::new(555);
        for _ in 0..50 {
            let q = random_vec(&mut qrng, d);
            let before: Vec<u64> = idx.search(&q, 10).iter().map(|r| r.id).collect();
            let after: Vec<u64> = loaded.search(&q, 10).iter().map(|r| r.id).collect();
            assert_eq!(before, after);
        }

        // Payloads survive too, so filtered search still works after loading.
        let filter = Filter::Gt("k".into(), 0.0);
        let filtered = loaded.search_filtered(&random_vec(&mut qrng, d), 5, &filter);
        assert!(!filtered.is_empty());
    }

    #[test]
    fn quantization_keeps_recall_high() {
        // Quantized search should still recover most of the exact neighbours.
        let d = 32;
        let mut rng = SplitMix64::new(2024);

        let mut hnsw = HnswIndex::new(d, Metric::L2);
        let mut flat = FlatIndex::new(d, Metric::L2);
        let clusters: Vec<Vec<f32>> = (0..40).map(|_| random_vec(&mut rng, d)).collect();
        for id in 0..3000u64 {
            let c = &clusters[(rng.next_u64() as usize) % clusters.len()];
            let v: Vec<f32> = (0..d)
                .map(|i| c[i] + 0.1 * (rng.next_f32() * 2.0 - 1.0))
                .collect();
            hnsw.add(id, &v);
            flat.add(id, &v);
        }

        assert!(!hnsw.is_quantized());
        hnsw.quantize();
        assert!(hnsw.is_quantized());

        let k = 10;
        let queries = 200;
        let mut total = 0.0f32;
        for _ in 0..queries {
            let c = &clusters[(rng.next_u64() as usize) % clusters.len()];
            let q: Vec<f32> = (0..d)
                .map(|i| c[i] + 0.1 * (rng.next_f32() * 2.0 - 1.0))
                .collect();
            let truth: HashSet<u64> = flat.search(&q, k).into_iter().map(|r| r.id).collect();
            let got = hnsw.search(&q, k);
            total += got.iter().filter(|r| truth.contains(&r.id)).count() as f32 / k as f32;
        }
        let recall = total / queries as f32;
        assert!(recall > 0.85, "quantized recall too low: {recall:.3}");
    }

    #[test]
    fn quantized_index_survives_save_load() {
        let d = 16;
        let mut rng = SplitMix64::new(321);
        let mut idx = HnswIndex::new(d, Metric::Cosine);
        for id in 0..500u64 {
            idx.add(id, &random_vec(&mut rng, d));
        }
        idx.quantize();

        let path = std::env::temp_dir().join("velo_quantized_roundtrip.bin");
        idx.save(&path).unwrap();
        let loaded = HnswIndex::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.is_quantized());
        let q = random_vec(&mut rng, d);
        let before: Vec<u64> = idx.search(&q, 10).iter().map(|r| r.id).collect();
        let after: Vec<u64> = loaded.search(&q, 10).iter().map(|r| r.id).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn load_rejects_a_non_velo_file() {
        let path = std::env::temp_dir().join("velo_bad_magic_test.bin");
        std::fs::write(&path, b"not a velo file at all").unwrap();
        let result = HnswIndex::load(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
