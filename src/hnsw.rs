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

use crate::rng::SplitMix64;
use crate::{Metric, SearchResult, VectorIndex};

/// Hard ceiling on layer count, so a pathological random draw can never blow up
/// per-node storage. Reaching even layer 16 is astronomically unlikely.
const MAX_LEVEL: usize = 32;

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
    /// Row-major `len × dim` matrix of every stored vector.
    data: Vec<f32>,
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
            data: Vec::new(),
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

    #[inline]
    fn vector(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
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
    ) -> Vec<Scored> {
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        // `candidates` is a min-heap (nearest first) of frontier nodes to expand.
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        // `w` is a max-heap (farthest first) of the best `ef` results so far.
        let mut w: BinaryHeap<Scored> = BinaryHeap::new();

        for &ep in entry_points {
            let s = Scored {
                dist: self.metric.distance(query, self.vector(ep as usize)),
                node: ep,
            };
            visited.insert(ep);
            candidates.push(Reverse(s));
            w.push(s);
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
                    let d = self.metric.distance(query, self.vector(e as usize));
                    let admit = w.len() < ef || w.peek().map_or(true, |f| d < f.dist);
                    if admit {
                        let s = Scored { dist: d, node: e };
                        candidates.push(Reverse(s));
                        w.push(s);
                        if w.len() > ef {
                            w.pop();
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
                let d = self
                    .metric
                    .distance(self.vector(cand.node as usize), self.vector(r as usize));
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
        let base = self.vector(node as usize).to_vec();
        let mut cands: Vec<Scored> = self.links[node as usize][layer]
            .iter()
            .map(|&x| Scored {
                dist: self.metric.distance(&base, self.vector(x as usize)),
                node: x,
            })
            .collect();
        cands.sort_unstable();
        self.links[node as usize][layer] = self.select_neighbors(&cands, m_max);
    }
}

impl VectorIndex for HnswIndex {
    fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");

        let node = self.ids.len() as u32;
        self.ids.push(id);
        self.data.extend_from_slice(vector);

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
                if let Some(nearest) = self.search_layer(vector, &[ep], 1, lc).first() {
                    ep = nearest.node;
                }
            }
        }

        // Phase 2: from our insertion level down to 0, find neighbours with a
        // full-width beam, wire them bidirectionally, and prune as needed.
        let start = level.min(max_layer);
        let mut entry_points = vec![ep];
        for lc in (0..=start).rev() {
            let found = self.search_layer(vector, &entry_points, self.config.ef_construction, lc);
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
            if let Some(nearest) = self.search_layer(query, &[ep], 1, lc).first() {
                ep = nearest.node;
            }
        }

        let ef = self.config.ef_search.max(k);
        self.search_layer(query, &[ep], ef, 0)
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
}
