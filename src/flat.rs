//! Exact, brute-force index.

use crate::{metric::Metric, SearchResult, VectorIndex};

/// Exhaustive ("flat") index: it compares the query against every stored vector.
///
/// This is `O(n · d)` per query and therefore slow at scale, but it is *exact*.
/// That makes it useful twice over: a correct baseline to ship on day one, and
/// the ground truth the approximate HNSW index is scored against (see
/// `src/bin/recall.rs`).
#[derive(Debug, Clone)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    ids: Vec<u64>,
    /// Row-major `len × dim` matrix of every stored vector, flattened into one
    /// allocation so a full scan stays cache-friendly.
    data: Vec<f32>,
}

impl FlatIndex {
    /// Create an empty index over `dim`-dimensional vectors.
    ///
    /// # Panics
    /// Panics if `dim` is zero.
    pub fn new(dim: usize, metric: Metric) -> Self {
        assert!(dim > 0, "dimension must be non-zero");
        Self {
            dim,
            metric,
            ids: Vec::new(),
            data: Vec::new(),
        }
    }

    /// The dimensionality this index expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric this index searches with.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");
        self.ids.push(id);
        self.data.extend_from_slice(vector);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");
        if k == 0 || self.ids.is_empty() {
            return Vec::new();
        }

        let mut hits: Vec<SearchResult> = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, &id)| SearchResult {
                id,
                distance: self.metric.distance(query, self.row(i)),
            })
            .collect();

        let k = k.min(hits.len());
        // We only need the k smallest distances, so partition first (O(n)) and
        // then sort just that prefix rather than the whole vector.
        hits.select_nth_unstable_by(k - 1, |a, b| total_cmp(a.distance, b.distance));
        hits.truncate(k);
        hits.sort_by(|a, b| total_cmp(a.distance, b.distance));
        hits
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Total ordering over `f32` distances. Distances here are finite and
/// non-negative, but `total_cmp` keeps sorting well-defined even so.
#[inline]
fn total_cmp(a: f32, b: f32) -> std::cmp::Ordering {
    a.total_cmp(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> FlatIndex {
        let mut idx = FlatIndex::new(2, Metric::L2);
        idx.add(1, &[0.0, 0.0]);
        idx.add(2, &[1.0, 0.0]);
        idx.add(3, &[5.0, 5.0]);
        idx
    }

    #[test]
    fn returns_k_closest_in_order() {
        let idx = build();
        let hits = idx.search(&[0.1, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 2);
        assert!(hits[0].distance <= hits[1].distance);
    }

    #[test]
    fn k_larger_than_len_returns_all() {
        let idx = build();
        let hits = idx.search(&[0.0, 0.0], 100);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn empty_index_and_zero_k_return_nothing() {
        let idx = FlatIndex::new(3, Metric::Cosine);
        assert!(idx.is_empty());
        assert!(idx.search(&[1.0, 0.0, 0.0], 5).is_empty());

        let populated = build();
        assert!(populated.search(&[0.0, 0.0], 0).is_empty());
    }
}
