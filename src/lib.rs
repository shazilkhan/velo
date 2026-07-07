//! `velo` — a small, readable vector database in Rust.
//!
//! `velo` implements approximate nearest-neighbour (ANN) search from first
//! principles: the distance metrics, the index structures, and the machinery to
//! prove the approximate index is actually correct. It is the engine that sits
//! underneath a retrieval-augmented-generation (RAG) system — the part that is
//! usually a closed-source black box behind an `.embed()` call.
//!
//! The crate ships two index types behind one [`VectorIndex`] trait:
//!
//! * [`FlatIndex`] — an exact, brute-force baseline. Correct, simple, and the
//!   ground truth every approximate index is measured against.
//! * [`HnswIndex`] — an HNSW graph index: fast, approximate, and scored for
//!   recall against the exact baseline.
//!
//! See `src/bin/recall.rs` for the harness that scores one against the other.

#![warn(missing_docs)]

pub mod flat;
pub mod hnsw;
pub mod metric;
pub mod rng;

pub use flat::FlatIndex;
pub use hnsw::{HnswConfig, HnswIndex};
pub use metric::Metric;

/// A single search hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// Id of the stored vector.
    pub id: u64,
    /// Distance from the query to this vector; smaller means more similar.
    pub distance: f32,
}

/// The interface every index type implements.
///
/// Both the exact [`FlatIndex`] and the approximate [`HnswIndex`] implement it.
/// Sharing one trait is what lets the recall harness swap the index under test
/// without touching the measurement code.
pub trait VectorIndex {
    /// Insert `vector` under `id`.
    ///
    /// Ids are expected to be unique; re-using an id simply appends another
    /// entry, since deduplication is a higher-layer concern.
    fn add(&mut self, id: u64, vector: &[f32]);

    /// Return the `k` nearest stored vectors to `query`, closest first.
    ///
    /// Fewer than `k` results are returned when the index holds fewer than `k`
    /// vectors. An empty index, or `k == 0`, yields an empty result.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Number of stored vectors.
    fn len(&self) -> usize;

    /// True when the index holds no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
