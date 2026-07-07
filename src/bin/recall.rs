//! Recall / throughput harness.
//!
//! An approximate index trades a little accuracy for a lot of speed, so "it
//! runs" is never enough — we have to *measure* how much accuracy survives. This
//! binary builds an exact [`FlatIndex`] as ground truth and reports recall@k
//! plus queries-per-second for the index under test.
//!
//! In Phase 0 the index under test *is* the flat index, so recall is 1.000 by
//! construction. That is the point: the measuring apparatus lands before the
//! HNSW index it exists to judge. In Phase 1 the `candidate` below becomes the
//! HNSW graph and these numbers start telling the real story.

use std::collections::HashSet;
use std::time::Instant;

use velo::{FlatIndex, Metric, SearchResult, VectorIndex};

fn main() {
    let n = 10_000; // dataset size
    let d = 128; // dimensions
    let q = 1_000; // number of queries
    let k = 10; // neighbours per query

    let mut rng = SplitMix64::new(0x9E37_79B9_7F4A_7C15);
    println!("building dataset: {n} vectors x {d} dims");

    let mut truth = FlatIndex::new(d, Metric::Cosine);
    for id in 0..n {
        truth.add(id as u64, &random_vector(&mut rng, d));
    }

    // Phase 0: the candidate is another exact index, so it trivially matches
    // ground truth. Swap this line for the HNSW index in Phase 1 and the same
    // harness scores it unchanged.
    let candidate = truth.clone();

    let queries: Vec<Vec<f32>> = (0..q).map(|_| random_vector(&mut rng, d)).collect();

    let truth_hits: Vec<Vec<SearchResult>> =
        queries.iter().map(|query| truth.search(query, k)).collect();

    let start = Instant::now();
    let candidate_hits: Vec<Vec<SearchResult>> = queries
        .iter()
        .map(|query| candidate.search(query, k))
        .collect();
    let elapsed = start.elapsed();

    let mean_recall = candidate_hits
        .iter()
        .zip(&truth_hits)
        .map(|(c, t)| recall_at_k(c, t))
        .sum::<f32>()
        / q as f32;

    let qps = q as f64 / elapsed.as_secs_f64();

    println!();
    println!("index      : FlatIndex (exact baseline)");
    println!("metric     : cosine");
    println!("dataset    : {n} vectors");
    println!("queries    : {q}");
    println!("k          : {k}");
    println!("recall@{k}  : {mean_recall:.3}");
    println!("throughput : {qps:.0} queries/sec");
}

/// Fraction of the true top-k that the candidate also returned, in `[0, 1]`.
fn recall_at_k(candidate: &[SearchResult], truth: &[SearchResult]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_ids: HashSet<u64> = truth.iter().map(|r| r.id).collect();
    let found = candidate
        .iter()
        .filter(|r| truth_ids.contains(&r.id))
        .count();
    found as f32 / truth.len() as f32
}

fn random_vector(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

/// Tiny deterministic PRNG (SplitMix64) so benchmark runs are reproducible
/// without pulling in a dependency. `velo` has zero runtime dependencies, and
/// the harness keeps it that way.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `f32` in `[0, 1)` drawn from the top 24 random bits.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}
