//! Recall / throughput harness.
//!
//! An approximate index trades a little accuracy for a lot of speed, so "it
//! runs" is never enough — we have to *measure* how much accuracy survives and
//! how much speed we bought. This binary builds an exact [`FlatIndex`] as ground
//! truth and an [`HnswIndex`] as the index under test, then reports recall@k and
//! the throughput of each.
//!
//! ```text
//! cargo run --release --bin recall
//! ```

use std::collections::HashSet;
use std::f32::consts::TAU;
use std::time::Instant;

use velo::rng::SplitMix64;
use velo::{FlatIndex, HnswIndex, Metric, SearchResult, VectorIndex};

fn main() {
    let n = 20_000; // dataset size
    let d = 128; // dimensions
    let q = 1_000; // number of queries
    let k = 10; // neighbours per query
    let clusters = 200; // topical clusters in the synthetic data
    let metric = Metric::Cosine;

    let mut rng = SplitMix64::new(0x9E37_79B9_7F4A_7C15);
    println!("dataset : {n} vectors x {d} dims, {clusters} clusters, metric = cosine\n");

    // Real embeddings cluster by meaning; uniformly random vectors do not, and
    // in high dimensions they sit at near-identical distances (the curse of
    // dimensionality), which makes "nearest" meaningless. So we sample around
    // random cluster centres — a faithful stand-in for how embeddings behave.
    let centers: Vec<Vec<f32>> = (0..clusters).map(|_| random_vector(&mut rng, d)).collect();
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| clustered_point(&mut rng, &centers, d))
        .collect();
    let queries: Vec<Vec<f32>> = (0..q)
        .map(|_| clustered_point(&mut rng, &centers, d))
        .collect();

    // Exact ground truth.
    let mut flat = FlatIndex::new(d, metric);
    for (id, v) in data.iter().enumerate() {
        flat.add(id as u64, v);
    }

    // Approximate index under test. Build time matters too, so we time it.
    let build_start = Instant::now();
    let mut hnsw = HnswIndex::new(d, metric);
    for (id, v) in data.iter().enumerate() {
        hnsw.add(id as u64, v);
    }
    let build = build_start.elapsed();

    let (flat_hits, flat_qps) = timed_search(&flat, &queries, k);

    println!("build   : HNSW built in {:.2}s", build.as_secs_f64());
    println!();
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "ef_search",
        "queries/sec",
        &format!("recall@{k}"),
        "speedup"
    );
    println!("{:-<48}", "");
    println!(
        "{:>10} {:>12.0} {:>12.3} {:>10}",
        "flat", flat_qps, 1.000, "1.0x"
    );

    // Sweep the search-time candidate width. Higher ef_search explores more of
    // the graph, so recall climbs toward the exact result while throughput
    // falls. This curve *is* the approximate-search tradeoff, measured rather
    // than asserted.
    for ef in [10usize, 20, 40, 80, 160] {
        hnsw.set_ef_search(ef);
        let (hits, qps) = timed_search(&hnsw, &queries, k);
        let recall = mean_recall(&hits, &flat_hits, k);
        println!(
            "{:>10} {:>12.0} {:>12.3} {:>9.1}x",
            ef,
            qps,
            recall,
            qps / flat_qps
        );
    }
}

/// Run every query against `index` and return the hit lists plus queries/sec.
fn timed_search(
    index: &impl VectorIndex,
    queries: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<SearchResult>>, f64) {
    let start = Instant::now();
    let hits: Vec<Vec<SearchResult>> = queries.iter().map(|query| index.search(query, k)).collect();
    let qps = queries.len() as f64 / start.elapsed().as_secs_f64();
    (hits, qps)
}

/// Mean recall@k of `candidate` against `truth` across all queries.
fn mean_recall(candidate: &[Vec<SearchResult>], truth: &[Vec<SearchResult>], k: usize) -> f32 {
    let sum: f32 = candidate
        .iter()
        .zip(truth)
        .map(|(c, t)| {
            let truth_ids: HashSet<u64> = t.iter().map(|r| r.id).collect();
            let found = c.iter().filter(|r| truth_ids.contains(&r.id)).count();
            found as f32 / k as f32
        })
        .sum();
    sum / candidate.len() as f32
}

fn random_vector(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

/// A point sampled near a randomly chosen cluster centre: `centre + noise`.
fn clustered_point(rng: &mut SplitMix64, centers: &[Vec<f32>], d: usize) -> Vec<f32> {
    let center = &centers[(rng.next_u64() as usize) % centers.len()];
    const SIGMA: f32 = 0.15;
    (0..d).map(|i| center[i] + SIGMA * gaussian(rng)).collect()
}

/// A standard-normal sample via the Box–Muller transform.
fn gaussian(rng: &mut SplitMix64) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}
