//! End-to-end query latency for the HNSW index.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use velo::rng::SplitMix64;
use velo::{HnswIndex, Metric, VectorIndex};

fn random_vector(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

fn clustered_point(rng: &mut SplitMix64, centers: &[Vec<f32>], d: usize) -> Vec<f32> {
    let center = &centers[(rng.next_u64() as usize) % centers.len()];
    (0..d)
        .map(|i| center[i] + 0.15 * (rng.next_f32() * 2.0 - 1.0))
        .collect()
}

fn search(c: &mut Criterion) {
    let d = 96;
    let n = 10_000;
    let clusters = 100;
    let mut rng = SplitMix64::new(7);

    let centers: Vec<Vec<f32>> = (0..clusters).map(|_| random_vector(&mut rng, d)).collect();
    let mut index = HnswIndex::new(d, Metric::Cosine);
    for id in 0..n {
        let v = clustered_point(&mut rng, &centers, d);
        index.add(id as u64, &v);
    }
    let queries: Vec<Vec<f32>> = (0..256)
        .map(|_| clustered_point(&mut rng, &centers, d))
        .collect();

    let mut i = 0usize;
    c.bench_function("hnsw/search_k10", |bench| {
        bench.iter(|| {
            let hits = index.search(black_box(&queries[i % queries.len()]), 10);
            i += 1;
            hits
        })
    });
}

criterion_group!(benches, search);
criterion_main!(benches);
