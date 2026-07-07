//! Micro-benchmarks for the distance kernels (the engine's hot loop).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use velo::rng::SplitMix64;
use velo::Metric;

fn random_vector(rng: &mut SplitMix64, d: usize) -> Vec<f32> {
    (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

fn distance(c: &mut Criterion) {
    let d = 128;
    let mut rng = SplitMix64::new(1);
    let a = random_vector(&mut rng, d);
    let b = random_vector(&mut rng, d);

    let mut group = c.benchmark_group("distance/128d");
    group.bench_function("cosine", |bench| {
        bench.iter(|| Metric::Cosine.distance(black_box(&a), black_box(&b)))
    });
    group.bench_function("l2", |bench| {
        bench.iter(|| Metric::L2.distance(black_box(&a), black_box(&b)))
    });
    group.bench_function("dot", |bench| {
        bench.iter(|| Metric::Dot.distance(black_box(&a), black_box(&b)))
    });
    group.finish();
}

criterion_group!(benches, distance);
criterion_main!(benches);
