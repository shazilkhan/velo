# velo

A small, readable **vector database in Rust**, built from first principles.

`velo` implements approximate nearest-neighbour (ANN) search from the ground up:
the distance metrics, the index structures, on-disk persistence, and — just as
importantly — the machinery to *prove* the approximate index is correct. It is
the engine that sits underneath a retrieval-augmented-generation (RAG) system:
the part that is usually a closed-source black box behind an `.embed()` call.

The goal is not to out-benchmark [FAISS](https://github.com/facebookresearch/faiss)
or [Qdrant](https://github.com/qdrant/qdrant). It is to build the same core they
build — an HNSW graph index — in code you can actually read end to end, with the
correctness and throughput measured at every step rather than assumed.

Zero runtime dependencies.

## Why

Almost every "AI app" calls an embeddings API and hands the vectors to a database
someone else wrote. This is the database someone else wrote. Building it means
understanding what actually happens when you ask for the ten nearest neighbours
of a query in a million-vector index — the graph traversal, the distance math,
the accuracy/latency trade-off you are really making.

## Status

Built in public, one phase at a time. Each phase is a working, tested commit.

- [x] **Phase 0 — Foundation.** Metric trait (cosine / dot / L2), an exact
  brute-force `FlatIndex`, and a recall + throughput harness. This is the exact
  baseline everything after it is measured against.
- [x] **Phase 1 — HNSW.** The hierarchical navigable small-world graph index:
  layered insertion, greedy `ef`-search, and the neighbour-selection heuristic —
  stored as index-based adjacency lists. Scored against the Phase 0 baseline for
  recall (see below).
- [ ] **Phase 2 — Speed.** SIMD distance kernels, `criterion` benchmarks,
  recall-vs-throughput curves.
- [ ] **Phase 3 — Persistence.** Memory-mapped segments + a write-ahead log,
  payload storage, and metadata filtering during search.
- [ ] **Phase 4 — Server.** An `axum` HTTP API, collections, and a Docker image.

Roadmap, deliberately out of scope for v1: sharding / distribution, vector
quantization (PQ / SQ), and multiple index types. One node, one index, done
properly first.

## Quick start

```rust
use velo::{HnswIndex, Metric, VectorIndex};

// Both index types share the same VectorIndex interface, so this is identical
// for the exact `FlatIndex`.
let mut index = HnswIndex::new(3, Metric::Cosine);
index.add(1, &[0.1, 0.2, 0.3]);
index.add(2, &[0.9, 0.1, 0.0]);

let hits = index.search(&[0.1, 0.2, 0.25], 1);
assert_eq!(hits[0].id, 1);
```

## The recall harness

An approximate index is only worth anything if you know how much accuracy it
keeps. `velo` treats that as a first-class, measured number, not a footnote:

```
cargo run --release --bin recall
```

```
dataset : 20000 vectors x 128 dims, 200 clusters, metric = cosine

build   : HNSW built in 4.41s

index         queries/sec      recall@10
----------------------------------------
flat (exact)          787          1.000
hnsw                16168          1.000

speedup : 20.5x faster than exact search
```

The flat index is exact, so its recall is 1.000 by definition — it *is* the
ground truth. HNSW recovers the same top-10 while answering ~20× more queries
per second. That is the whole point of an ANN index: near-exact results at a
fraction of the cost.

> **A note on the data.** The harness samples vectors around random cluster
> centres, because that is how real embeddings behave — they group by meaning.
> Uniformly random vectors would be misleading: in 128 dimensions they sit at
> nearly identical distances (the curse of dimensionality), so "nearest
> neighbour" becomes meaningless and *no* ANN index scores well. Clustered data
> is the honest, representative benchmark, and it is why standard ANN suites
> (SIFT, GloVe) use structured real vectors too.

*(Numbers from a single developer laptop; run `cargo run --release --bin recall`
to reproduce. Correctness is also enforced in CI by a unit test that requires
HNSW recall to stay above 0.90 against brute force.)*

## Design notes

- **Distance as a single convention.** Every metric returns a *distance* where
  smaller means more similar, so search is always "find the k smallest" and the
  index code never branches on the metric.
- **Graphs without pointer soup.** The HNSW index (Phase 1) stores its graph as
  index-based adjacency lists (`u32` node ids into flat `Vec`s), not
  `Rc<RefCell<Node>>`. This is the idiomatic way to build graph structures in
  Rust and it keeps the borrow checker out of the way.
- **One trait, many indexes.** `FlatIndex` and the coming HNSW index both
  implement `VectorIndex`, which is what lets the harness swap the index under
  test without touching the measurement code.

## Development

```
cargo test            # unit tests
cargo clippy --all-targets -- -D warnings
cargo fmt --all
cargo run --release --bin recall
```

## License

MIT © Shazil Khan
