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
- [x] **Phase 2 — Speed.** Hand-written AVX2 + FMA distance kernels (runtime
  feature-detected, scalar fallback), `criterion` micro-benchmarks, and a
  recall-vs-throughput sweep (see below).
- [x] **Phase 3 — Database features.** Per-vector metadata payloads, filtered
  search (typed `Eq`/`Gt`/`Lt` + `And`/`Or`/`Not`), and full save/load to a
  compact binary format — all dependency-free and round-trip tested.
- [ ] **Phase 4 — Server.** An `axum` HTTP API, collections, and a Docker image.

Roadmap, deliberately out of scope for v1: memory-mapped / write-ahead-logged
storage (the current persistence is a whole-index snapshot, which is correct and
simple; mmap + WAL is the incremental-durability optimization for later),
sharding / distribution, vector quantization (PQ / SQ), and multiple index
types. One node, one index, done properly first.

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

### Metadata filtering and persistence

```rust
use velo::{Filter, HnswIndex, Metric, Payload, Value, VectorIndex};

let mut index = HnswIndex::new(3, Metric::Cosine);

let mut meta = Payload::new();
meta.insert("lang".into(), Value::Str("en".into()));
meta.insert("year".into(), Value::Int(2023));
index.add_with_payload(1, &[0.1, 0.2, 0.3], meta);

// Nearest neighbours, restricted to vectors whose payload matches.
let filter = Filter::And(vec![
    Filter::Eq("lang".into(), Value::Str("en".into())),
    Filter::Gt("year".into(), 2020.0),
]);
let hits = index.search_filtered(&[0.1, 0.2, 0.25], 10, &filter);

// Snapshot the whole index — vectors, graph, and payloads — to disk and back.
index.save("index.velo").unwrap();
let reloaded = HnswIndex::load("index.velo").unwrap();
```

Filtered search still traverses the graph by distance for connectivity, but only
matching vectors enter the result set — so a very selective filter just needs a
larger `ef_search`.

## The recall harness

An approximate index is only worth anything if you know how much accuracy it
keeps. `velo` treats that as a first-class, measured number, not a footnote:

```
cargo run --release --bin recall
```

```
dataset : 20000 vectors x 128 dims, 200 clusters, metric = cosine

build   : HNSW built in 2.96s

 ef_search  queries/sec    recall@10    speedup
------------------------------------------------
      flat         1648        1.000       1.0x
        10        44536        0.939      27.0x
        20        34051        0.991      20.7x
        40        20807        1.000      12.6x
        80        17132        1.000      10.4x
       160         8061        1.000       4.9x
```

The flat index is exact, so its recall is 1.000 by definition — it *is* the
ground truth. Sweeping `ef_search` traces the entire approximate-search
tradeoff: at `ef_search = 20`, HNSW recovers **99%** of the true neighbours at
**~21× the throughput** of exact search; push it to 40 and recall is
indistinguishable from exact while still ~13× faster. That curve is the whole
point of an ANN index, and it is measured here, not asserted.

The distance kernels themselves are benchmarked with `criterion` (`cargo bench`).
Per call on 128-dim `f32` vectors, via the AVX2 + FMA path:

| kernel | time    |
| ------ | ------- |
| dot    | ~7.8 ns |
| L2²    | ~8.7 ns |
| cosine | ~27 ns  |

(Cosine is roughly three dot products — `a·b`, `a·a`, `b·b` — hence ~3× dot.)

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
- **One trait, many indexes.** `FlatIndex` and `HnswIndex` both implement
  `VectorIndex`, which is what lets the harness swap the index under test without
  touching the measurement code.
- **SIMD without a dependency.** Distance is the hot loop, so on x86-64 it runs a
  hand-written AVX2 + FMA kernel (eight lanes per instruction), selected at
  *runtime* via `is_x86_feature_detected!` with a scalar fallback for other CPUs
  and architectures. No `unsafe` leaks into the public API, and the crate stays
  dependency-free.
- **Filtering is graph-aware, not post-hoc.** A filtered search keeps exploring
  the graph by distance (so connectivity is preserved) while admitting only
  matching vectors to the result set — the same `search_layer` primitive, with an
  optional predicate threaded through.
- **Persistence is hand-rolled and readable.** Save/load is a small, explicit
  little-endian format (`persist.rs`) rather than a `serde` + `bincode`
  dependency — fixed-width scalars, length-prefixed strings and payloads, a magic
  header and a version byte.

## Development

```
cargo test            # unit tests, incl. the HNSW recall gate
cargo clippy --all-targets -- -D warnings
cargo fmt --all
cargo run --release --bin recall   # recall-vs-throughput sweep
cargo bench           # criterion micro-benchmarks
```

## License

MIT © Shazil Khan
