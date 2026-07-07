//! The vector backing store behind an index.
//!
//! An index does not care whether its vectors are stored as full `f32`s or as
//! quantized `u8` codes — it only needs distances. This enum hides that choice
//! behind two distance primitives, so [`crate::hnsw::HnswIndex`] can switch to a
//! 4x-smaller quantized representation without any change to its graph logic.

use crate::metric::Metric;
use crate::quant::ScalarQuantizer;

#[derive(Debug, Clone)]
pub(crate) enum VectorStore {
    /// Full-precision vectors, row-major `len × dim`.
    Plain { dim: usize, data: Vec<f32> },
    /// Scalar-quantized vectors: one `u8` code per component.
    Quantized {
        dim: usize,
        codes: Vec<u8>,
        quant: ScalarQuantizer,
    },
}

impl VectorStore {
    pub(crate) fn plain(dim: usize) -> Self {
        VectorStore::Plain {
            dim,
            data: Vec::new(),
        }
    }

    pub(crate) fn plain_from(dim: usize, data: Vec<f32>) -> Self {
        VectorStore::Plain { dim, data }
    }

    pub(crate) fn quantized_from(dim: usize, codes: Vec<u8>, quant: ScalarQuantizer) -> Self {
        VectorStore::Quantized { dim, codes, quant }
    }

    pub(crate) fn is_quantized(&self) -> bool {
        matches!(self, VectorStore::Quantized { .. })
    }

    /// Append one vector, encoding it if the store is quantized.
    pub(crate) fn push(&mut self, vector: &[f32]) {
        match self {
            VectorStore::Plain { data, .. } => data.extend_from_slice(vector),
            VectorStore::Quantized {
                dim, codes, quant, ..
            } => {
                let start = codes.len();
                codes.resize(start + *dim, 0);
                quant.encode_into(vector, &mut codes[start..]);
            }
        }
    }

    /// Distance from stored vector `i` to an external full-precision `query`.
    #[inline]
    pub(crate) fn dist_to_query(&self, metric: Metric, i: usize, query: &[f32]) -> f32 {
        match self {
            VectorStore::Plain { dim, data } => {
                metric.distance(query, &data[i * dim..(i + 1) * dim])
            }
            VectorStore::Quantized { dim, codes, quant } => {
                let base = i * dim;
                quantized_distance(
                    metric,
                    *dim,
                    |d| quant.dequantize(d, codes[base + d]),
                    |d| query[d],
                )
            }
        }
    }

    /// Distance between two stored vectors `i` and `j`.
    #[inline]
    pub(crate) fn dist_between(&self, metric: Metric, i: usize, j: usize) -> f32 {
        match self {
            VectorStore::Plain { dim, data } => {
                metric.distance(&data[i * dim..(i + 1) * dim], &data[j * dim..(j + 1) * dim])
            }
            VectorStore::Quantized { dim, codes, quant } => {
                let (bi, bj) = (i * dim, j * dim);
                quantized_distance(
                    metric,
                    *dim,
                    |d| quant.dequantize(d, codes[bi + d]),
                    |d| quant.dequantize(d, codes[bj + d]),
                )
            }
        }
    }
}

/// Compute a metric distance over two dimension-indexed value functions,
/// dequantizing lazily so no decompressed vector is materialised. Mirrors the
/// semantics of [`Metric::distance`] exactly.
#[inline]
fn quantized_distance(
    metric: Metric,
    dim: usize,
    a: impl Fn(usize) -> f32,
    b: impl Fn(usize) -> f32,
) -> f32 {
    match metric {
        Metric::L2 => {
            let mut sum = 0.0;
            for d in 0..dim {
                let diff = a(d) - b(d);
                sum += diff * diff;
            }
            sum
        }
        Metric::Dot => {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += a(d) * b(d);
            }
            -dot
        }
        Metric::Cosine => {
            let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for d in 0..dim {
                let (av, bv) = (a(d), b(d));
                dot += av * bv;
                na += av * av;
                nb += bv * bv;
            }
            let denom = (na * nb).sqrt();
            if denom == 0.0 {
                1.0
            } else {
                1.0 - dot / denom
            }
        }
    }
}
