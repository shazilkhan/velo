//! Scalar quantization: compress `f32` vectors to `u8` codes.
//!
//! Each dimension is mapped independently from its observed `[min, max]` range
//! onto the 256 buckets of a byte. That is a 4x reduction in vector storage
//! (`f32` -> `u8`) for the cost of a little precision — and because embedding
//! coordinates are bounded and roughly uniform within a dimension, the precision
//! lost is small, which the recall harness quantifies.
//!
//! Distance is computed by dequantizing on the fly (`min + code * scale`), so no
//! decompressed copy of the dataset is ever materialised.

/// A per-dimension affine map between `f32` values and `u8` codes, learned from
/// a sample of vectors.
#[derive(Debug, Clone)]
pub(crate) struct ScalarQuantizer {
    dim: usize,
    /// Per-dimension minimum; the value a code of `0` decodes to.
    min: Vec<f32>,
    /// Per-dimension step, `(max - min) / 255`. Zero for a constant dimension.
    scale: Vec<f32>,
}

impl ScalarQuantizer {
    /// Learn per-dimension ranges from a row-major `count × dim` matrix.
    pub(crate) fn train(dim: usize, data: &[f32]) -> Self {
        let mut min = vec![f32::INFINITY; dim];
        let mut max = vec![f32::NEG_INFINITY; dim];
        for row in data.chunks_exact(dim) {
            for (d, &v) in row.iter().enumerate() {
                min[d] = min[d].min(v);
                max[d] = max[d].max(v);
            }
        }
        // An empty dataset (or a dimension that never varied) collapses to a
        // zero range, which encodes everything to code 0.
        for d in 0..dim {
            if !min[d].is_finite() {
                min[d] = 0.0;
                max[d] = 0.0;
            }
        }
        let scale = (0..dim)
            .map(|d| {
                let range = max[d] - min[d];
                if range > 0.0 {
                    range / 255.0
                } else {
                    0.0
                }
            })
            .collect();
        Self { dim, min, scale }
    }

    /// Reconstruct a quantizer from previously-saved parameters.
    pub(crate) fn from_parts(dim: usize, min: Vec<f32>, scale: Vec<f32>) -> Self {
        debug_assert_eq!(min.len(), dim);
        debug_assert_eq!(scale.len(), dim);
        Self { dim, min, scale }
    }

    pub(crate) fn min(&self) -> &[f32] {
        &self.min
    }

    pub(crate) fn scale(&self) -> &[f32] {
        &self.scale
    }

    /// Encode one `dim`-length vector into `out` (also `dim` bytes).
    pub(crate) fn encode_into(&self, vector: &[f32], out: &mut [u8]) {
        for d in 0..self.dim {
            out[d] = if self.scale[d] > 0.0 {
                (((vector[d] - self.min[d]) / self.scale[d]).round()).clamp(0.0, 255.0) as u8
            } else {
                0
            };
        }
    }

    /// Dequantize a single code back to an approximate `f32`.
    #[inline]
    pub(crate) fn dequantize(&self, d: usize, code: u8) -> f32 {
        self.min[d] + code as f32 * self.scale[d]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_within_one_step() {
        let dim = 4;
        let data = vec![
            -1.0, 0.0, 1.0, 2.0, //
            -0.5, 0.25, 0.5, 1.5, //
            0.0, 0.5, -1.0, 2.0,
        ];
        let q = ScalarQuantizer::train(dim, &data);

        let mut code = vec![0u8; dim];
        for row in data.chunks_exact(dim) {
            q.encode_into(row, &mut code);
            for d in 0..dim {
                let back = q.dequantize(d, code[d]);
                // Reconstruction error is bounded by one quantization step.
                assert!((back - row[d]).abs() <= q.scale()[d] + 1e-6);
            }
        }
    }

    #[test]
    fn constant_dimension_is_stable() {
        let dim = 2;
        let data = vec![3.0, 7.0, 3.0, 7.0, 3.0, 7.0];
        let q = ScalarQuantizer::train(dim, &data);
        let mut code = vec![0u8; dim];
        q.encode_into(&[3.0, 7.0], &mut code);
        assert_eq!(q.dequantize(0, code[0]), 3.0);
        assert_eq!(q.dequantize(1, code[1]), 7.0);
    }
}
