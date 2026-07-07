//! Distance metrics for vector similarity search.
//!
//! Every metric is expressed as a *distance*: smaller means more similar. That
//! single convention lets the rest of the engine treat search uniformly as
//! "find the k smallest distances", no matter which metric the caller picked.

/// A distance metric over dense `f32` vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Cosine distance, `1 - cosine_similarity`. Range `[0, 2]`; direction only,
    /// magnitude-invariant. The right default for text embeddings.
    Cosine,
    /// Negative inner product. A larger dot product yields a smaller distance,
    /// so this ranks by raw similarity (assumes vectors are pre-normalised).
    Dot,
    /// Squared Euclidean (L2) distance. Monotonic with true L2 but skips the
    /// square root, which does not change ordering.
    L2,
}

impl Metric {
    /// Distance between two equal-length vectors under this metric.
    ///
    /// # Panics
    /// Panics if `a.len() != b.len()`.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "vectors must have equal length");
        match self {
            Metric::Cosine => cosine_distance(a, b),
            Metric::Dot => -dot(a, b),
            Metric::L2 => l2_squared(a, b),
        }
    }
}

/// Inner (dot) product of two equal-length vectors.
///
/// SIMD-accelerated where the CPU supports it; see the crate-internal `simd` module.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot(a, b)
}

/// Squared Euclidean distance between two equal-length vectors.
///
/// SIMD-accelerated where the CPU supports it; see the crate-internal `simd` module.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::l2_squared(a, b)
}

/// Cosine distance (`1 - cosine_similarity`) between two equal-length vectors.
///
/// If either vector has zero magnitude the similarity is undefined, and we
/// report the maximally-dissimilar value `1.0` rather than a `NaN`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = crate::simd::dot(a, b);
    let norm_a = crate::simd::dot(a, a);
    let norm_b = crate::simd::dot(b, b);
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - (dot / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_cosine_distance() {
        let a = [1.0, 2.0, 3.0];
        assert!(cosine_distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_cosine_distance_one() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_vector_cosine_is_defined() {
        let a = [0.0, 0.0];
        let b = [1.0, 1.0];
        assert_eq!(cosine_distance(&a, &b), 1.0);
    }

    #[test]
    fn l2_of_identical_is_zero() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(l2_squared(&a, &a), 0.0);
    }

    #[test]
    fn dot_metric_ranks_more_similar_as_closer() {
        let q = [1.0, 0.0];
        let near = [1.0, 0.0];
        let far = [-1.0, 0.0];
        assert!(Metric::Dot.distance(&q, &near) < Metric::Dot.distance(&q, &far));
    }
}
