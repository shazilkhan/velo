//! SIMD-accelerated distance kernels, with scalar fallbacks.
//!
//! Distance computation is the hot loop of the whole engine — every insert and
//! every query does thousands of them. On x86-64 we dispatch to a hand-written
//! AVX2 + FMA kernel that processes eight `f32` lanes per instruction, chosen at
//! *runtime* via feature detection so the same binary still runs (a little
//! slower) on a CPU without AVX2, and on non-x86 targets the scalar path is used
//! directly. This keeps `velo` dependency-free while still going fast.

/// Dot product of two equal-length vectors.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check on this exact line.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

/// Squared Euclidean distance of two equal-length vectors.
#[inline]
pub(crate) fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: guarded by the runtime feature check on this exact line.
            return unsafe { l2_avx2(a, b) };
        }
    }
    l2_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = horizontal_sum(acc);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }
    let mut sum = horizontal_sum(acc);
    while i < n {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Sum the eight lanes of an AVX register. Storing to the stack and summing
/// scalar avoids the SSE3 shuffle intrinsics and is negligible next to the main
/// loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::_mm256_storeu_ps;
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v);
    lanes.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_matches_reference_across_lengths() {
        // Lengths that straddle the 8-lane boundary exercise the SIMD tail.
        for len in [0usize, 1, 7, 8, 9, 16, 31, 128, 129] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 3.0).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32).sin()).collect();
            let got = dot(&a, &b);
            let want = reference_dot(&a, &b);
            assert!((got - want).abs() < 1e-3, "len {len}: {got} vs {want}");
        }
    }

    #[test]
    fn l2_matches_reference_across_lengths() {
        for len in [0usize, 1, 7, 8, 9, 16, 31, 128, 129] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25).collect();
            let b: Vec<f32> = (0..len).map(|i| 10.0 - i as f32).collect();
            let got = l2_squared(&a, &b);
            let want: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
            assert!((got - want).abs() < 1e-2, "len {len}: {got} vs {want}");
        }
    }
}
