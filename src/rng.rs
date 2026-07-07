//! A tiny deterministic PRNG.
//!
//! HNSW construction is randomised (each vector is assigned a random layer), and
//! benchmarks generate random data. Both need to be *reproducible* — a given
//! seed must always produce the same index and the same numbers — without
//! pulling in an external crate. SplitMix64 is small, fast, and good enough for
//! both jobs.

/// A [SplitMix64](https://prng.di.unimi.it/splitmix64.c) pseudo-random generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Draw the next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draw a uniform `f32` in `[0, 1)` from the top 24 random bits.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Draw a uniform `f64` in the *open* interval `(0, 1)`.
    ///
    /// The result is never exactly `0.0`, so it is safe to pass to `ln()` — as
    /// the HNSW layer-assignment formula does.
    pub fn next_f64_open(&mut self) -> f64 {
        // 53-bit mantissa, nudged by half a step to land strictly inside (0, 1).
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_for_a_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f32_stays_in_unit_range() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn f64_open_is_never_zero() {
        let mut rng = SplitMix64::new(9);
        for _ in 0..100_000 {
            let x = rng.next_f64_open();
            assert!(x > 0.0 && x < 1.0);
        }
    }
}
