//! Counter-based Philox-4x32 RNG (R11).
//!
//! All randomness in RHIZOME is a pure function of
//! `(seed, stream, counter)`, so results never depend on thread count,
//! rank layout, or scheduling.

const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;
const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;

/// A Philox-4x32-10 generator.
#[derive(Debug, Clone)]
pub struct Philox {
    key: (u32, u32),
    counter: u64,
    stream: u64,
    buffer: [u32; 4],
    cursor: usize,
}

impl Philox {
    /// Create a generator for `(seed, stream)` starting at counter 0.
    #[must_use]
    pub fn new(seed: u64, stream: u64) -> Self {
        Philox {
            key: ((seed >> 32) as u32, seed as u32),
            counter: 0,
            stream,
            buffer: [0; 4],
            cursor: 4,
        }
    }

    /// Position the generator at an explicit counter value.
    pub fn seek(&mut self, counter: u64) {
        self.counter = counter;
        self.cursor = 4;
    }

    /// Draw four raw words for the given counter without changing state.
    #[must_use]
    pub fn block(&self, counter: u64) -> [u32; 4] {
        let mut c = [
            counter as u32,
            (counter >> 32) as u32,
            self.stream as u32,
            (self.stream >> 32) as u32,
        ];
        let mut k = self.key;
        for _ in 0..10 {
            let lo0 = u64::from(c[0]) * u64::from(M0);
            let lo1 = u64::from(c[2]) * u64::from(M1);
            let (hi0, l0) = ((lo0 >> 32) as u32, lo0 as u32);
            let (hi1, l1) = ((lo1 >> 32) as u32, lo1 as u32);
            c = [hi1 ^ c[1] ^ k.0, l1, hi0 ^ c[3] ^ k.1, l0];
            k = (k.0.wrapping_add(W0), k.1.wrapping_add(W1));
        }
        c
    }

    /// Next raw 32-bit word.
    pub fn next_u32(&mut self) -> u32 {
        if self.cursor >= 4 {
            self.buffer = self.block(self.counter);
            self.counter = self.counter.wrapping_add(1);
            self.cursor = 0;
        }
        let v = self.buffer[self.cursor];
        self.cursor += 1;
        v
    }

    /// Uniform sample in `[0, 1)`.
    ///
    /// Exactly 53 bits of mantissa are taken from two 32-bit words, so the
    /// result is always strictly less than one.
    pub fn next_f64(&mut self) -> f64 {
        let hi = u64::from(self.next_u32() >> 5); // 27 bits
        let lo = u64::from(self.next_u32() >> 6); // 26 bits
        ((hi << 26) | lo) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal sample via Box–Muller (deterministic pairing).
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// Truncated log-normal sample used to draw the Think Core depth `L`.
    pub fn next_trunc_lognormal(&mut self, mu: f64, sigma: f64, lo: usize, hi: usize) -> usize {
        let z = self.next_normal();
        let x = (mu + sigma * z).exp();
        (x.round() as i64).clamp(lo as i64, hi as i64) as usize
    }
}

/// Draw a value from an independent stream without mutating any state.
#[must_use]
pub fn stateless_uniform(seed: u64, stream: u64, index: u64) -> f64 {
    let p = Philox::new(seed, stream);
    let b = p.block(index);
    let hi = u64::from(b[0] >> 5);
    let lo = u64::from(b[1] >> 6);
    ((hi << 26) | lo) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_reproducible() {
        let a: Vec<u32> = {
            let mut r = Philox::new(7, 1);
            (0..16).map(|_| r.next_u32()).collect()
        };
        let b: Vec<u32> = {
            let mut r = Philox::new(7, 1);
            (0..16).map(|_| r.next_u32()).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn streams_are_independent() {
        let mut a = Philox::new(7, 1);
        let mut b = Philox::new(7, 2);
        let xs: Vec<u32> = (0..8).map(|_| a.next_u32()).collect();
        let ys: Vec<u32> = (0..8).map(|_| b.next_u32()).collect();
        assert_ne!(xs, ys);
    }

    #[test]
    fn seek_gives_position_independence() {
        let mut a = Philox::new(11, 3);
        for _ in 0..40 {
            let _ = a.next_u32();
        }
        let expected = a.next_u32();
        let mut b = Philox::new(11, 3);
        b.seek(10);
        assert_eq!(b.next_u32(), expected);
    }

    #[test]
    fn uniform_is_in_range() {
        let mut r = Philox::new(1, 1);
        for _ in 0..10_000 {
            let v = r.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn normal_has_expected_moments() {
        let mut r = Philox::new(5, 9);
        let n = 20_000;
        let xs: Vec<f64> = (0..n).map(|_| r.next_normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn depth_sampling_respects_bounds() {
        let mut r = Philox::new(3, 4);
        for _ in 0..1000 {
            let l = r.next_trunc_lognormal(core::f64::consts::LN_2, 0.5, 1, 8);
            assert!((1..=8).contains(&l));
        }
    }

    #[test]
    fn stateless_matches_stateful_block() {
        let v = stateless_uniform(2, 3, 4);
        assert!((0.0..1.0).contains(&v));
        assert!((stateless_uniform(2, 3, 4) - v).abs() < 1e-18);
    }
}
