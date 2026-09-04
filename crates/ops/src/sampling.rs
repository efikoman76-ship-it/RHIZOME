//! Seeded sampling with grammar masking (SPEC.md §4, §6.6).
//!
//! Sampling is non-differentiable; it is validated by property tests only.

use crate::elementwise::softmax;
use crate::rng::Philox;

/// Philox stream id reserved for token sampling (R11).
pub const SAMPLING_STREAM: u64 = 0x5A11_0000;

/// Sampling parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature; `0` means greedy.
    pub temperature: f64,
    /// Keep only the `top_k` highest logits (`0` disables).
    pub top_k: usize,
    /// Nucleus threshold in `(0, 1]`.
    pub top_p: f64,
    /// Minimum probability relative to the mode.
    pub min_p: f64,
    /// Repetition penalty applied to previously emitted tokens.
    pub repetition_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Apply masking, penalties and filters, returning the final distribution.
#[must_use]
pub fn distribution(logits: &[f64], p: &SamplingParams, history: &[usize], mask: &[bool]) -> Vec<f64> {
    let mut l = logits.to_vec();
    for (i, allowed) in mask.iter().enumerate() {
        if !allowed {
            l[i] = f64::NEG_INFINITY;
        }
    }
    if (p.repetition_penalty - 1.0).abs() > f64::EPSILON {
        for &t in history {
            if t < l.len() && l[t].is_finite() {
                l[t] = if l[t] > 0.0 {
                    l[t] / p.repetition_penalty
                } else {
                    l[t] * p.repetition_penalty
                };
            }
        }
    }
    if p.temperature > 0.0 {
        for v in &mut l {
            if v.is_finite() {
                *v /= p.temperature;
            }
        }
    }
    let mut probs = softmax(&l);
    if p.top_k > 0 && p.top_k < probs.len() {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| {
            probs[*b]
                .partial_cmp(&probs[*a])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        for &i in &idx[p.top_k..] {
            probs[i] = 0.0;
        }
    }
    if p.top_p < 1.0 {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| {
            probs[*b]
                .partial_cmp(&probs[*a])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
        let mut cum = 0.0;
        let mut cutoff = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            cum += probs[i];
            if cum >= p.top_p {
                cutoff = rank + 1;
                break;
            }
        }
        for &i in &idx[cutoff..] {
            probs[i] = 0.0;
        }
    }
    if p.min_p > 0.0 {
        let max = probs.iter().cloned().fold(0.0f64, f64::max);
        let thresh = p.min_p * max;
        for v in &mut probs {
            if *v < thresh {
                *v = 0.0;
            }
        }
    }
    let sum: f64 = probs.iter().sum();
    if sum > 0.0 {
        for v in &mut probs {
            *v /= sum;
        }
    }
    probs
}

/// Draw one token. Greedy when `temperature == 0`, ties to the lower index.
#[must_use]
pub fn sample(
    logits: &[f64],
    p: &SamplingParams,
    history: &[usize],
    mask: &[bool],
    seed: u64,
    step: u64,
) -> usize {
    if p.temperature == 0.0 {
        let mut best = usize::MAX;
        let mut best_v = f64::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if !mask[i] {
                continue;
            }
            if l > best_v {
                best_v = l;
                best = i;
            }
        }
        return if best == usize::MAX { 0 } else { best };
    }
    let probs = distribution(logits, p, history, mask);
    let mut rng = Philox::new(seed, SAMPLING_STREAM);
    rng.seek(step);
    let u = rng.next_f64();
    let mut cum = 0.0;
    for (i, pr) in probs.iter().enumerate() {
        cum += pr;
        if u < cum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_deterministic_and_respects_mask() {
        let logits = [1.0, 5.0, 3.0];
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample(&logits, &p, &[], &[true; 3], 1, 0), 1);
        assert_eq!(sample(&logits, &p, &[], &[true, false, true], 1, 0), 2);
    }

    #[test]
    fn distribution_sums_to_one() {
        let logits = [0.5, -1.0, 2.0, 0.1];
        let p = SamplingParams {
            top_k: 2,
            top_p: 0.9,
            ..Default::default()
        };
        let d = distribution(&logits, &p, &[], &[true; 4]);
        assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn top_k_zeroes_the_tail() {
        let logits = [0.5, -1.0, 2.0, 0.1];
        let p = SamplingParams {
            top_k: 1,
            ..Default::default()
        };
        let d = distribution(&logits, &p, &[], &[true; 4]);
        assert!((d[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn masked_tokens_never_get_probability() {
        let logits = [10.0, 0.0];
        let d = distribution(
            &logits,
            &SamplingParams::default(),
            &[],
            &[false, true],
        );
        assert!(d[0] < 1e-12);
        assert!((d[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sampling_is_reproducible_for_a_seed() {
        let logits = [0.2, 0.4, 0.9, -0.3];
        let p = SamplingParams::default();
        let a: Vec<usize> = (0..50)
            .map(|s| sample(&logits, &p, &[], &[true; 4], 7, s))
            .collect();
        let b: Vec<usize> = (0..50)
            .map(|s| sample(&logits, &p, &[], &[true; 4], 7, s))
            .collect();
        assert_eq!(a, b);
    }

    #[test]
    fn repetition_penalty_lowers_seen_tokens() {
        let logits = [2.0, 2.0];
        let p = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let d = distribution(&logits, &p, &[0], &[true; 2]);
        assert!(d[0] < d[1]);
    }
}
