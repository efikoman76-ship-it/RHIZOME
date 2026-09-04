//! Losses and the halt head (SPEC.md §3.5, §3.7).

use crate::elementwise::sigmoid;

/// Default z-loss weight.
pub const Z_LOSS_WEIGHT: f64 = 1e-4;

/// Fused cross-entropy with a z-loss regulariser.
///
/// Returns `(loss, d_loss/d_logits)`.
#[must_use]
pub fn cross_entropy_zloss(logits: &[f64], target: usize, z_weight: f64) -> (f64, Vec<f64>) {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let logz = m + sum.ln();
    let ce = logz - logits[target];
    let zl = z_weight * logz * logz;
    let mut grad = Vec::with_capacity(logits.len());
    for (i, e) in exps.iter().enumerate() {
        let p = e / sum;
        let mut g = p - if i == target { 1.0 } else { 0.0 };
        g += 2.0 * z_weight * logz * p;
        grad.push(g);
    }
    (ce + zl, grad)
}

/// Binary cross-entropy used to train the halt head.
#[must_use]
pub fn halt_bce(logit: f64, target: f64) -> (f64, f64) {
    let p = sigmoid(logit);
    let eps = 1e-12;
    let loss = -(target * (p + eps).ln() + (1.0 - target) * (1.0 - p + eps).ln());
    (loss, p - target)
}

/// Expected Think Core depth `E[depth] = sum_i prod_{j<i} (1 - p_j)`.
#[must_use]
pub fn expected_depth(p_halt: &[f64]) -> f64 {
    let mut survive = 1.0;
    let mut e = 0.0;
    for &p in p_halt {
        e += survive;
        survive *= 1.0 - p;
    }
    e
}

/// Gradient of [`expected_depth`] with respect to each `p_halt[i]`.
#[must_use]
pub fn expected_depth_grad(p_halt: &[f64]) -> Vec<f64> {
    let n = p_halt.len();
    let mut grad = vec![0.0; n];
    for i in 0..n {
        // d/dp_i sum_{k>i} prod_{j<k}(1-p_j)
        let mut acc = 0.0;
        for k in i + 1..n {
            let mut prod = 1.0;
            for (j, p) in p_halt.iter().enumerate().take(k) {
                if j != i {
                    prod *= 1.0 - p;
                }
            }
            acc += prod;
        }
        grad[i] = -acc;
    }
    grad
}

/// Convergence-based halt target: relative change below `eps_c`.
#[must_use]
pub fn convergence_target(prev: &[f64], cur: &[f64], eps_c: f64) -> f64 {
    let num: f64 = prev
        .iter()
        .zip(cur)
        .map(|(a, b)| (b - a) * (b - a))
        .sum::<f64>()
        .sqrt();
    let den: f64 = cur.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    if num / den < eps_c { 1.0 } else { 0.0 }
}

/// Expected calibration error of the halt head over `bins` buckets.
#[must_use]
pub fn expected_calibration_error(probs: &[f64], labels: &[f64], bins: usize) -> f64 {
    if probs.is_empty() || bins == 0 {
        return 0.0;
    }
    let n = probs.len() as f64;
    let mut ece = 0.0;
    for b in 0..bins {
        let lo = b as f64 / bins as f64;
        let hi = (b + 1) as f64 / bins as f64;
        let idx: Vec<usize> = (0..probs.len())
            .filter(|&i| probs[i] >= lo && (probs[i] < hi || (b + 1 == bins && probs[i] <= hi)))
            .collect();
        if idx.is_empty() {
            continue;
        }
        let conf: f64 = idx.iter().map(|&i| probs[i]).sum::<f64>() / idx.len() as f64;
        let acc: f64 = idx.iter().map(|&i| labels[i]).sum::<f64>() / idx.len() as f64;
        ece += idx.len() as f64 / n * (conf - acc).abs();
    }
    ece
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck;

    #[test]
    fn cross_entropy_gradient_matches_finite_difference() {
        let logits = vec![0.3, -1.2, 2.0, 0.7];
        let (_, grad) = cross_entropy_zloss(&logits, 2, Z_LOSS_WEIGHT);
        let r = gradcheck::check(
            |v| cross_entropy_zloss(v, 2, Z_LOSS_WEIGHT).0,
            &logits,
            &grad,
        );
        assert!(r.passes(1e-6), "{r:?}");
    }

    #[test]
    fn perfect_prediction_has_near_zero_loss() {
        let logits = vec![-50.0, 50.0, -50.0];
        let (loss, _) = cross_entropy_zloss(&logits, 1, 0.0);
        assert!(loss < 1e-20, "loss {loss}");
    }

    #[test]
    fn uniform_loss_is_log_vocab() {
        let logits = vec![0.0; 8];
        let (loss, _) = cross_entropy_zloss(&logits, 3, 0.0);
        assert!((loss - 8f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn halt_bce_gradient_matches_finite_difference() {
        for target in [0.0, 1.0] {
            let x = vec![0.4];
            let (_, g) = halt_bce(x[0], target);
            let r = gradcheck::check(|v| halt_bce(v[0], target).0, &x, &[g]);
            assert!(r.passes(1e-5), "{r:?}");
        }
    }

    #[test]
    fn expected_depth_is_one_when_always_halting() {
        assert!((expected_depth(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-12);
        assert!((expected_depth(&[0.0, 0.0, 0.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn expected_depth_gradient_matches_finite_difference() {
        let p = vec![0.2, 0.5, 0.3, 0.7];
        let g = expected_depth_grad(&p);
        let r = gradcheck::check(expected_depth, &p, &g);
        assert!(r.passes(1e-6), "{r:?}");
    }

    #[test]
    fn convergence_target_fires_on_small_change() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.001, 2.001, 3.001];
        assert!((convergence_target(&a, &b, 0.05) - 1.0).abs() < 1e-12);
        let c = vec![2.0, 4.0, 6.0];
        assert!(convergence_target(&a, &c, 0.05) < 0.5);
    }

    #[test]
    fn calibrated_predictor_has_low_ece() {
        let probs: Vec<f64> = (0..100).map(|i| f64::from(i) / 100.0).collect();
        let labels: Vec<f64> = probs.iter().map(|p| if *p >= 0.5 { 1.0 } else { 0.0 }).collect();
        let perfect: Vec<f64> = labels.clone();
        assert!(expected_calibration_error(&perfect, &labels, 10) < 1e-9);
        assert!(expected_calibration_error(&probs, &labels, 10) > 0.1);
    }
}
