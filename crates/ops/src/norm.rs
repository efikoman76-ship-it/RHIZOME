//! RMSNorm and per-head group norm (SPEC.md §3.1).

/// Default RMSNorm epsilon.
pub const RMSNORM_EPS: f64 = 1e-6;

/// RMSNorm with a learned per-channel scale.
#[must_use]
pub fn rmsnorm(x: &[f64], scale: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter()
        .zip(scale)
        .map(|(v, s)| v * inv * s)
        .collect()
}

/// Adjoint of [`rmsnorm`] with respect to the input.
#[must_use]
pub fn rmsnorm_backward_input(x: &[f64], scale: &[f64], eps: f64, grad_out: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    let dot: f64 = x
        .iter()
        .zip(scale)
        .zip(grad_out)
        .map(|((xi, si), gi)| gi * si * xi)
        .sum();
    x.iter()
        .zip(scale)
        .zip(grad_out)
        .map(|((xi, si), gi)| gi * si * inv - xi * inv.powi(3) * dot / n)
        .collect()
}

/// Adjoint of [`rmsnorm`] with respect to the learned scale.
#[must_use]
pub fn rmsnorm_backward_scale(x: &[f64], eps: f64, grad_out: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(grad_out).map(|(xi, gi)| gi * xi * inv).collect()
}

/// RMSNorm applied independently to each head of a `heads x head_dim` layout.
#[must_use]
pub fn groupnorm_heads(x: &[f64], heads: usize, scale: &[f64], eps: f64) -> Vec<f64> {
    if heads == 0 || x.len() % heads != 0 {
        return Vec::new();
    }
    let dh = x.len() / heads;
    let mut out = Vec::with_capacity(x.len());
    for h in 0..heads {
        let slice = &x[h * dh..(h + 1) * dh];
        out.extend(rmsnorm(slice, &scale[..dh], eps));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck;

    #[test]
    fn scale_invariance_up_to_epsilon() {
        let x = [1.0, -2.0, 3.0, 0.5];
        let s = [1.0; 4];
        let a = rmsnorm(&x, &s, 1e-12);
        let scaled: Vec<f64> = x.iter().map(|v| v * 10.0).collect();
        let b = rmsnorm(&scaled, &s, 1e-12);
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-9);
        }
    }

    #[test]
    fn input_gradient_matches_finite_difference() {
        let x = vec![0.7, -1.3, 2.0, 0.2];
        let s = vec![1.1, 0.9, 1.0, 1.3];
        let w = [0.3, -0.8, 1.5, 0.4];
        let analytic = rmsnorm_backward_input(&x, &s, RMSNORM_EPS, &w);
        let r = gradcheck::check(
            |v| {
                rmsnorm(v, &s, RMSNORM_EPS)
                    .iter()
                    .zip(&w)
                    .map(|(a, b)| a * b)
                    .sum()
            },
            &x,
            &analytic,
        );
        assert!(r.passes(1e-6), "{r:?}");
    }

    #[test]
    fn scale_gradient_matches_finite_difference() {
        let x = vec![0.7, -1.3, 2.0, 0.2];
        let s = vec![1.1, 0.9, 1.0, 1.3];
        let w = [0.3, -0.8, 1.5, 0.4];
        let analytic = rmsnorm_backward_scale(&x, RMSNORM_EPS, &w);
        let r = gradcheck::check(
            |v| {
                rmsnorm(&x, v, RMSNORM_EPS)
                    .iter()
                    .zip(&w)
                    .map(|(a, b)| a * b)
                    .sum()
            },
            &s,
            &analytic,
        );
        assert!(r.passes(1e-6), "{r:?}");
    }

    #[test]
    fn heads_are_normalised_independently() {
        let x = [1.0, 1.0, 100.0, 100.0];
        let s = [1.0, 1.0];
        let y = groupnorm_heads(&x, 2, &s, 1e-12);
        assert!((y[0] - y[2]).abs() < 1e-6);
    }
}
