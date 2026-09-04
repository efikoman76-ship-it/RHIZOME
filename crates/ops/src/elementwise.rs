//! Elementwise ops and their adjoints.

/// Logistic sigmoid.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Derivative of [`sigmoid`].
#[must_use]
pub fn dsigmoid(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// SiLU / swish activation.
#[must_use]
pub fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Derivative of [`silu`].
#[must_use]
pub fn dsilu(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Numerically stable softplus.
#[must_use]
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Derivative of [`softplus`].
#[must_use]
pub fn dsoftplus(x: f64) -> f64 {
    sigmoid(x)
}

/// SwiGLU over a concatenated `[gate, up]` vector of even length.
///
/// # Panics
/// Never: odd-length inputs are rejected by returning an empty vector.
#[must_use]
pub fn swiglu(gate_up: &[f64]) -> Vec<f64> {
    if gate_up.len() % 2 != 0 {
        return Vec::new();
    }
    let h = gate_up.len() / 2;
    (0..h).map(|i| silu(gate_up[i]) * gate_up[h + i]).collect()
}

/// Adjoint of [`swiglu`].
#[must_use]
pub fn swiglu_backward(gate_up: &[f64], grad_out: &[f64]) -> Vec<f64> {
    let h = gate_up.len() / 2;
    let mut g = vec![0.0; gate_up.len()];
    for i in 0..h {
        let a = gate_up[i];
        let b = gate_up[h + i];
        g[i] = grad_out[i] * dsilu(a) * b;
        g[h + i] = grad_out[i] * silu(a);
    }
    g
}

/// Attention logit softcap: `cap * tanh(x / cap)`.
#[must_use]
pub fn softcap(x: f64, cap: f64) -> f64 {
    if cap <= 0.0 {
        x
    } else {
        cap * (x / cap).tanh()
    }
}

/// Derivative of [`softcap`].
#[must_use]
pub fn dsoftcap(x: f64, cap: f64) -> f64 {
    if cap <= 0.0 {
        1.0
    } else {
        let t = (x / cap).tanh();
        1.0 - t * t
    }
}

/// Numerically stable softmax over a slice.
#[must_use]
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let m = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return vec![0.0; x.len()];
    }
    let exps: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Adjoint of [`softmax`] given the output `y` and upstream gradient.
#[must_use]
pub fn softmax_backward(y: &[f64], grad_out: &[f64]) -> Vec<f64> {
    let dot: f64 = y.iter().zip(grad_out).map(|(a, b)| a * b).sum();
    y.iter()
        .zip(grad_out)
        .map(|(yi, gi)| yi * (gi - dot))
        .collect()
}

/// L2 normalisation with an optional rescale factor.
#[must_use]
pub fn l2norm(x: &[f64], scale: f64) -> Vec<f64> {
    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    x.iter().map(|v| v / n * scale).collect()
}

/// Apply decoupled RoPE to a `(pairs * 2)`-length vector at `pos`.
#[must_use]
pub fn rope_apply(x: &[f64], pos: usize, base: f64) -> Vec<f64> {
    let pairs = x.len() / 2;
    let mut out = x.to_vec();
    for p in 0..pairs {
        let theta = pos as f64 / base.powf(2.0 * p as f64 / x.len() as f64);
        let (s, c) = theta.sin_cos();
        let (a, b) = (x[2 * p], x[2 * p + 1]);
        out[2 * p] = a * c - b * s;
        out[2 * p + 1] = a * s + b * c;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck::central_difference;

    #[test]
    fn sigmoid_is_stable_for_large_magnitudes() {
        assert!(sigmoid(-1000.0) >= 0.0 && sigmoid(-1000.0) < 1e-300);
        assert!((sigmoid(1000.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn silu_gradient_matches_finite_difference() {
        for x in [-2.5, -0.3, 0.7, 3.1] {
            let fd = central_difference(|v| silu(v[0]), &[x], 0);
            assert!((fd - dsilu(x)).abs() < 1e-7, "x={x}");
        }
    }

    #[test]
    fn swiglu_gradient_matches_finite_difference() {
        let x = vec![0.4, -1.2, 0.9, 2.0];
        let g = vec![1.0, 1.0];
        let analytic = swiglu_backward(&x, &g);
        for i in 0..x.len() {
            let fd = central_difference(|v| swiglu(v).iter().sum::<f64>(), &x, i);
            assert!((fd - analytic[i]).abs() < 1e-7, "i={i}");
        }
    }

    #[test]
    fn softmax_sums_to_one_and_is_shift_invariant() {
        let a = softmax(&[1.0, 2.0, 3.0]);
        let b = softmax(&[101.0, 102.0, 103.0]);
        assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_gradient_matches_finite_difference() {
        let x = vec![0.3, -0.7, 1.1];
        let y = softmax(&x);
        let w = [0.5, -1.0, 2.0];
        let grad_out = w.to_vec();
        let analytic = softmax_backward(&y, &grad_out);
        for i in 0..x.len() {
            let fd = central_difference(
                |v| softmax(v).iter().zip(&w).map(|(a, b)| a * b).sum::<f64>(),
                &x,
                i,
            );
            assert!((fd - analytic[i]).abs() < 1e-7, "i={i}");
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let x = [0.3, -1.2, 0.5, 0.9];
        let y = rope_apply(&x, 7, 1e6);
        let nx: f64 = x.iter().map(|v| v * v).sum();
        let ny: f64 = y.iter().map(|v| v * v).sum();
        assert!((nx - ny).abs() < 1e-12);
    }

    #[test]
    fn softcap_is_bounded_and_monotone() {
        let cap = 30.0;
        let mut prev = f64::NEG_INFINITY;
        for i in -100..100 {
            let v = softcap(f64::from(i) * 5.0, cap);
            assert!(v.abs() <= cap);
            assert!(v > prev);
            prev = v;
        }
    }

    #[test]
    fn l2norm_produces_requested_length() {
        let y = l2norm(&[3.0, 4.0], 2.0);
        let n: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 2.0).abs() < 1e-12);
    }
}
