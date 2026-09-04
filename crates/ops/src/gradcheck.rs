//! Finite-difference gradient checking in f64 (SPEC.md §4, R3).

/// Central-difference derivative of `f` with respect to `x[i]` (h = 1e-5).
pub fn central_difference<F>(mut f: F, x: &[f64], i: usize) -> f64
where
    F: FnMut(&[f64]) -> f64,
{
    let h = 1e-5;
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[i] += h;
    xm[i] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

/// Outcome of a full gradient check.
#[derive(Debug, Clone, PartialEq)]
pub struct GradCheckReport {
    /// Largest relative error observed.
    pub max_rel_err: f64,
    /// Index at which the maximum occurred.
    pub worst_index: usize,
}

impl GradCheckReport {
    /// Whether the check passes at the given relative tolerance.
    #[must_use]
    pub fn passes(&self, rtol: f64) -> bool {
        self.max_rel_err <= rtol
    }
}

/// Compare an analytic gradient against central differences of `f`.
pub fn check<F>(mut f: F, x: &[f64], analytic: &[f64]) -> GradCheckReport
where
    F: FnMut(&[f64]) -> f64,
{
    let mut max_rel_err = 0.0;
    let mut worst_index = 0;
    for i in 0..x.len() {
        let fd = central_difference(&mut f, x, i);
        let denom = fd.abs().max(analytic[i].abs()).max(1e-6);
        let rel = (fd - analytic[i]).abs() / denom;
        if rel > max_rel_err {
            max_rel_err = rel;
            worst_index = i;
        }
    }
    GradCheckReport {
        max_rel_err,
        worst_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_correct_gradient() {
        let x = vec![0.3, -1.7, 2.2];
        let analytic: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let r = check(|v| v.iter().map(|a| a * a).sum(), &x, &analytic);
        assert!(r.passes(1e-6), "{r:?}");
    }

    #[test]
    fn detects_an_incorrect_gradient() {
        let x = vec![0.3, -1.7];
        let analytic = vec![0.0, 0.0];
        let r = check(|v| v.iter().map(|a| a * a).sum(), &x, &analytic);
        assert!(!r.passes(1e-6));
        assert_eq!(r.worst_index, 1);
    }
}
