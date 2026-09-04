//! Dense and grouped linear algebra reference kernels.

/// Row-major matrix with `rows x cols` f64 elements.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Row-major data of length `rows * cols`.
    pub data: Vec<f64>,
}

impl Matrix {
    /// Create a zero matrix.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a matrix from row-major data, or `None` on a length mismatch.
    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Option<Self> {
        (data.len() == rows * cols).then_some(Matrix { rows, cols, data })
    }

    /// Element accessor.
    #[must_use]
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Mutable element accessor.
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.at(r, c));
            }
        }
        out
    }
}

/// `y = x · Wᵀ`, where `x` is `m x k` and `w` is `n x k`.
#[must_use]
pub fn linear(x: &Matrix, w: &Matrix) -> Option<Matrix> {
    if x.cols != w.cols {
        return None;
    }
    let mut out = Matrix::zeros(x.rows, w.rows);
    for i in 0..x.rows {
        for j in 0..w.rows {
            let mut acc = 0.0;
            for p in 0..x.cols {
                acc += x.at(i, p) * w.at(j, p);
            }
            out.set(i, j, acc);
        }
    }
    Some(out)
}

/// Adjoint of [`linear`]: returns `(grad_x, grad_w)`.
#[must_use]
pub fn linear_backward(x: &Matrix, w: &Matrix, grad_y: &Matrix) -> (Matrix, Matrix) {
    let mut gx = Matrix::zeros(x.rows, x.cols);
    let mut gw = Matrix::zeros(w.rows, w.cols);
    for i in 0..x.rows {
        for j in 0..w.rows {
            let g = grad_y.at(i, j);
            for p in 0..x.cols {
                gx.data[i * x.cols + p] += g * w.at(j, p);
                gw.data[j * w.cols + p] += g * x.at(i, p);
            }
        }
    }
    (gx, gw)
}

/// Grouped GEMM with variable group sizes (dropless MoE).
///
/// `group_sizes[g]` rows of `x` are multiplied by `weights[g]`.
#[must_use]
pub fn grouped_linear(x: &Matrix, group_sizes: &[usize], weights: &[Matrix]) -> Option<Matrix> {
    if group_sizes.len() != weights.len() {
        return None;
    }
    if group_sizes.iter().sum::<usize>() != x.rows {
        return None;
    }
    let n = weights.first().map_or(0, |w| w.rows);
    if weights.iter().any(|w| w.rows != n || w.cols != x.cols) {
        return None;
    }
    let mut out = Matrix::zeros(x.rows, n);
    let mut row = 0usize;
    for (g, &sz) in group_sizes.iter().enumerate() {
        for _ in 0..sz {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..x.cols {
                    acc += x.at(row, p) * weights[g].at(j, p);
                }
                out.set(row, j, acc);
            }
            row += 1;
        }
    }
    Some(out)
}

/// Five-iteration Newton–Schulz orthogonalisation used by Muon.
#[must_use]
pub fn muon_newton_schulz(g: &Matrix, iters: usize) -> Matrix {
    let (a, b, c) = (3.4445, -4.7750, 2.0315);
    let transposed = g.rows > g.cols;
    let mut x = if transposed { g.transpose() } else { g.clone() };
    let norm = x.data.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    for v in &mut x.data {
        *v /= norm;
    }
    for _ in 0..iters {
        let xt = x.transpose();
        let Some(a_mat) = matmul(&x, &xt) else { break };
        let Some(b_mat) = matmul(&a_mat, &x) else { break };
        let Some(a2) = matmul(&a_mat, &a_mat) else {
            break;
        };
        let Some(c_mat) = matmul(&a2, &x) else { break };
        for i in 0..x.data.len() {
            x.data[i] = a * x.data[i] + b * b_mat.data[i] + c * c_mat.data[i];
        }
    }
    if transposed {
        x.transpose()
    } else {
        x
    }
}

/// Plain `m x k` by `k x n` matrix product.
#[must_use]
pub fn matmul(a: &Matrix, b: &Matrix) -> Option<Matrix> {
    if a.cols != b.rows {
        return None;
    }
    let mut out = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for p in 0..a.cols {
            let av = a.at(i, p);
            for j in 0..b.cols {
                out.data[i * b.cols + j] += av * b.at(p, j);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradcheck;
    use crate::rng::Philox;

    fn random_matrix(rows: usize, cols: usize, stream: u64) -> Matrix {
        let mut rng = Philox::new(42, stream);
        Matrix {
            rows,
            cols,
            data: (0..rows * cols).map(|_| rng.next_normal()).collect(),
        }
    }

    #[test]
    fn linear_matches_manual_product() {
        let x = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("shape");
        let w = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("shape");
        let y = linear(&x, &w).expect("compatible");
        assert_eq!(y.data, vec![1.0, 2.0]);
    }

    #[test]
    fn linear_gradients_match_finite_difference() {
        let x = random_matrix(3, 4, 1);
        let w = random_matrix(2, 4, 2);
        let gy = Matrix {
            rows: 3,
            cols: 2,
            data: vec![1.0; 6],
        };
        let (gx, gw) = linear_backward(&x, &w, &gy);
        let r = gradcheck::check(
            |v| {
                let xm = Matrix::from_vec(3, 4, v.to_vec()).expect("shape");
                linear(&xm, &w).expect("shape").data.iter().sum()
            },
            &x.data,
            &gx.data,
        );
        assert!(r.passes(1e-6), "grad x: {r:?}");
        let r = gradcheck::check(
            |v| {
                let wm = Matrix::from_vec(2, 4, v.to_vec()).expect("shape");
                linear(&x, &wm).expect("shape").data.iter().sum()
            },
            &w.data,
            &gw.data,
        );
        assert!(r.passes(1e-6), "grad w: {r:?}");
    }

    #[test]
    fn grouped_linear_equals_per_group_linear() {
        let x = random_matrix(5, 3, 3);
        let w0 = random_matrix(2, 3, 4);
        let w1 = random_matrix(2, 3, 5);
        let out = grouped_linear(&x, &[2, 3], &[w0.clone(), w1.clone()]).expect("valid");
        let head = Matrix::from_vec(2, 3, x.data[..6].to_vec()).expect("shape");
        let tail = Matrix::from_vec(3, 3, x.data[6..].to_vec()).expect("shape");
        let a = linear(&head, &w0).expect("shape");
        let b = linear(&tail, &w1).expect("shape");
        let mut expected = a.data;
        expected.extend(b.data);
        for (p, q) in out.data.iter().zip(&expected) {
            assert!((p - q).abs() < 1e-12);
        }
    }

    #[test]
    fn grouped_linear_rejects_bad_group_sizes() {
        let x = random_matrix(4, 3, 6);
        let w = random_matrix(2, 3, 7);
        assert!(grouped_linear(&x, &[3], &[w]).is_none());
    }

    #[test]
    fn newton_schulz_produces_near_orthogonal_rows() {
        let g = random_matrix(4, 8, 8);
        let o = muon_newton_schulz(&g, 5);
        let gram = matmul(&o, &o.transpose()).expect("shape");
        for i in 0..4 {
            for j in 0..4 {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram.at(i, j) - target).abs() < 0.35,
                    "gram[{i}][{j}] = {}",
                    gram.at(i, j)
                );
            }
        }
    }
}
