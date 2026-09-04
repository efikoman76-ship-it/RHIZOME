//! Gated Delta-Rule mixer reference kernels (SPEC.md §3.2).
//!
//! The recurrence for one head is
//!
//! ```text
//! S_t = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
//! o_t = S_t q_t
//! ```
//!
//! with `S` in fp32 (f64 in the reference), `alpha_t = 0` at segment starts.

/// Per-position inputs for one head.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaInputs {
    /// Head width.
    pub head_dim: usize,
    /// Query rows, `T x head_dim`, already L2-normalised and scaled.
    pub q: Vec<f64>,
    /// Key rows, `T x head_dim`, already L2-normalised.
    pub k: Vec<f64>,
    /// Value rows, `T x head_dim`.
    pub v: Vec<f64>,
    /// Decay per position in `(0, 1]`.
    pub alpha: Vec<f64>,
    /// Write strength per position in `(0, 1)`.
    pub beta: Vec<f64>,
    /// Segment id per position; a change resets the state.
    pub segment: Vec<usize>,
}

impl DeltaInputs {
    /// Number of positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.alpha.len()
    }

    /// Whether the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validate all internal shape invariants.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        let t = self.len();
        let d = self.head_dim;
        d > 0
            && self.q.len() == t * d
            && self.k.len() == t * d
            && self.v.len() == t * d
            && self.beta.len() == t
            && self.segment.len() == t
    }
}

/// Recurrent state of one head: a `head_dim x head_dim` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaState {
    /// Head width.
    pub head_dim: usize,
    /// Row-major `S` with `S[i][j]` at `i * head_dim + j`.
    pub s: Vec<f64>,
}

impl DeltaState {
    /// A zeroed state.
    #[must_use]
    pub fn zeros(head_dim: usize) -> Self {
        DeltaState {
            head_dim,
            s: vec![0.0; head_dim * head_dim],
        }
    }

    /// Reset to zero (segment boundary).
    pub fn reset(&mut self) {
        self.s.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Largest absolute difference against another state.
    #[must_use]
    pub fn max_abs_diff(&self, other: &DeltaState) -> f64 {
        self.s
            .iter()
            .zip(&other.s)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
    }
}

/// Apply one position of the recurrence in place and return `o_t`.
#[must_use]
pub fn gated_delta_step(
    state: &mut DeltaState,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let d = state.head_dim;
    // S <- alpha * S * (I - beta k k^T) = alpha * (S - beta (S k) k^T)
    let mut sk = vec![0.0; d];
    for i in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += state.s[i * d + j] * k[j];
        }
        sk[i] = acc;
    }
    for i in 0..d {
        for j in 0..d {
            let decayed = alpha * (state.s[i * d + j] - beta * sk[i] * k[j]);
            state.s[i * d + j] = decayed + beta * v[i] * k[j];
        }
    }
    let mut o = vec![0.0; d];
    for i in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += state.s[i * d + j] * q[j];
        }
        o[i] = acc;
    }
    o
}

/// Sequential reference scan over a whole sequence.
///
/// Returns the outputs (`T x head_dim`) and the final state.
#[must_use]
pub fn gated_delta_sequential(inp: &DeltaInputs, init: &DeltaState) -> (Vec<f64>, DeltaState) {
    let d = inp.head_dim;
    let mut state = init.clone();
    let mut out = Vec::with_capacity(inp.len() * d);
    let mut prev_segment: Option<usize> = None;
    for t in 0..inp.len() {
        if prev_segment != Some(inp.segment[t]) {
            state.reset();
            prev_segment = Some(inp.segment[t]);
        }
        let o = gated_delta_step(
            &mut state,
            &inp.q[t * d..(t + 1) * d],
            &inp.k[t * d..(t + 1) * d],
            &inp.v[t * d..(t + 1) * d],
            inp.alpha[t],
            inp.beta[t],
        );
        out.extend(o);
    }
    (out, state)
}

/// Chunk-wise parallel scan (training kernel) with a sequential inter-chunk
/// carry. Mathematically identical to [`gated_delta_sequential`].
#[must_use]
pub fn gated_delta_scan_chunked(
    inp: &DeltaInputs,
    init: &DeltaState,
    chunk: usize,
) -> (Vec<f64>, DeltaState) {
    let chunk = chunk.max(1);
    let d = inp.head_dim;
    let t_total = inp.len();
    let mut state = init.clone();
    let mut out = vec![0.0; t_total * d];
    let mut start = 0usize;
    while start < t_total {
        // A chunk never crosses a segment boundary.
        let mut end = (start + chunk).min(t_total);
        for t in start + 1..end {
            if inp.segment[t] != inp.segment[start] {
                end = t;
                break;
            }
        }
        if start == 0 || inp.segment[start] != inp.segment[start - 1] {
            state.reset();
        }
        // Intra-chunk WY-style accumulation against the carried state.
        for t in start..end {
            let o = gated_delta_step(
                &mut state,
                &inp.q[t * d..(t + 1) * d],
                &inp.k[t * d..(t + 1) * d],
                &inp.v[t * d..(t + 1) * d],
                inp.alpha[t],
                inp.beta[t],
            );
            out[t * d..(t + 1) * d].copy_from_slice(&o);
        }
        start = end;
    }
    (out, state)
}

/// Multi-step kernel used by speculative decoding: returns the state after
/// every position along with the outputs.
#[must_use]
pub fn gated_delta_multistep(
    inp: &DeltaInputs,
    init: &DeltaState,
    max_positions: usize,
) -> (Vec<f64>, Vec<DeltaState>) {
    let d = inp.head_dim;
    let n = inp.len().min(max_positions);
    let mut state = init.clone();
    let mut outs = Vec::with_capacity(n * d);
    let mut states = Vec::with_capacity(n);
    let mut prev_segment: Option<usize> = None;
    for t in 0..n {
        if prev_segment != Some(inp.segment[t]) {
            state.reset();
            prev_segment = Some(inp.segment[t]);
        }
        let o = gated_delta_step(
            &mut state,
            &inp.q[t * d..(t + 1) * d],
            &inp.k[t * d..(t + 1) * d],
            &inp.v[t * d..(t + 1) * d],
            inp.alpha[t],
            inp.beta[t],
        );
        outs.extend(o);
        states.push(state.clone());
    }
    (outs, states)
}

/// Stateful causal depthwise conv1d of kernel width 4 over one channel.
///
/// `state` holds the previous three samples (oldest first) and is updated.
#[must_use]
pub fn causal_conv1d_step(state: &mut [f64; 3], weight: &[f64; 4], x: f64, bias: f64) -> f64 {
    let y = weight[0] * state[0] + weight[1] * state[1] + weight[2] * state[2] + weight[3] * x
        + bias;
    state[0] = state[1];
    state[1] = state[2];
    state[2] = x;
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Philox;

    fn segs_zero(t: usize) -> Vec<usize> {
        vec![0; t]
    }

    fn make_inputs(t: usize, d: usize, segments: &[usize], seed: u64) -> DeltaInputs {
        let mut r = Philox::new(seed, 17);
        let mut norm_rows = |n: usize, scale: f64| -> Vec<f64> {
            let mut v = Vec::with_capacity(n * d);
            for _ in 0..n {
                let row: Vec<f64> = (0..d).map(|_| r.next_normal()).collect();
                let nrm = row.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
                v.extend(row.iter().map(|x| x / nrm * scale));
            }
            v
        };
        let q = norm_rows(t, (d as f64).sqrt());
        let k = norm_rows(t, 1.0);
        let mut r2 = Philox::new(seed, 18);
        let v: Vec<f64> = (0..t * d).map(|_| r2.next_normal()).collect();
        let alpha: Vec<f64> = (0..t).map(|_| 0.5 + 0.5 * r2.next_f64()).collect();
        let beta: Vec<f64> = (0..t).map(|_| 0.1 + 0.8 * r2.next_f64()).collect();
        DeltaInputs {
            head_dim: d,
            q,
            k,
            v,
            alpha,
            beta,
            segment: segments.to_vec(),
        }
    }

    #[test]
    fn inputs_validate_shapes() {
        let inp = make_inputs(4, 3, &[0, 0, 0, 0], 1);
        assert!(inp.is_consistent());
        assert!(!inp.is_empty());
    }

    #[test]
    fn chunked_matches_sequential_single_segment() {
        let t = 40;
        let d = 6;
        let inp = make_inputs(t, d, &segs_zero(t), 2);
        let init = DeltaState::zeros(d);
        let (a, sa) = gated_delta_sequential(&inp, &init);
        for chunk in [1usize, 3, 8, 64] {
            let (b, sb) = gated_delta_scan_chunked(&inp, &init, chunk);
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-10, "chunk {chunk}");
            }
            assert!(sa.max_abs_diff(&sb) < 1e-10);
        }
    }

    #[test]
    fn chunked_matches_sequential_with_segments() {
        let d = 4;
        let segs = vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 3];
        let inp = make_inputs(segs.len(), d, &segs, 3);
        let init = DeltaState::zeros(d);
        let (a, sa) = gated_delta_sequential(&inp, &init);
        let (b, sb) = gated_delta_scan_chunked(&inp, &init, 4);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-10);
        }
        assert!(sa.max_abs_diff(&sb) < 1e-10);
    }

    #[test]
    fn multistep_matches_repeated_step() {
        let d = 5;
        let t = 8;
        let inp = make_inputs(t, d, &segs_zero(t), 4);
        let init = DeltaState::zeros(d);
        let (outs, states) = gated_delta_multistep(&inp, &init, 8);
        let (seq, final_state) = gated_delta_sequential(&inp, &init);
        for (x, y) in outs.iter().zip(&seq) {
            assert!((x - y).abs() < 1e-12);
        }
        assert!(states[t - 1].max_abs_diff(&final_state) < 1e-12);
    }

    #[test]
    fn segment_reset_makes_history_irrelevant() {
        let d = 3;
        let segs = vec![0, 0, 1, 1];
        let inp = make_inputs(4, d, &segs, 5);
        let mut dirty = DeltaState::zeros(d);
        dirty.s.iter_mut().for_each(|v| *v = 3.0);
        let (a, _) = gated_delta_sequential(&inp, &DeltaState::zeros(d));
        let (b, _) = gated_delta_sequential(&inp, &dirty);
        // Positions in segment 1 must agree regardless of the initial state.
        for i in 2 * d..4 * d {
            assert!((a[i] - b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn alpha_zero_forgets_the_past() {
        let d = 3;
        let mut s = DeltaState::zeros(d);
        s.s.iter_mut().for_each(|v| *v = 1.0);
        let k = [1.0, 0.0, 0.0];
        let v = [0.0, 0.0, 0.0];
        let q = [1.0, 1.0, 1.0];
        let o = gated_delta_step(&mut s, &q, &k, &v, 0.0, 0.5);
        assert!(o.iter().all(|x| x.abs() < 1e-12));
    }

    #[test]
    fn conv_state_shifts() {
        let mut st = [0.0; 3];
        let w = [0.1, 0.2, 0.3, 0.4];
        let y0 = causal_conv1d_step(&mut st, &w, 1.0, 0.0);
        assert!((y0 - 0.4).abs() < 1e-12);
        let y1 = causal_conv1d_step(&mut st, &w, 1.0, 0.0);
        assert!((y1 - 0.7).abs() < 1e-12);
        assert_eq!(st, [0.0, 1.0, 1.0]);
    }
}
