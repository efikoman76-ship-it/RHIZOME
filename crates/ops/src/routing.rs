//! MoE routing and product-key memory selection (SPEC.md §3.4, §3.6).
//!
//! Selection is non-differentiable; the gate values are differentiable.

use crate::elementwise::sigmoid;

/// Router configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouterConfig {
    /// Number of routed experts.
    pub experts: usize,
    /// Number of groups the experts are split into.
    pub groups: usize,
    /// Experts selected per token.
    pub top_k: usize,
}

impl RouterConfig {
    /// Experts per group.
    #[must_use]
    pub const fn per_group(&self) -> usize {
        if self.groups == 0 {
            0
        } else {
            self.experts / self.groups
        }
    }
}

/// One routing decision for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    /// Selected expert indices, ascending.
    pub experts: Vec<usize>,
    /// Gate values, aligned with `experts`, summing to one.
    pub gates: Vec<f64>,
    /// Selected group indices, ascending (at most two).
    pub groups: Vec<usize>,
    /// Raw affinities `s_e` for every expert (used by the balance loss).
    pub affinities: Vec<f64>,
}

/// Affinity `s_e = sigmoid(clamp(logit, -30, 30))`.
#[must_use]
pub fn affinities(logits: &[f64]) -> Vec<f64> {
    logits.iter().map(|l| sigmoid(l.clamp(-30.0, 30.0))).collect()
}

/// Group-limited top-k routing with deterministic tie-breaks.
///
/// Ties are broken toward the lower index, both for experts and groups.
#[must_use]
pub fn router_topk_grouped(cfg: &RouterConfig, logits: &[f64], bias: &[f64]) -> Routing {
    let s = affinities(logits);
    let sel: Vec<f64> = s.iter().zip(bias).map(|(a, b)| a + b).collect();
    let per_group = cfg.per_group();

    // Group score = sum of its top-3 selection scores.
    let mut group_scores: Vec<(usize, f64)> = (0..cfg.groups)
        .map(|g| {
            let mut vals: Vec<f64> = sel[g * per_group..(g + 1) * per_group].to_vec();
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
            let take = vals.len().min(3);
            (g, vals[..take].iter().sum::<f64>())
        })
        .collect();
    group_scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut groups: Vec<usize> = group_scores
        .iter()
        .take(cfg.groups.min(2))
        .map(|(g, _)| *g)
        .collect();
    groups.sort_unstable();

    let mut candidates: Vec<(usize, f64)> = groups
        .iter()
        .flat_map(|&g| (g * per_group..(g + 1) * per_group).map(|e| (e, sel[e])))
        .collect();
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut experts: Vec<usize> = candidates
        .iter()
        .take(cfg.top_k.min(candidates.len()))
        .map(|(e, _)| *e)
        .collect();
    experts.sort_unstable();

    let raw: Vec<f64> = experts.iter().map(|&e| s[e]).collect();
    let sum: f64 = raw.iter().sum();
    let gates = if sum > 0.0 {
        raw.iter().map(|v| v / sum).collect()
    } else {
        vec![1.0 / experts.len().max(1) as f64; experts.len()]
    };

    Routing {
        experts,
        gates,
        groups,
        affinities: s,
    }
}

/// Loss-free balancing update of the per-expert selection biases.
pub fn update_router_bias(bias: &mut [f64], load: &[usize], gamma: f64) {
    let n = load.len() as f64;
    if n == 0.0 {
        return;
    }
    let mean = load.iter().sum::<usize>() as f64 / n;
    for (b, &l) in bias.iter_mut().zip(load) {
        let diff = l as f64 - mean;
        if diff > 0.0 {
            *b -= gamma;
        } else if diff < 0.0 {
            *b += gamma;
        }
    }
}

/// Sequence-wise auxiliary balance loss (SPEC.md §3.4).
#[must_use]
pub fn balance_loss(cfg: &RouterConfig, routings: &[Routing]) -> f64 {
    let t = routings.len();
    if t == 0 {
        return 0.0;
    }
    let mut counts = vec![0usize; cfg.experts];
    let mut p = vec![0.0f64; cfg.experts];
    for r in routings {
        for &e in &r.experts {
            counts[e] += 1;
        }
        for (e, a) in r.affinities.iter().enumerate() {
            p[e] += a;
        }
    }
    let denom = (cfg.top_k * t) as f64;
    (0..cfg.experts)
        .map(|e| {
            let f = cfg.experts as f64 / denom * counts[e] as f64;
            f * (p[e] / t as f64)
        })
        .sum()
}

/// Product-key top-k selection over two key tables.
///
/// Returns the `(slot, score)` pairs of the global top-k, with ties broken
/// toward the lower slot index.
#[must_use]
pub fn pkm_topk_product(
    scores1: &[f64],
    scores2: &[f64],
    per_table_k: usize,
    final_k: usize,
) -> Vec<(usize, f64)> {
    let pick = |s: &[f64]| -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = s.iter().cloned().enumerate().collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        v.truncate(per_table_k.min(s.len()));
        v
    };
    let t1 = pick(scores1);
    let t2 = pick(scores2);
    let n2 = scores2.len();
    let mut cands: Vec<(usize, f64)> = Vec::with_capacity(t1.len() * t2.len());
    for (i, si) in &t1 {
        for (j, sj) in &t2 {
            cands.push((i * n2 + j, si + sj));
        }
    }
    cands.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    cands.truncate(final_k.min(cands.len()));
    cands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Philox;

    fn zeros(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    fn cfg() -> RouterConfig {
        RouterConfig {
            experts: 8,
            groups: 4,
            top_k: 2,
        }
    }

    #[test]
    fn gates_sum_to_one_and_respect_k() {
        let mut r = Philox::new(1, 1);
        let c = cfg();
        for _ in 0..200 {
            let logits: Vec<f64> = (0..c.experts).map(|_| r.next_normal() * 3.0).collect();
            let bias = vec![0.0; c.experts];
            let out = router_topk_grouped(&c, &logits, &bias);
            assert_eq!(out.experts.len(), c.top_k);
            assert!(out.groups.len() <= 2);
            assert!((out.gates.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(out.gates.iter().all(|g| *g >= 0.0));
        }
    }

    #[test]
    fn selection_is_confined_to_two_groups() {
        let mut r = Philox::new(2, 2);
        let c = RouterConfig {
            experts: 12,
            groups: 4,
            top_k: 3,
        };
        for _ in 0..200 {
            let logits: Vec<f64> = (0..c.experts).map(|_| r.next_normal() * 2.0).collect();
            let out = router_topk_grouped(&c, &logits, &zeros(c.experts));
            for e in &out.experts {
                let g = e / c.per_group();
                assert!(out.groups.contains(&g));
            }
        }
    }

    #[test]
    fn ties_break_toward_lower_index() {
        let c = RouterConfig {
            experts: 4,
            groups: 2,
            top_k: 2,
        };
        let out = router_topk_grouped(&c, &[0.0; 4], &[0.0; 4]);
        assert_eq!(out.experts, vec![0, 1]);
        assert_eq!(out.groups, vec![0, 1]);
    }

    #[test]
    fn bias_affects_selection_but_not_gates() {
        let c = cfg();
        let logits = vec![0.1, 0.2, 0.3, 0.05, 0.0, 0.0, 0.0, 0.0];
        let mut bias = vec![0.0; 8];
        bias[6] = 5.0;
        bias[7] = 5.0;
        let out = router_topk_grouped(&c, &logits, &bias);
        assert!(out.experts.contains(&6) || out.experts.contains(&7));
        let s = affinities(&logits);
        let sum: f64 = out.experts.iter().map(|&e| s[e]).sum();
        for (g, &e) in out.gates.iter().zip(&out.experts) {
            assert!((g - s[e] / sum).abs() < 1e-12);
        }
    }

    #[test]
    fn bias_update_moves_toward_balance() {
        let mut bias = vec![0.0; 4];
        update_router_bias(&mut bias, &[10, 0, 5, 5], 1e-3);
        assert!(bias[0] < 0.0);
        assert!(bias[1] > 0.0);
        assert!((bias[2]).abs() < 1e-15);
    }

    #[test]
    fn balance_loss_is_lower_when_balanced() {
        let c = RouterConfig {
            experts: 4,
            groups: 2,
            top_k: 2,
        };
        let balanced: Vec<Routing> = (0..4)
            .map(|i| Routing {
                experts: vec![i % 4, (i + 1) % 4],
                gates: vec![0.5, 0.5],
                groups: vec![0, 1],
                affinities: vec![0.25; 4],
            })
            .collect();
        let skewed: Vec<Routing> = (0..4)
            .map(|_| Routing {
                experts: vec![0, 1],
                gates: vec![0.5, 0.5],
                groups: vec![0],
                affinities: vec![0.25; 4],
            })
            .collect();
        assert!(balance_loss(&c, &balanced) < balance_loss(&c, &skewed));
    }

    #[test]
    fn pkm_selection_matches_brute_force() {
        let mut r = Philox::new(3, 3);
        let s1: Vec<f64> = (0..32).map(|_| r.next_normal()).collect();
        let s2: Vec<f64> = (0..32).map(|_| r.next_normal()).collect();
        let got = pkm_topk_product(&s1, &s2, 8, 4);
        let mut all: Vec<(usize, f64)> = Vec::new();
        for (i, a) in s1.iter().enumerate() {
            for (j, b) in s2.iter().enumerate() {
                all.push((i * 32 + j, a + b));
            }
        }
        all.sort_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(x.0.cmp(&y.0))
        });
        for (g, w) in got.iter().zip(&all[..4]) {
            assert_eq!(g.0, w.0);
            assert!((g.1 - w.1).abs() < 1e-12);
        }
    }
}
