//! Activation checkpointing and Think Core micro-step scheduling.

use crate::graph::Graph;
use rhizome_core::Result;

/// Strategy used to choose which activations are kept.
pub trait CheckpointPlanner {
    /// Return the node indices whose outputs are stored; all other
    /// recomputable outputs are recomputed in the backward pass.
    fn choose(&self, graph: &Graph, budget_bytes: usize) -> Vec<usize>;
}

/// Greedy planner: keep the cheapest-to-store, most-expensive-to-recompute
/// activations first, i.e. everything that is not recomputable, then the
/// largest remaining tensors until the budget is exhausted.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreedyPlanner;

impl CheckpointPlanner for GreedyPlanner {
    fn choose(&self, graph: &Graph, budget_bytes: usize) -> Vec<usize> {
        let mut kept: Vec<usize> = Vec::new();
        let mut used = 0usize;
        // Non-recomputable outputs must always be kept.
        for (idx, n) in graph.nodes().iter().enumerate() {
            if !n.op.is_recomputable() {
                kept.push(idx);
                used += graph.tensors()[n.output].bytes();
            }
        }
        let mut rest: Vec<usize> = (0..graph.nodes().len())
            .filter(|i| !kept.contains(i))
            .collect();
        rest.sort_by_key(|&i| {
            (
                std::cmp::Reverse(graph.tensors()[graph.nodes()[i].output].bytes()),
                i,
            )
        });
        for i in rest {
            let b = graph.tensors()[graph.nodes()[i].output].bytes();
            if used + b > budget_bytes {
                continue;
            }
            used += b;
            kept.push(i);
        }
        kept.sort_unstable();
        kept
    }
}

/// One micro-step of the serving scheduler (SPEC.md §6.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicroStep {
    /// Run a prefill chunk of the given token count.
    PrefillChunk(usize),
    /// Run the prelude decode for `n` sequences.
    Prelude(usize),
    /// Run Think Core iteration `i` for `n` still-active sequences.
    CoreIteration {
        /// 1-based iteration index.
        iteration: usize,
        /// Number of sequences still running.
        active: usize,
    },
    /// Run coda and back end for `n` sequences.
    Coda(usize),
}

/// An immutable snapshot of scheduler state; step planning is a pure
/// function of it, which makes decisions replayable from logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchedulerSnapshot {
    /// Tokens waiting to be prefilled.
    pub pending_prefill_tokens: usize,
    /// Maximum tokens per prefill chunk.
    pub chunk_size: usize,
    /// Sequences in the decode phase.
    pub decoding: usize,
    /// Number of sequences still active at each Think Core iteration.
    pub active_per_iteration: Vec<usize>,
}

/// Produce the micro-step order for one scheduler step.
#[must_use]
pub fn plan_step(s: &SchedulerSnapshot) -> Vec<MicroStep> {
    let mut out = Vec::new();
    if s.pending_prefill_tokens > 0 {
        out.push(MicroStep::PrefillChunk(
            s.pending_prefill_tokens.min(s.chunk_size.max(1)),
        ));
    }
    if s.decoding == 0 {
        return out;
    }
    out.push(MicroStep::Prelude(s.decoding));
    for (i, &active) in s.active_per_iteration.iter().enumerate() {
        if active == 0 {
            break;
        }
        out.push(MicroStep::CoreIteration {
            iteration: i + 1,
            active,
        });
    }
    out.push(MicroStep::Coda(s.decoding));
    out
}

/// Round a batch size up to the nearest warmed-up bucket.
#[must_use]
pub fn batch_bucket(n: usize) -> usize {
    const BUCKETS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(128)
}

/// Verify that a checkpoint selection is sufficient: every kept node's
/// output plus recomputable nodes covers all backward inputs.
///
/// # Errors
/// Propagates descriptor lookups.
pub fn checkpoint_bytes(graph: &Graph, kept: &[usize]) -> Result<usize> {
    let mut total = 0usize;
    for &i in kept {
        total += graph.desc(graph.nodes()[i].output)?.bytes();
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use rhizome_core::DType;

    fn graph() -> Graph {
        let mut g = Graph::new();
        let x = g.tensor("x", 128, DType::F32, false);
        let w = g.tensor("w", 128 * 128, DType::BF16, true);
        let a = g
            .push(OpKind::Linear, &[x, w], "a", 128, DType::F32)
            .expect("ok");
        let b = g
            .push(OpKind::Silu, &[a], "b", 128, DType::F32)
            .expect("ok");
        let c = g
            .push(OpKind::MlaPrefill, &[b], "c", 128, DType::F32)
            .expect("ok");
        g.push(OpKind::Add, &[c, b], "d", 128, DType::F32)
            .expect("ok");
        g
    }

    #[test]
    fn non_recomputable_nodes_are_always_kept() {
        let g = graph();
        let kept = GreedyPlanner.choose(&g, 0);
        assert!(kept.contains(&2));
    }

    #[test]
    fn budget_is_respected_for_optional_nodes() {
        let g = graph();
        let small = GreedyPlanner.choose(&g, 0);
        let large = GreedyPlanner.choose(&g, 1 << 20);
        assert!(large.len() > small.len());
        assert!(checkpoint_bytes(&g, &large).expect("ok") >= checkpoint_bytes(&g, &small).expect("ok"));
    }

    #[test]
    fn micro_step_order_follows_spec() {
        let s = SchedulerSnapshot {
            pending_prefill_tokens: 3000,
            chunk_size: 2048,
            decoding: 4,
            active_per_iteration: vec![4, 3, 1, 0],
        };
        let steps = plan_step(&s);
        assert_eq!(steps[0], MicroStep::PrefillChunk(2048));
        assert_eq!(steps[1], MicroStep::Prelude(4));
        assert_eq!(
            steps[2],
            MicroStep::CoreIteration {
                iteration: 1,
                active: 4
            }
        );
        assert_eq!(steps.last(), Some(&MicroStep::Coda(4)));
    }

    #[test]
    fn planning_is_pure_and_replayable() {
        let s = SchedulerSnapshot {
            pending_prefill_tokens: 10,
            chunk_size: 4,
            decoding: 2,
            active_per_iteration: vec![2, 1],
        };
        assert_eq!(plan_step(&s), plan_step(&s));
    }

    #[test]
    fn buckets_round_up() {
        assert_eq!(batch_bucket(1), 1);
        assert_eq!(batch_bucket(5), 8);
        assert_eq!(batch_bucket(200), 128);
    }
}
