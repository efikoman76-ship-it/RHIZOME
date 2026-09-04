//! Liveness analysis and arena offset assignment.

use crate::graph::{Graph, TensorId};
use rhizome_core::{Error, Result};

/// The live range of one tensor, in node indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    /// Tensor the interval belongs to.
    pub tensor: TensorId,
    /// First node index at which the tensor is live.
    pub start: usize,
    /// Last node index at which the tensor is live (inclusive).
    pub end: usize,
    /// Byte offset assigned inside the arena.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
}

impl Interval {
    /// Whether two intervals overlap in time.
    #[must_use]
    pub fn overlaps_time(&self, other: &Interval) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Whether two intervals overlap in memory.
    #[must_use]
    pub fn overlaps_memory(&self, other: &Interval) -> bool {
        self.offset < other.offset + other.size && other.offset < self.offset + self.size
    }
}

/// The result of arena planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaPlan {
    /// Assigned intervals, ordered by tensor id.
    pub intervals: Vec<Interval>,
    /// Total arena size in bytes.
    pub total_bytes: usize,
}

impl ArenaPlan {
    /// Verify that no two temporally overlapping tensors share memory.
    ///
    /// # Errors
    /// Returns [`Error::Shape`] naming the first conflicting pair.
    pub fn verify(&self) -> Result<()> {
        for (i, a) in self.intervals.iter().enumerate() {
            for b in &self.intervals[i + 1..] {
                if a.overlaps_time(b) && a.overlaps_memory(b) {
                    return Err(Error::Shape(format!(
                        "tensors {} and {} are live together but share arena bytes",
                        a.tensor, b.tensor
                    )));
                }
            }
        }
        Ok(())
    }

    /// Peak bytes required if every tensor were given its own allocation.
    #[must_use]
    pub fn unplanned_bytes(&self) -> usize {
        self.intervals.iter().map(|i| i.size).sum()
    }
}

/// Alignment of every arena allocation, in bytes.
pub const ALIGN: usize = 256;

fn align_up(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

/// Plan arena offsets with a greedy best-fit-by-size strategy.
///
/// Parameters are excluded: they live in the weight mmap, not the arena.
///
/// # Errors
/// Propagates descriptor lookup failures.
pub fn plan(graph: &Graph) -> Result<ArenaPlan> {
    let n = graph.tensors().len();
    let mut start = vec![usize::MAX; n];
    let mut end = vec![0usize; n];
    let mut used = vec![false; n];

    for (idx, node) in graph.nodes().iter().enumerate() {
        for &i in &node.inputs {
            if graph.desc(i)?.is_param {
                continue;
            }
            used[i] = true;
            start[i] = start[i].min(idx);
            end[i] = end[i].max(idx);
        }
        let o = node.output;
        used[o] = true;
        start[o] = start[o].min(idx);
        end[o] = end[o].max(idx);
    }

    let mut order: Vec<TensorId> = (0..n)
        .filter(|&i| used[i] && !graph.tensors()[i].is_param)
        .collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(graph.tensors()[i].bytes()), i));

    let mut placed: Vec<Interval> = Vec::new();
    for id in order {
        let size = align_up(graph.tensors()[id].bytes().max(1));
        let cand = Interval {
            tensor: id,
            start: start[id],
            end: end[id],
            offset: 0,
            size,
        };
        // Collect forbidden ranges from temporally overlapping intervals.
        let mut blocks: Vec<(usize, usize)> = placed
            .iter()
            .filter(|p| p.overlaps_time(&cand))
            .map(|p| (p.offset, p.offset + p.size))
            .collect();
        blocks.sort_unstable();
        let mut offset = 0usize;
        for (lo, hi) in blocks {
            if offset + size <= lo {
                break;
            }
            offset = offset.max(hi);
        }
        placed.push(Interval { offset, ..cand });
    }

    placed.sort_by_key(|i| i.tensor);
    let total_bytes = placed.iter().map(|i| i.offset + i.size).max().unwrap_or(0);
    let out = ArenaPlan {
        intervals: placed,
        total_bytes,
    };
    out.verify()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpKind;
    use rhizome_core::DType;

    fn chain(len: usize) -> Graph {
        let mut g = Graph::new();
        let mut cur = g.tensor("x", 1024, DType::F32, false);
        for i in 0..len {
            cur = g
                .push(OpKind::Silu, &[cur], &format!("h{i}"), 1024, DType::F32)
                .expect("valid");
        }
        g
    }

    #[test]
    fn chain_reuses_memory() {
        let g = chain(8);
        let p = plan(&g).expect("plannable");
        assert!(p.total_bytes < p.unplanned_bytes());
        p.verify().expect("no conflicts");
    }

    #[test]
    fn offsets_are_aligned() {
        let p = plan(&chain(5)).expect("plannable");
        for i in &p.intervals {
            assert_eq!(i.offset % ALIGN, 0);
        }
    }

    #[test]
    fn parameters_are_not_placed_in_the_arena() {
        let mut g = Graph::new();
        let x = g.tensor("x", 16, DType::F32, false);
        let w = g.tensor("w", 4096, DType::BF16, true);
        g.push(OpKind::Linear, &[x, w], "y", 16, DType::F32)
            .expect("valid");
        let p = plan(&g).expect("plannable");
        assert!(p.intervals.iter().all(|i| i.tensor != w));
    }

    #[test]
    fn long_lived_tensor_does_not_alias() {
        let mut g = Graph::new();
        let x = g.tensor("x", 256, DType::F32, false);
        let a = g
            .push(OpKind::Silu, &[x], "a", 256, DType::F32)
            .expect("valid");
        let b = g
            .push(OpKind::Silu, &[a], "b", 256, DType::F32)
            .expect("valid");
        g.push(OpKind::Add, &[x, b], "c", 256, DType::F32)
            .expect("valid");
        let p = plan(&g).expect("plannable");
        p.verify().expect("no conflicts");
        let xi = p.intervals.iter().find(|i| i.tensor == x).expect("x placed");
        let bi = p.intervals.iter().find(|i| i.tensor == b).expect("b placed");
        assert!(!xi.overlaps_memory(bi));
    }

    #[test]
    fn detects_injected_conflicts() {
        let mut p = plan(&chain(3)).expect("plannable");
        let first = p.intervals[0];
        for i in &mut p.intervals {
            i.offset = first.offset;
            i.start = 0;
            i.end = 10;
        }
        assert!(p.verify().is_err());
    }
}
