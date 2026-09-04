//! The graph IR and backward derivation.

use rhizome_core::{DType, Error, Result};
use std::collections::BTreeSet;

/// Identifier of a tensor inside one graph.
pub type TensorId = usize;

/// Which execution phase a plan targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Prefill over a chunk of tokens.
    Prefill,
    /// Single-token decode.
    Decode,
    /// One Think Core iteration.
    CoreStep,
    /// Training forward + backward.
    Train,
}

impl Phase {
    /// Stable name used in plan hashes and metrics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
            Phase::CoreStep => "core_step",
            Phase::Train => "train",
        }
    }
}

/// The closed op vocabulary (SPEC.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// Elementwise addition.
    Add,
    /// Elementwise multiplication.
    Mul,
    /// SiLU activation.
    Silu,
    /// Gated SwiGLU FFN activation.
    Swiglu,
    /// Residual add (kept distinct for fusion and checkpointing).
    ResidualAdd,
    /// RMS normalisation.
    RmsNorm,
    /// Dense matrix multiply.
    Linear,
    /// Grouped matrix multiply with variable group sizes.
    GroupedLinear,
    /// Softmax over the last axis.
    Softmax,
    /// Gated delta chunk scan.
    DeltaScan,
    /// Single-token gated delta update.
    DeltaStep,
    /// MLA latent compression.
    MlaCompress,
    /// MLA prefill attention.
    MlaPrefill,
    /// MLA paged decode attention.
    MlaDecode,
    /// Group-limited top-k routing (non-differentiable selection).
    RouterTopk,
    /// Permute tokens into expert order.
    MoePermute,
    /// Inverse of [`OpKind::MoePermute`].
    MoeUnpermute,
    /// Product-key memory lookup.
    PkmLookup,
    /// Fused cross-entropy with z-loss.
    CrossEntropyZLoss,
    /// Think Core halt head.
    HaltHead,
    /// Token sampling (non-differentiable).
    Sample,
    /// Quantise to an MX block format (straight-through).
    QuantizeMx,
    /// Collective all-reduce.
    AllReduce,
}

impl OpKind {
    /// Whether the op propagates gradients.
    #[must_use]
    pub const fn is_differentiable(self) -> bool {
        !matches!(
            self,
            OpKind::RouterTopk | OpKind::Sample | OpKind::PkmLookup
        )
    }

    /// Whether the op is recomputable for activation checkpointing.
    #[must_use]
    pub const fn is_recomputable(self) -> bool {
        matches!(
            self,
            OpKind::Silu
                | OpKind::Swiglu
                | OpKind::RmsNorm
                | OpKind::Linear
                | OpKind::GroupedLinear
                | OpKind::Add
                | OpKind::Mul
        )
    }

    /// Stable name used in hashing and diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            OpKind::Add => "add",
            OpKind::Mul => "mul",
            OpKind::Silu => "silu",
            OpKind::Swiglu => "swiglu",
            OpKind::ResidualAdd => "residual_add",
            OpKind::RmsNorm => "rmsnorm",
            OpKind::Linear => "linear",
            OpKind::GroupedLinear => "grouped_linear",
            OpKind::Softmax => "softmax",
            OpKind::DeltaScan => "gated_delta_scan_chunked",
            OpKind::DeltaStep => "gated_delta_step",
            OpKind::MlaCompress => "mla_compress",
            OpKind::MlaPrefill => "mla_prefill",
            OpKind::MlaDecode => "mla_decode",
            OpKind::RouterTopk => "router_topk_grouped",
            OpKind::MoePermute => "moe_permute",
            OpKind::MoeUnpermute => "moe_unpermute",
            OpKind::PkmLookup => "pkm_gather_sum",
            OpKind::CrossEntropyZLoss => "cross_entropy_zloss",
            OpKind::HaltHead => "halt_head",
            OpKind::Sample => "sampling",
            OpKind::QuantizeMx => "quantize_mx",
            OpKind::AllReduce => "all_reduce",
        }
    }
}

/// A tensor declaration inside a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    /// Diagnostic name.
    pub name: String,
    /// Element count.
    pub elems: usize,
    /// Element format.
    pub dtype: DType,
    /// Whether the tensor is a parameter (lives outside the arena).
    pub is_param: bool,
}

impl TensorDesc {
    /// Storage requirement in bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.dtype.bytes_for(self.elems)
    }
}

/// One node of the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Op executed by this node.
    pub op: OpKind,
    /// Input tensors.
    pub inputs: Vec<TensorId>,
    /// Output tensor.
    pub output: TensorId,
}

/// A directed acyclic execution graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    tensors: Vec<TensorDesc>,
    nodes: Vec<Node>,
}

impl Graph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Graph::default()
    }

    /// Declare a tensor and return its id.
    pub fn tensor(&mut self, name: &str, elems: usize, dtype: DType, is_param: bool) -> TensorId {
        self.tensors.push(TensorDesc {
            name: name.to_string(),
            elems,
            dtype,
            is_param,
        });
        self.tensors.len() - 1
    }

    /// Append a node whose output is a freshly declared tensor.
    ///
    /// # Errors
    /// Returns [`Error::Shape`] if an input id is unknown.
    pub fn push(
        &mut self,
        op: OpKind,
        inputs: &[TensorId],
        out_name: &str,
        out_elems: usize,
        out_dtype: DType,
    ) -> Result<TensorId> {
        for &i in inputs {
            if i >= self.tensors.len() {
                return Err(Error::Shape(format!("unknown input tensor {i}")));
            }
        }
        let output = self.tensor(out_name, out_elems, out_dtype, false);
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            output,
        });
        Ok(output)
    }

    /// All nodes in topological (insertion) order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// All tensor descriptors.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    /// Descriptor of one tensor.
    ///
    /// # Errors
    /// Returns [`Error::Shape`] when the id is unknown.
    pub fn desc(&self, id: TensorId) -> Result<&TensorDesc> {
        self.tensors
            .get(id)
            .ok_or_else(|| Error::Shape(format!("unknown tensor {id}")))
    }

    /// Validate structural invariants: SSA outputs, defined-before-use,
    /// and no parameter written by a node.
    ///
    /// # Errors
    /// Returns [`Error::Shape`] describing the first violation.
    pub fn validate(&self) -> Result<()> {
        let mut defined: BTreeSet<TensorId> = self
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.is_param)
            .map(|(i, _)| i)
            .collect();
        // Tensors that are never written by a node are graph inputs.
        let written: BTreeSet<TensorId> = self.nodes.iter().map(|n| n.output).collect();
        for (i, _) in self.tensors.iter().enumerate() {
            if !written.contains(&i) {
                defined.insert(i);
            }
        }
        let mut seen_outputs: BTreeSet<TensorId> = BTreeSet::new();
        let mut live: BTreeSet<TensorId> = defined;
        for (idx, n) in self.nodes.iter().enumerate() {
            for &i in &n.inputs {
                if !live.contains(&i) {
                    return Err(Error::Shape(format!(
                        "node {idx} ({}) uses tensor {i} before it is defined",
                        n.op.name()
                    )));
                }
            }
            if !seen_outputs.insert(n.output) {
                return Err(Error::Shape(format!(
                    "tensor {} is written more than once (SSA violation)",
                    n.output
                )));
            }
            if self.tensors[n.output].is_param {
                return Err(Error::Shape(format!(
                    "node {idx} writes parameter tensor {}",
                    n.output
                )));
            }
            live.insert(n.output);
        }
        Ok(())
    }

    /// Derive the backward graph, returning it together with the number of
    /// gradient nodes appended.
    ///
    /// Non-differentiable ops contribute no gradient nodes; their inputs keep
    /// the straight-through gradient of the corresponding forward tensor.
    ///
    /// # Errors
    /// Propagates validation failures of the derived graph.
    pub fn backward(&self) -> Result<Graph> {
        let mut g = self.clone();
        let mut grad_of: Vec<Option<TensorId>> = vec![None; self.tensors.len()];
        if let Some(last) = self.nodes.last() {
            let seed = g.tensor(
                &format!("grad_{}", self.tensors[last.output].name),
                self.tensors[last.output].elems,
                DType::F32,
                false,
            );
            grad_of[last.output] = Some(seed);
        }
        for node in self.nodes.iter().rev() {
            let Some(gy) = grad_of[node.output] else {
                continue;
            };
            if !node.op.is_differentiable() {
                continue;
            }
            for &inp in &node.inputs {
                let desc = &self.tensors[inp];
                let contribution = g.push(
                    adjoint_op(node.op),
                    &[gy, inp],
                    &format!("grad_{}", desc.name),
                    desc.elems,
                    DType::F32,
                )?;
                grad_of[inp] = Some(match grad_of[inp] {
                    None => contribution,
                    Some(prev) => g.push(
                        OpKind::Add,
                        &[prev, contribution],
                        &format!("grad_acc_{}", desc.name),
                        desc.elems,
                        DType::F32,
                    )?,
                });
            }
        }
        g.validate()?;
        Ok(g)
    }
}

/// The op used to compute the adjoint contribution of a forward op.
#[must_use]
pub const fn adjoint_op(op: OpKind) -> OpKind {
    match op {
        OpKind::Linear | OpKind::GroupedLinear => OpKind::Linear,
        OpKind::Softmax => OpKind::Softmax,
        OpKind::RmsNorm => OpKind::RmsNorm,
        OpKind::DeltaScan => OpKind::DeltaScan,
        OpKind::MlaPrefill => OpKind::MlaPrefill,
        OpKind::MlaDecode => OpKind::MlaDecode,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mlp_graph() -> Graph {
        let mut g = Graph::new();
        let x = g.tensor("x", 16, DType::F32, false);
        let w1 = g.tensor("w1", 16 * 32, DType::BF16, true);
        let w2 = g.tensor("w2", 32 * 16, DType::BF16, true);
        let h = g
            .push(OpKind::Linear, &[x, w1], "h", 32, DType::F32)
            .expect("valid");
        let a = g
            .push(OpKind::Silu, &[h], "a", 32, DType::F32)
            .expect("valid");
        g.push(OpKind::Linear, &[a, w2], "y", 16, DType::F32)
            .expect("valid");
        g
    }

    #[test]
    fn valid_graph_passes_validation() {
        mlp_graph().validate().expect("valid graph");
    }

    #[test]
    fn unknown_input_is_rejected() {
        let mut g = Graph::new();
        assert!(g.push(OpKind::Silu, &[7], "y", 4, DType::F32).is_err());
    }

    #[test]
    fn backward_reaches_every_parameter() {
        let g = mlp_graph();
        let b = g.backward().expect("derivable");
        assert!(b.nodes().len() > g.nodes().len());
        let names: Vec<&str> = b.tensors().iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"grad_w1"));
        assert!(names.contains(&"grad_w2"));
        assert!(names.contains(&"grad_x"));
    }

    #[test]
    fn backward_skips_non_differentiable_ops() {
        let mut g = Graph::new();
        let x = g.tensor("x", 8, DType::F32, false);
        let r = g
            .push(OpKind::RouterTopk, &[x], "route", 2, DType::F32)
            .expect("valid");
        g.push(OpKind::Add, &[r, r], "y", 2, DType::F32)
            .expect("valid");
        let b = g.backward().expect("derivable");
        let names: Vec<&str> = b.tensors().iter().map(|t| t.name.as_str()).collect();
        assert!(!names.contains(&"grad_x"));
    }

    #[test]
    fn gradient_accumulates_for_reused_tensors() {
        let mut g = Graph::new();
        let x = g.tensor("x", 4, DType::F32, false);
        g.push(OpKind::Add, &[x, x], "y", 4, DType::F32)
            .expect("valid");
        let b = g.backward().expect("derivable");
        assert!(b.tensors().iter().any(|t| t.name.starts_with("grad_acc_x")));
    }

    #[test]
    fn op_metadata_is_consistent() {
        assert!(!OpKind::Sample.is_differentiable());
        assert!(OpKind::Linear.is_recomputable());
        assert_eq!(OpKind::MlaDecode.name(), "mla_decode");
        assert_eq!(Phase::CoreStep.name(), "core_step");
    }
}
