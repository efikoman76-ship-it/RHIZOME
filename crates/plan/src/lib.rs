//! Static execution plans: graph IR, adjoints, liveness and arena planning.
//!
//! A plan is built once per `(phase, shape, depth)` and then executed with
//! zero heap allocation on the hot path (SPEC.md §5.1, §6.1).

#![forbid(unsafe_code)]

pub mod arena;
pub mod graph;
pub mod hash;
pub mod schedule;

pub use arena::{ArenaPlan, Interval};
pub use graph::{Graph, Node, OpKind, Phase, TensorId};
pub use hash::plan_hash;
