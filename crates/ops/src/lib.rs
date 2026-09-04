//! The closed op vocabulary: f64 reference implementations and adjoints.
//!
//! Every op in SPEC.md §4 is defined here in double precision. The reference
//! is normative: optimised backends are validated against it by the
//! conformance suite, and the adjoints are validated by finite differences
//! (see [`gradcheck`]).

#![forbid(unsafe_code)]

pub mod attention;
pub mod delta;
pub mod elementwise;
pub mod gradcheck;
pub mod linalg;
pub mod loss;
pub mod norm;
pub mod rng;
pub mod routing;
pub mod sampling;

pub use elementwise::{silu, sigmoid, swiglu};
pub use gradcheck::{central_difference, GradCheckReport};
pub use norm::rmsnorm;
