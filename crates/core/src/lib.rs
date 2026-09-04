//! Core schema, configuration, dtypes and error types for RHIZOME.
//!
//! This crate is dependency-free so that it can be built hermetically with no
//! network access (see ADR 0001). It defines the model schema
//! ([`ModelConfig`]), the numeric formats used across the stack ([`DType`]),
//! and the parameter / FLOP / memory accounting used by `rhizome params`.

#![forbid(unsafe_code)]

pub mod config;
pub mod dtype;
pub mod error;
pub mod params;
pub mod toml_lite;

pub use config::{FrontEnd, ModelConfig, ReadSource};
pub use dtype::DType;
pub use error::{Error, Result};
pub use params::ParamReport;
