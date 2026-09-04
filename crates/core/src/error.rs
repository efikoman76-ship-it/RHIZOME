//! Typed error handling for RHIZOME core.

use core::fmt;

/// Errors produced while loading, validating or accounting a configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A configuration file could not be parsed.
    Parse {
        /// 1-based line number of the offending input.
        line: usize,
        /// Human readable reason.
        reason: String,
    },
    /// A configuration key was missing.
    MissingKey(String),
    /// A configuration value was of the wrong type.
    BadType {
        /// Key whose value was invalid.
        key: String,
        /// Expected type name.
        expected: &'static str,
    },
    /// A semantic validation rule was violated.
    Invalid(String),
    /// A shape mismatch was detected by an op or the planner.
    Shape(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse { line, reason } => write!(f, "parse error on line {line}: {reason}"),
            Error::MissingKey(k) => write!(f, "missing configuration key `{k}`"),
            Error::BadType { key, expected } => {
                write!(f, "key `{key}` is not a valid {expected}")
            }
            Error::Invalid(m) => write!(f, "invalid configuration: {m}"),
            Error::Shape(m) => write!(f, "shape error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Convenient result alias.
pub type Result<T> = core::result::Result<T, Error>;
