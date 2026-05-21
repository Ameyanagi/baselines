//! Error types used by baseline correction algorithms.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, BaselineError>;

/// Errors returned by baseline algorithms.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BaselineError {
    /// The input signal had no samples.
    #[error("input signal must not be empty")]
    EmptyInput,

    /// The input signal is too short for the requested algorithm.
    #[error("input length {len} is too short for {algorithm}; minimum is {min}")]
    TooShort {
        /// Algorithm name.
        algorithm: &'static str,
        /// Observed input length.
        len: usize,
        /// Minimum accepted input length.
        min: usize,
    },

    /// A slice length did not match the expected length.
    #[error("length mismatch for {name}: expected {expected}, got {actual}")]
    LengthMismatch {
        /// Slice or buffer name.
        name: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },

    /// A numeric parameter was invalid.
    #[error("invalid parameter {name}: {reason}")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Explanation.
        reason: &'static str,
    },

    /// An input value was not finite.
    #[error("input contains a non-finite value at index {index}")]
    NonFiniteInput {
        /// Index of the invalid value.
        index: usize,
    },

    /// A linear system could not be solved.
    #[error("linear solve failed: {reason}")]
    LinearSolve {
        /// Explanation.
        reason: &'static str,
    },

    /// A requested backend or operation is not available.
    #[error("{feature} is not supported yet: {reason}")]
    Unsupported {
        /// Feature or backend name.
        feature: &'static str,
        /// Explanation.
        reason: &'static str,
    },
}
