//! Shared validation and reusable workspaces.

use crate::{BaselineError, Result};

/// Validates a one-dimensional finite input signal.
pub fn validate_signal(y: &[f64]) -> Result<()> {
    if y.is_empty() {
        return Err(BaselineError::EmptyInput);
    }
    for (index, value) in y.iter().enumerate() {
        if !value.is_finite() {
            return Err(BaselineError::NonFiniteInput { index });
        }
    }
    Ok(())
}

/// Validates an output buffer length.
pub fn validate_output(name: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(BaselineError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Returns the root-mean-square of a slice.
#[must_use]
pub fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq = values.iter().map(|value| value * value).sum::<f64>();
    (sum_sq / values.len() as f64).sqrt()
}

/// Computes a numerically safe logistic function.
#[must_use]
pub fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

/// Reusable buffers for algorithms that iteratively update a baseline.
#[derive(Debug, Clone)]
pub struct IterWorkspace {
    /// Current weights.
    pub weights: Vec<f64>,
    /// Previous weights.
    pub previous_weights: Vec<f64>,
    /// Residual vector.
    pub residual: Vec<f64>,
    /// Right-hand side or temporary signal.
    pub rhs: Vec<f64>,
}

impl IterWorkspace {
    /// Creates a workspace sized for `n` samples.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            weights: vec![1.0; n],
            previous_weights: vec![1.0; n],
            residual: vec![0.0; n],
            rhs: vec![0.0; n],
        }
    }

    /// Resizes all buffers to `n` samples.
    pub fn resize(&mut self, n: usize) {
        self.weights.resize(n, 1.0);
        self.previous_weights.resize(n, 1.0);
        self.residual.resize(n, 0.0);
        self.rhs.resize(n, 0.0);
    }
}
