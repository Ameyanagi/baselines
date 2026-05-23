//! Fit outputs and convergence metadata.

use crate::Result;
use crate::data::validate_matrix_len;
use crate::workspace::validate_output;

/// Metadata describing a baseline fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitReport {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm met its convergence tolerance.
    pub converged: bool,
    /// Final convergence metric reported by the algorithm.
    pub tolerance: f64,
}

impl FitReport {
    /// Creates a convergence report.
    #[must_use]
    pub fn new(iterations: usize, converged: bool, tolerance: f64) -> Self {
        Self {
            iterations,
            converged,
            tolerance,
        }
    }
}

/// One-dimensional baseline output with convergence metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit1D {
    /// Estimated baseline.
    pub baseline: Vec<f64>,
    /// Fit metadata.
    pub report: FitReport,
}

impl Fit1D {
    /// Returns `y - baseline`.
    pub fn corrected(&self, y: &[f64]) -> Result<Vec<f64>> {
        validate_output("y", self.baseline.len(), y.len())?;
        Ok(y.iter()
            .zip(&self.baseline)
            .map(|(observed, baseline)| observed - baseline)
            .collect())
    }

    /// Writes `y - baseline` into an existing output buffer.
    pub fn corrected_into(&self, y: &[f64], output: &mut [f64]) -> Result<()> {
        validate_output("y", self.baseline.len(), y.len())?;
        validate_output("output", self.baseline.len(), output.len())?;
        for ((target, observed), baseline) in output.iter_mut().zip(y).zip(&self.baseline) {
            *target = observed - baseline;
        }
        Ok(())
    }
}

/// Backward-compatible alias for one-dimensional fit output.
pub type Fit = Fit1D;

/// Two-dimensional row-major baseline output with convergence metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit2D {
    /// Estimated row-major baseline.
    pub baseline: Vec<f64>,
    /// Number of matrix rows.
    pub rows: usize,
    /// Number of matrix columns.
    pub cols: usize,
    /// Fit metadata.
    pub report: FitReport,
}

impl Fit2D {
    /// Creates a two-dimensional fit after validating the baseline length.
    pub fn new(baseline: Vec<f64>, rows: usize, cols: usize, report: FitReport) -> Result<Self> {
        validate_matrix_len("baseline", rows, cols, baseline.len())?;
        Ok(Self {
            baseline,
            rows,
            cols,
            report,
        })
    }

    /// Returns the number of baseline elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.baseline.len()
    }

    /// Returns whether the baseline has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.baseline.is_empty()
    }

    /// Returns the matrix shape as `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns `data - baseline`.
    pub fn corrected(&self, data: &[f64]) -> Result<Vec<f64>> {
        validate_output("data", self.baseline.len(), data.len())?;
        Ok(data
            .iter()
            .zip(&self.baseline)
            .map(|(observed, baseline)| observed - baseline)
            .collect())
    }

    /// Writes `data - baseline` into an existing output buffer.
    pub fn corrected_into(&self, data: &[f64], output: &mut [f64]) -> Result<()> {
        validate_output("data", self.baseline.len(), data.len())?;
        validate_output("output", self.baseline.len(), output.len())?;
        for ((target, observed), baseline) in output.iter_mut().zip(data).zip(&self.baseline) {
            *target = observed - baseline;
        }
        Ok(())
    }
}
