//! Fit outputs and convergence metadata.

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

/// Baseline output with convergence metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    /// Estimated baseline.
    pub baseline: Vec<f64>,
    /// Fit metadata.
    pub report: FitReport,
}

impl Fit {
    /// Returns `y - baseline`.
    #[must_use]
    pub fn corrected(&self, y: &[f64]) -> Vec<f64> {
        y.iter()
            .zip(&self.baseline)
            .map(|(observed, baseline)| observed - baseline)
            .collect()
    }
}
