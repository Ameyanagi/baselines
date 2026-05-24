//! Shared banded linear algebra helpers.

use crate::{BaselineError, Result};

/// Reusable storage for a symmetric banded linear system.
#[derive(Debug, Clone)]
pub(crate) struct SymmetricBandedWorkspace {
    bands: Vec<Vec<f64>>,
    intermediate: Vec<f64>,
}

impl SymmetricBandedWorkspace {
    /// Creates an empty symmetric banded workspace.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            bands: Vec::new(),
            intermediate: Vec::new(),
        }
    }

    /// Resizes and zeroes the stored bands for an `n` by `n` system.
    pub(crate) fn reset(&mut self, n: usize, bandwidth: usize) {
        self.bands.resize_with(bandwidth + 1, Vec::new);
        for band in &mut self.bands {
            band.resize(n, 0.0);
            band.fill(0.0);
        }
        self.intermediate.resize(n, 0.0);
    }

    /// Returns mutable symmetric bands.
    pub(crate) fn bands_mut(&mut self) -> &mut [Vec<f64>] {
        &mut self.bands
    }

    /// Solves the current symmetric positive-definite banded system.
    pub(crate) fn solve_spd(&mut self, rhs: &[f64], output: &mut [f64]) -> Result<()> {
        solve_spd_banded_into(&mut self.bands, rhs, &mut self.intermediate, output)
    }
}

fn solve_spd_banded_into(
    bands: &mut [Vec<f64>],
    rhs: &[f64],
    intermediate: &mut [f64],
    output: &mut [f64],
) -> Result<()> {
    let n = rhs.len();
    if output.len() != n {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: n,
            actual: output.len(),
        });
    }
    if intermediate.len() != n {
        return Err(BaselineError::LengthMismatch {
            name: "intermediate",
            expected: n,
            actual: intermediate.len(),
        });
    }
    if bands.is_empty() || bands.iter().any(|band| band.len() != n) {
        return Err(BaselineError::LengthMismatch {
            name: "bands",
            expected: n,
            actual: bands.first().map_or(0, Vec::len),
        });
    }

    let bandwidth = bands.len() - 1;
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        for col in start..row {
            let mut value = symmetric_band_value(bands, row, col);
            let sum_start = start.max(col.saturating_sub(bandwidth));
            for mid in sum_start..col {
                value -=
                    symmetric_band_value(bands, row, mid) * symmetric_band_value(bands, col, mid);
            }
            let col_diag = symmetric_band_value(bands, col, col);
            if col_diag.abs() <= f64::EPSILON {
                return Err(BaselineError::LinearSolve {
                    reason: "singular banded Cholesky factor",
                });
            }
            set_symmetric_band_value(bands, row, col, value / col_diag);
        }

        let mut diag = symmetric_band_value(bands, row, row);
        for col in start..row {
            let value = symmetric_band_value(bands, row, col);
            diag -= value * value;
        }
        if diag <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "matrix was not positive definite",
            });
        }
        set_symmetric_band_value(bands, row, row, diag.sqrt());
    }

    intermediate.fill(0.0);
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        let tail = (start..row)
            .map(|col| symmetric_band_value(bands, row, col) * intermediate[col])
            .sum::<f64>();
        intermediate[row] = (rhs[row] - tail) / symmetric_band_value(bands, row, row);
    }

    output.fill(0.0);
    for row in (0..n).rev() {
        let end = (row + bandwidth).min(n - 1);
        let tail = (row + 1..=end)
            .map(|lower| symmetric_band_value(bands, lower, row) * output[lower])
            .sum::<f64>();
        output[row] = (intermediate[row] - tail) / symmetric_band_value(bands, row, row);
    }
    Ok(())
}

fn set_symmetric_band_value(bands: &mut [Vec<f64>], row: usize, col: usize, value: f64) {
    let offset = row.abs_diff(col);
    let lower = row.min(col);
    bands[offset][lower] = value;
}

fn symmetric_band_value(bands: &[Vec<f64>], row: usize, col: usize) -> f64 {
    let offset = row.abs_diff(col);
    let lower = row.min(col);
    if offset < bands.len() {
        bands[offset][lower]
    } else {
        0.0
    }
}
