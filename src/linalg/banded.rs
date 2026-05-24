//! Shared banded linear algebra helpers.

use crate::{BaselineError, Result};

/// Reusable storage for a symmetric banded linear system.
#[derive(Debug, Clone)]
pub(crate) struct SymmetricBandedWorkspace {
    bands: Vec<f64>,
    n: usize,
    bandwidth: usize,
    intermediate: Vec<f64>,
}

impl SymmetricBandedWorkspace {
    /// Creates an empty symmetric banded workspace.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            bands: Vec::new(),
            n: 0,
            bandwidth: 0,
            intermediate: Vec::new(),
        }
    }

    /// Resizes and zeroes the stored bands for an `n` by `n` system.
    pub(crate) fn reset(&mut self, n: usize, bandwidth: usize) {
        self.n = n;
        self.bandwidth = bandwidth;
        self.bands.resize((bandwidth + 1) * n, 0.0);
        self.bands.fill(0.0);
        self.intermediate.resize(n, 0.0);
    }

    /// Sets a value by band offset and lower-index coordinate.
    pub(crate) fn set_band_value(&mut self, offset: usize, lower: usize, value: f64) {
        self.bands[band_index(self.n, offset, lower)] = value;
    }

    /// Returns a symmetric matrix value from the stored bands.
    #[cfg(test)]
    pub(crate) fn value(&self, row: usize, col: usize) -> f64 {
        let offset = row.abs_diff(col);
        if offset <= self.bandwidth {
            self.bands[band_index(self.n, offset, row.min(col))]
        } else {
            0.0
        }
    }

    /// Solves the current symmetric positive-definite banded system.
    pub(crate) fn solve_spd(&mut self, rhs: &[f64], output: &mut [f64]) -> Result<()> {
        solve_spd_banded_into(
            &mut self.bands,
            self.n,
            self.bandwidth,
            rhs,
            &mut self.intermediate,
            output,
        )
    }
}

#[inline]
fn band_index(n: usize, offset: usize, lower: usize) -> usize {
    offset * n + lower
}

fn solve_spd_banded_into(
    bands: &mut [f64],
    n: usize,
    bandwidth: usize,
    rhs: &[f64],
    intermediate: &mut [f64],
    output: &mut [f64],
) -> Result<()> {
    if rhs.len() != n {
        return Err(BaselineError::LengthMismatch {
            name: "rhs",
            expected: n,
            actual: rhs.len(),
        });
    }
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
    let expected_bands_len = (bandwidth + 1) * n;
    if bands.len() != expected_bands_len {
        return Err(BaselineError::LengthMismatch {
            name: "bands",
            expected: expected_bands_len,
            actual: bands.len(),
        });
    }

    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        for col in start..row {
            let row_col_index = band_index(n, row - col, col);
            let mut value = bands[row_col_index];
            let sum_start = start.max(col.saturating_sub(bandwidth));
            for mid in sum_start..col {
                value -=
                    bands[band_index(n, row - mid, mid)] * bands[band_index(n, col - mid, mid)];
            }
            let col_diag = bands[col];
            if col_diag.abs() <= f64::EPSILON {
                return Err(BaselineError::LinearSolve {
                    reason: "singular banded Cholesky factor",
                });
            }
            bands[row_col_index] = value / col_diag;
        }

        let mut diag = bands[row];
        for col in start..row {
            let value = bands[band_index(n, row - col, col)];
            diag -= value * value;
        }
        if diag <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "matrix was not positive definite",
            });
        }
        bands[row] = diag.sqrt();
    }

    intermediate.fill(0.0);
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        let mut tail = 0.0;
        for col in start..row {
            tail += bands[band_index(n, row - col, col)] * intermediate[col];
        }
        intermediate[row] = (rhs[row] - tail) / bands[row];
    }

    output.fill(0.0);
    for row in (0..n).rev() {
        let end = (row + bandwidth).min(n - 1);
        let mut tail = 0.0;
        for lower in row + 1..=end {
            tail += bands[band_index(n, lower - row, row)] * output[lower];
        }
        output[row] = (intermediate[row] - tail) / bands[row];
    }
    Ok(())
}
