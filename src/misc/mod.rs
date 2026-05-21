//! Miscellaneous baseline algorithms.

use crate::fit::{Fit, FitReport};
use crate::whittaker::{AslsParams, asls};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

/// Parameters for BEADS-style baseline estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadsParams {
    /// Smoothness penalty passed to the internal Whittaker approximation.
    pub lambda: f64,
}

impl Default for BeadsParams {
    fn default() -> Self {
        Self { lambda: 1.0e6 }
    }
}

/// Interpolates a baseline through user-provided anchor points.
///
/// # References
///
/// - `pybaselines.Baseline.interp_pts` is used as a behavioral reference.
pub fn interp_pts(y: &[f64], points: &[(usize, f64)]) -> Result<Fit> {
    validate_signal(y)?;
    if points.is_empty() {
        return Err(BaselineError::InvalidParameter {
            name: "points",
            reason: "must contain at least one anchor",
        });
    }
    let mut sorted = points.to_vec();
    sorted.sort_by_key(|(index, _)| *index);
    if sorted.iter().any(|(index, _)| *index >= y.len()) {
        return Err(BaselineError::InvalidParameter {
            name: "points",
            reason: "anchor index is outside the input length",
        });
    }

    let mut baseline = vec![0.0; y.len()];
    let (first_index, first_value) = sorted[0];
    baseline[..=first_index].fill(first_value);
    for pair in sorted.windows(2) {
        let (start, y0) = pair[0];
        let (end, y1) = pair[1];
        let width = (end - start).max(1) as f64;
        for (offset, target) in baseline[start..=end].iter_mut().enumerate() {
            let t = offset as f64 / width;
            *target = y0.mul_add(1.0 - t, y1 * t);
        }
    }
    let (last_index, last_value) = *sorted.last().expect("points were checked as non-empty");
    baseline[last_index..].fill(last_value);

    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Estimates a baseline with a BEADS-inspired sparse/smooth approximation.
///
/// # References
///
/// - `pybaselines.Baseline.beads` is used as a behavioral reference.
pub fn beads(y: &[f64], params: BeadsParams) -> Result<Fit> {
    let mut asls_params = AslsParams::default();
    asls_params.whittaker.lambda = params.lambda;
    asls(y, asls_params)
}
