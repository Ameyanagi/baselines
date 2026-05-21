//! Asymmetric least-squares smoothing.

use crate::fit::{Fit, FitReport};
use crate::whittaker::engine::{
    Reweighter, WhittakerParams, WhittakerWorkspace, fit_alloc, fit_into, relative_change,
};
use crate::{BaselineError, Result};

/// Parameters for [`asls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AslsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
}

impl Default for AslsParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams::default(),
            p: 0.01,
        }
    }
}

impl AslsParams {
    /// Validates AsLS parameters.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        Ok(())
    }
}

/// Fits an AsLS baseline.
///
/// # References
///
/// - P. H. C. Eilers and H. F. M. Boelens, "Baseline Correction with
///   Asymmetric Least Squares Smoothing", 2005.
/// - `pybaselines.Baseline.asls` is used as a behavioral reference.
pub fn asls(y: &[f64], params: AslsParams) -> Result<Fit> {
    params.validate()?;
    fit_alloc(y, params.whittaker, AslsWeights { p: params.p })
}

/// Fits an AsLS baseline into an existing output buffer.
pub fn asls_into(
    y: &[f64],
    params: AslsParams,
    baseline: &mut [f64],
    workspace: &mut WhittakerWorkspace,
) -> Result<FitReport> {
    params.validate()?;
    fit_into(
        y,
        params.whittaker,
        AslsWeights { p: params.p },
        baseline,
        workspace,
    )
}

#[derive(Debug, Clone, Copy)]
struct AslsWeights {
    p: f64,
}

impl Reweighter for AslsWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], _iter: usize) -> f64 {
        let previous = weights.to_vec();
        for ((weight, observed), fitted) in weights.iter_mut().zip(y).zip(baseline) {
            *weight = if observed > fitted {
                self.p
            } else {
                1.0 - self.p
            };
        }
        relative_change(&previous, weights)
    }
}
