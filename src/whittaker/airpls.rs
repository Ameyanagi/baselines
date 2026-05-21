//! Adaptive iteratively reweighted penalized least-squares smoothing.

use crate::Result;
use crate::fit::{Fit, FitReport};
use crate::whittaker::engine::{
    Reweighter, WhittakerParams, WhittakerWorkspace, fit_alloc, fit_into, relative_change,
};

/// Parameters for [`airpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AirPlsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
}

/// Fits an airPLS baseline.
///
/// # References
///
/// - Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using
///   adaptive iteratively reweighted penalized least squares", *Analyst*, 2010.
/// - `pybaselines.Baseline.airpls` is used as a behavioral reference.
pub fn airpls(y: &[f64], params: AirPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    fit_alloc(y, params.whittaker, AirPlsWeights)
}

/// Fits an airPLS baseline into an existing output buffer.
pub fn airpls_into(
    y: &[f64],
    params: AirPlsParams,
    baseline: &mut [f64],
    workspace: &mut WhittakerWorkspace,
) -> Result<FitReport> {
    params.whittaker.validate()?;
    fit_into(y, params.whittaker, AirPlsWeights, baseline, workspace)
}

#[derive(Debug, Clone, Copy)]
struct AirPlsWeights;

impl Reweighter for AirPlsWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], iter: usize) -> f64 {
        let previous = weights.to_vec();
        let residuals = y
            .iter()
            .zip(baseline)
            .map(|(observed, fitted)| observed - fitted);
        let negative_sum = residuals
            .clone()
            .filter(|residual| *residual < 0.0)
            .map(f64::abs)
            .sum::<f64>();

        if negative_sum <= f64::EPSILON {
            weights.fill(1.0);
            return relative_change(&previous, weights);
        }

        let scale = (iter + 1) as f64 / negative_sum;
        for (weight, residual) in weights.iter_mut().zip(y.iter().zip(baseline)) {
            let value = residual.0 - residual.1;
            if value >= 0.0 {
                *weight = 0.0;
            } else {
                *weight = (scale * value.abs()).exp().min(1.0e12);
            }
        }

        if let Some(first) = weights.first_mut() {
            *first = 1.0;
        }
        if let Some(last) = weights.last_mut() {
            *last = 1.0;
        }
        relative_change(&previous, weights)
    }
}
