//! Adaptive iteratively reweighted penalized least-squares smoothing.

use crate::Result;
use crate::fit::{Fit, FitReport};
use crate::whittaker::engine::{
    Reweighter, WhittakerParams, WhittakerWorkspace, active_at, fit_alloc, fit_into,
    relative_change,
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
pub(crate) struct AirPlsWeights;

impl Reweighter for AirPlsWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], iter: usize) -> f64 {
        self.update_masked(y, baseline, weights, iter, None)
    }

    fn update_masked(
        &self,
        y: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        iter: usize,
        active_mask: Option<&[bool]>,
    ) -> f64 {
        let previous = weights.to_vec();
        let negative_sum = y
            .iter()
            .zip(baseline)
            .enumerate()
            .filter(|(index, _)| active_at(active_mask, *index))
            .map(|(_, (observed, fitted))| observed - fitted)
            .filter(|residual| *residual < 0.0)
            .map(f64::abs)
            .sum::<f64>();

        if negative_sum <= f64::EPSILON {
            for (index, weight) in weights.iter_mut().enumerate() {
                *weight = if active_at(active_mask, index) {
                    1.0
                } else {
                    0.0
                };
            }
            return relative_change(&previous, weights);
        }

        let scale = (iter + 1) as f64 / negative_sum;
        for (index, (weight, residual)) in
            weights.iter_mut().zip(y.iter().zip(baseline)).enumerate()
        {
            let value = residual.0 - residual.1;
            if !active_at(active_mask, index) || value >= 0.0 {
                *weight = 0.0;
            } else {
                *weight = (scale * value.abs()).exp().min(1.0e12);
            }
        }

        if active_at(active_mask, 0)
            && let Some(first) = weights.first_mut()
        {
            *first = 1.0;
        }
        if active_at(active_mask, weights.len() - 1)
            && let Some(last) = weights.last_mut()
        {
            *last = 1.0;
        }
        relative_change(&previous, weights)
    }
}
