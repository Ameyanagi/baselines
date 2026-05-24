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

        anchor_active_endpoints(weights, active_mask);
        relative_change(&previous, weights)
    }
}

fn anchor_active_endpoints(weights: &mut [f64], active_mask: Option<&[bool]>) {
    match active_mask {
        Some(mask) => {
            if let Some(first_active) = mask.iter().position(|active| *active) {
                weights[first_active] = 1.0;
            }
            if let Some(last_active) = mask.iter().rposition(|active| *active) {
                weights[last_active] = 1.0;
            }
        }
        None => {
            if let Some(first) = weights.first_mut() {
                *first = 1.0;
            }
            if let Some(last) = weights.last_mut() {
                *last = 1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::whittaker::engine::Reweighter;

    use super::AirPlsWeights;

    #[test]
    fn masked_airpls_anchors_first_and_last_active_points() {
        let y = [0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let baseline = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let active = [false, true, true, true, true, true, false];
        let mut weights = vec![0.5; y.len()];

        AirPlsWeights.update_masked(&y, &baseline, &mut weights, 0, Some(&active));

        assert_eq!(weights[0], 0.0);
        assert_eq!(weights[1], 1.0);
        assert!(weights[2] > 0.0);
        assert!(weights[4] > 0.0);
        assert_eq!(weights[5], 1.0);
        assert_eq!(weights[6], 0.0);
    }
}
