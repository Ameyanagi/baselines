//! Asymmetrically reweighted penalized least-squares smoothing.

use crate::Result;
use crate::fit::{Fit, FitReport};
use crate::whittaker::engine::{
    Reweighter, WhittakerParams, WhittakerWorkspace, fit_alloc, fit_into, relative_change,
};
use crate::workspace::logistic;

/// Parameters for [`arpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArPlsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
}

/// Fits an arPLS baseline.
///
/// # References
///
/// - J. Baek et al., "Baseline correction using asymmetrically reweighted
///   penalized least squares smoothing", *Analyst*, 2015.
/// - `pybaselines.Baseline.arpls` is used as a behavioral reference.
pub fn arpls(y: &[f64], params: ArPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    fit_alloc(y, params.whittaker, ArPlsWeights)
}

/// Fits an arPLS baseline into an existing output buffer.
pub fn arpls_into(
    y: &[f64],
    params: ArPlsParams,
    baseline: &mut [f64],
    workspace: &mut WhittakerWorkspace,
) -> Result<FitReport> {
    params.whittaker.validate()?;
    fit_into(y, params.whittaker, ArPlsWeights, baseline, workspace)
}

#[derive(Debug, Clone, Copy)]
struct ArPlsWeights;

impl Reweighter for ArPlsWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], _iter: usize) -> f64 {
        let previous = weights.to_vec();
        let negative: Vec<f64> = y
            .iter()
            .zip(baseline)
            .map(|(observed, fitted)| observed - fitted)
            .filter(|residual| *residual < 0.0)
            .collect();

        if negative.is_empty() {
            weights.fill(1.0);
            return relative_change(&previous, weights);
        }

        let mean = negative.iter().sum::<f64>() / negative.len() as f64;
        let variance = negative
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f64>()
            / negative.len() as f64;
        let std = variance.sqrt().max(f64::EPSILON);

        for ((weight, observed), fitted) in weights.iter_mut().zip(y).zip(baseline) {
            let residual = observed - fitted;
            let exponent = 2.0 * (residual - (2.0 * std - mean)) / std;
            *weight = 1.0 - logistic(exponent);
        }

        relative_change(&previous, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whittaker::asls::{AslsParams, asls};

    #[test]
    fn whittaker_methods_return_finite_baselines() {
        let y: Vec<f64> = (0..100)
            .map(|i| {
                let x = i as f64 / 99.0;
                0.5 + 0.2 * x + (-(x - 0.45).powi(2) / 0.002).exp()
            })
            .collect();

        let asls_fit = asls(&y, AslsParams::default()).unwrap();
        let arpls_fit = arpls(&y, ArPlsParams::default()).unwrap();

        assert!(asls_fit.baseline.iter().all(|value| value.is_finite()));
        assert!(arpls_fit.baseline.iter().all(|value| value.is_finite()));
    }
}
