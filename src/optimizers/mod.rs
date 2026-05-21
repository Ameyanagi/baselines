//! Optimizing and meta-algorithm baseline routines.

use crate::Result;
use crate::fit::{Fit, FitReport};
use crate::morphology::{MorphologyParams, mor};
use crate::whittaker::{AslsParams, asls};
use crate::workspace::validate_signal;

/// Parameters for lambda grid search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LambdaSearchParams {
    /// Smallest lambda exponent, using base 10.
    pub start_exp: f64,
    /// Largest lambda exponent, using base 10.
    pub stop_exp: f64,
    /// Number of candidates.
    pub steps: usize,
}

impl Default for LambdaSearchParams {
    fn default() -> Self {
        Self {
            start_exp: 2.0,
            stop_exp: 8.0,
            steps: 16,
        }
    }
}

/// Runs AsLS over a lambda grid and returns the smoothest finite candidate.
///
/// # References
///
/// - `pybaselines.Baseline.optimize_extended_range` is used as a behavioral reference.
pub fn optimize_extended_range(y: &[f64], params: LambdaSearchParams) -> Result<Fit> {
    validate_signal(y)?;
    let mut best: Option<(f64, Fit)> = None;
    let steps = params.steps.max(1);
    for i in 0..steps {
        let t = if steps == 1 {
            0.0
        } else {
            i as f64 / (steps - 1) as f64
        };
        let lambda = 10f64.powf(params.start_exp + t * (params.stop_exp - params.start_exp));
        let mut asls_params = AslsParams::default();
        asls_params.whittaker.lambda = lambda;
        let fit = asls(y, asls_params)?;
        let score = roughness(&fit.baseline);
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, fit));
        }
    }
    Ok(best.expect("at least one candidate is generated").1)
}

/// Applies a baseline function supplied by the caller.
///
/// # References
///
/// - `pybaselines.Baseline.custom_bc` is used as a behavioral reference.
pub fn custom_bc<F>(y: &[f64], baseline_fn: F) -> Result<Fit>
where
    F: FnOnce(&[f64]) -> Result<Fit>,
{
    validate_signal(y)?;
    baseline_fn(y)
}

/// Estimates a baseline by averaging small and large morphology windows.
///
/// # References
///
/// - `pybaselines.Baseline.adaptive_minmax` is used as a behavioral reference.
pub fn adaptive_minmax(y: &[f64], small: MorphologyParams, large: MorphologyParams) -> Result<Fit> {
    let left = mor(y, small)?;
    let right = mor(y, large)?;
    let baseline = left
        .baseline
        .iter()
        .zip(&right.baseline)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Runs collaborative PLS-style fitting over independent spectra.
///
/// # References
///
/// - `pybaselines.Baseline.collab_pls` is used as a behavioral reference.
pub fn collab_pls(spectra: &[Vec<f64>], params: AslsParams) -> Result<Vec<Fit>> {
    spectra
        .iter()
        .map(|spectrum| asls(spectrum, params))
        .collect()
}

fn roughness(values: &[f64]) -> f64 {
    values
        .windows(3)
        .map(|window| {
            let second_diff = window[0] - 2.0 * window[1] + window[2];
            second_diff * second_diff
        })
        .sum()
}
