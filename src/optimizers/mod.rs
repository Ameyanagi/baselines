//! Optimizing and meta-algorithm baseline routines.

use crate::BaselineError;
use crate::Result;
use crate::fit::{Fit, FitReport};
use crate::polynomial::fit_weighted_polynomial;
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

/// Parameters for adaptive min-max polynomial baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveMinmaxParams {
    /// Lower polynomial order. The upper order is `poly_order + 1`.
    pub poly_order: usize,
    /// Fraction of points at the left edge constrained in endpoint-weighted fits.
    pub left_constrained_fraction: f64,
    /// Fraction of points at the right edge constrained in endpoint-weighted fits.
    pub right_constrained_fraction: f64,
    /// Weight assigned to constrained edge points.
    pub constrained_weight: f64,
}

impl Default for AdaptiveMinmaxParams {
    fn default() -> Self {
        Self {
            poly_order: 2,
            left_constrained_fraction: 0.01,
            right_constrained_fraction: 0.01,
            constrained_weight: 1.0e5,
        }
    }
}

impl AdaptiveMinmaxParams {
    fn validate(&self, len: usize) -> Result<()> {
        if self.poly_order + 2 > len {
            return Err(BaselineError::TooShort {
                algorithm: "adaptive_minmax",
                len,
                min: self.poly_order + 2,
            });
        }
        if !self.left_constrained_fraction.is_finite()
            || self.left_constrained_fraction < 0.0
            || self.left_constrained_fraction > 1.0
        {
            return Err(BaselineError::InvalidParameter {
                name: "left_constrained_fraction",
                reason: "must be finite and between 0 and 1",
            });
        }
        if !self.right_constrained_fraction.is_finite()
            || self.right_constrained_fraction < 0.0
            || self.right_constrained_fraction > 1.0
        {
            return Err(BaselineError::InvalidParameter {
                name: "right_constrained_fraction",
                reason: "must be finite and between 0 and 1",
            });
        }
        if !self.constrained_weight.is_finite() || self.constrained_weight <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "constrained_weight",
                reason: "must be finite and positive",
            });
        }
        Ok(())
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

/// Estimates a baseline from the maximum of constrained and unconstrained polynomial fits.
///
/// # References
///
/// - A. Cao et al., "A robust method for automated background subtraction of
///   tissue fluorescence", *Journal of Raman Spectroscopy*, 2007.
/// - `pybaselines.Baseline.adaptive_minmax` is used as a behavioral reference.
pub fn adaptive_minmax(y: &[f64], params: AdaptiveMinmaxParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate(y.len())?;

    let weights = vec![1.0; y.len()];
    let mut constrained_weights = weights.clone();
    let left_count = ((y.len() as f64) * params.left_constrained_fraction).ceil() as usize;
    let right_count = ((y.len() as f64) * params.right_constrained_fraction).ceil() as usize;
    constrained_weights[..left_count.min(y.len())].fill(params.constrained_weight);
    let right_start = y.len().saturating_sub(right_count);
    constrained_weights[right_start..].fill(params.constrained_weight);

    let mut baseline = vec![f64::NEG_INFINITY; y.len()];
    let mut candidate = vec![0.0; y.len()];
    for order in [params.poly_order, params.poly_order + 1] {
        fit_weighted_polynomial(y, &weights, order, &mut candidate)?;
        for (target, value) in baseline.iter_mut().zip(&candidate) {
            *target = target.max(*value);
        }
        fit_weighted_polynomial(y, &constrained_weights, order, &mut candidate)?;
        for (target, value) in baseline.iter_mut().zip(&candidate) {
            *target = target.max(*value);
        }
    }

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
