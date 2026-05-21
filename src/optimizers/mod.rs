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

impl LambdaSearchParams {
    fn validate(&self) -> Result<()> {
        if !self.start_exp.is_finite() {
            return Err(BaselineError::InvalidParameter {
                name: "start_exp",
                reason: "must be finite",
            });
        }
        if !self.stop_exp.is_finite() {
            return Err(BaselineError::InvalidParameter {
                name: "stop_exp",
                reason: "must be finite",
            });
        }
        if self.steps == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "steps",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }
}

const EXTENDED_RANGE_WIDTH_SCALE: f64 = 0.1;
const EXTENDED_RANGE_HEIGHT_SCALE: f64 = 1.0;
const EXTENDED_RANGE_SIGMA_SCALE: f64 = 1.0 / 12.0;

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

/// Runs AsLS over an extended-range lambda grid and returns the best edge match.
///
/// # References
///
/// - F. Zhang et al., "An Automatic Baseline Correction Method Based on the
///   Penalized Least Squares Method", *Sensors*, 2020.
/// - H. Krishna et al., "Range-independent background subtraction algorithm
///   for recovery of Raman spectra of biological tissue", *Journal of Raman
///   Spectroscopy*, 2012.
/// - `pybaselines.Baseline.optimize_extended_range` is used as a behavioral reference.
pub fn optimize_extended_range(y: &[f64], params: LambdaSearchParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;

    let added_window = ((y.len() as f64) * EXTENDED_RANGE_WIDTH_SCALE) as usize;
    if added_window == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "start_exp",
            reason: "input is too short for the extended range window",
        });
    }
    let (added_left, added_right) = extrapolated_edges(y, added_window);
    let added_gaussian = added_gaussian(y, added_window);
    let mut fit_data = Vec::with_capacity(y.len() + 2 * added_window);
    fit_data.extend(
        added_left
            .iter()
            .zip(&added_gaussian)
            .map(|(background, peak)| background + peak),
    );
    fit_data.extend_from_slice(y);
    fit_data.extend(
        added_right
            .iter()
            .zip(&added_gaussian)
            .map(|(background, peak)| background + peak),
    );

    let mut best: Option<(f64, Vec<f64>)> = None;
    for lambda in lambda_candidates(params) {
        let mut asls_params = AslsParams::default();
        asls_params.whittaker.lambda = lambda;
        let fit = asls(&fit_data, asls_params)?;
        let score = extended_range_score(&fit.baseline, &added_left, &added_right);
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            let start = added_window;
            let end = start + y.len();
            best = Some((score, fit.baseline[start..end].to_vec()));
        }
    }

    Ok(Fit {
        baseline: best.expect("validated params always generate candidates").1,
        report: FitReport::new(params.steps, true, 0.0),
    })
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

fn lambda_candidates(params: LambdaSearchParams) -> Vec<f64> {
    if params.steps == 1 {
        return vec![10f64.powf(params.start_exp)];
    }
    (0..params.steps)
        .map(|index| {
            let t = index as f64 / (params.steps - 1) as f64;
            10f64.powf(params.start_exp + t * (params.stop_exp - params.start_exp))
        })
        .collect()
}

fn extrapolated_edges(y: &[f64], pad_length: usize) -> (Vec<f64>, Vec<f64>) {
    let fit_window = pad_length.min(y.len());
    let (left_intercept, left_slope) = linear_fit_edge(&y[..fit_window], 0);
    let right_start = y.len() - fit_window;
    let (right_intercept, right_slope) = linear_fit_edge(&y[right_start..], right_start);
    let left = (1..=pad_length)
        .rev()
        .map(|offset| left_intercept - left_slope * offset as f64)
        .collect();
    let right = (0..pad_length)
        .map(|offset| {
            let x = (y.len() + offset) as f64;
            right_intercept + right_slope * x
        })
        .collect();
    (left, right)
}

fn linear_fit_edge(y: &[f64], start_index: usize) -> (f64, f64) {
    if y.len() <= 1 {
        return (*y.first().unwrap_or(&0.0), 0.0);
    }
    let len = y.len() as f64;
    let mean_x = start_index as f64 + (len - 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / len;
    let (numerator, denominator) =
        y.iter()
            .enumerate()
            .fold((0.0, 0.0), |(num, den), (offset, value)| {
                let centered_x = (start_index + offset) as f64 - mean_x;
                let centered_y = value - mean_y;
                (
                    centered_x.mul_add(centered_y, num),
                    centered_x.mul_add(centered_x, den),
                )
            });
    let slope = numerator / denominator;
    (mean_y - slope * mean_x, slope)
}

fn added_gaussian(y: &[f64], added_window: usize) -> Vec<f64> {
    let height =
        EXTENDED_RANGE_HEIGHT_SCALE * y.iter().copied().fold(f64::NEG_INFINITY, f64::max).abs();
    let sigma = added_window as f64 * EXTENDED_RANGE_SIGMA_SCALE;
    if added_window == 1 {
        return vec![height];
    }
    (0..added_window)
        .map(|index| {
            let x = -0.5 * added_window as f64
                + index as f64 * added_window as f64 / (added_window - 1) as f64;
            height * (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect()
}

fn extended_range_score(baseline: &[f64], added_left: &[f64], added_right: &[f64]) -> f64 {
    let added_window = added_left.len();
    let left_start = 0;
    let original_start = added_window;
    let right_start = baseline.len() - added_window;
    let right_score = baseline[right_start..]
        .iter()
        .zip(added_right)
        .map(|(fit, known)| {
            let residual = known - fit;
            residual * residual
        })
        .sum::<f64>();
    let left_score = baseline[left_start..original_start]
        .iter()
        .zip(added_left)
        .map(|(fit, known)| {
            let residual = known - fit;
            residual * residual
        })
        .sum::<f64>();
    right_score + left_score
}
