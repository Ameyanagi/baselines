//! Polynomial baseline algorithms.
//!
//! # References
//!
//! - C. A. Lieber and A. Mahadevan-Jansen, "Automated Method for Subtraction
//!   of Fluorescence from Biological Raman Spectra", *Applied Spectroscopy*,
//!   2003.
//! - J. Zhao et al., "Automated Autofluorescence Background Subtraction
//!   Algorithm for Biomedical Raman Spectroscopy", *Applied Spectroscopy*,
//!   2007.
//! - R. Koenker and G. Bassett Jr., "Regression Quantiles", *Econometrica*,
//!   1978.
//! - V. Mazet et al., "Background removal from spectra by designing and
//!   minimising a non-quadratic cost function", *Chemometrics and Intelligent
//!   Laboratory Systems*, 2005.
//! - J. Liu et al., "Goldindec: A Novel Algorithm for Raman Spectrum Baseline
//!   Correction", *Applied Spectroscopy*, 2015.
//! - `pybaselines` is used as a behavioral reference.

use crate::fit::{Fit, FitReport};
use crate::workspace::{rms, validate_output, validate_signal};
use crate::{BaselineError, Result};

/// Parameters for a direct polynomial baseline fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyParams {
    /// Polynomial order.
    pub order: usize,
}

impl Default for PolyParams {
    fn default() -> Self {
        Self { order: 2 }
    }
}

/// Parameters for modified polynomial baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModPolyParams {
    /// Polynomial order.
    pub order: usize,
    /// Maximum number of clipped least-squares iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
}

impl Default for ModPolyParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 100,
            tol: 1.0e-3,
        }
    }
}

/// Parameters for improved modified polynomial baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImodPolyParams {
    /// Polynomial order.
    pub order: usize,
    /// Maximum number of clipped least-squares iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
}

impl Default for ImodPolyParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 100,
            tol: 1.0e-3,
        }
    }
}

/// Fits a least-squares polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline.poly` is used as a behavioral reference.
pub fn poly(y: &[f64], params: PolyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = poly_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a least-squares polynomial baseline into an existing output buffer.
pub fn poly_into(y: &[f64], params: PolyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_poly_input(y, params.order, baseline)?;
    let weights = vec![1.0; y.len()];
    fit_weighted_polynomial(y, &weights, params.order, baseline)?;
    Ok(FitReport::new(1, true, 0.0))
}

/// Fits a modified polynomial baseline.
///
/// # References
///
/// - C. A. Lieber and A. Mahadevan-Jansen, *Applied Spectroscopy*, 2003.
/// - `pybaselines.Baseline.modpoly` is used as a behavioral reference.
pub fn modpoly(y: &[f64], params: ModPolyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = modpoly_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a modified polynomial baseline into an existing output buffer.
pub fn modpoly_into(y: &[f64], params: ModPolyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_iter_params(params.max_iter, params.tol)?;
    validate_poly_input(y, params.order, baseline)?;
    let mut clipped = y.to_vec();
    let mut previous = vec![0.0; y.len()];
    let weights = vec![1.0; y.len()];
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(baseline);
        fit_weighted_polynomial(&clipped, &weights, params.order, baseline)?;
        tolerance = relative_baseline_change(&previous, baseline);
        let residual: Vec<f64> = clipped
            .iter()
            .zip(baseline.iter())
            .map(|(observed, fitted)| observed - fitted)
            .collect();
        let spread = rms(&residual);
        for ((target, observed), fitted) in clipped.iter_mut().zip(y).zip(baseline.iter()) {
            *target = observed.min(fitted + spread);
        }
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

/// Fits an improved modified polynomial baseline.
///
/// # References
///
/// - J. Zhao et al., *Applied Spectroscopy*, 2007.
/// - `pybaselines.Baseline.imodpoly` is used as a behavioral reference.
pub fn imodpoly(y: &[f64], params: ImodPolyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = imodpoly_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Non-quadratic cost function used by [`penalized_poly`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenalizedCost {
    /// Asymmetric truncated quadratic loss.
    AsymmetricTruncatedQuadratic,
    /// Symmetric truncated quadratic loss.
    SymmetricTruncatedQuadratic,
    /// Asymmetric Huber loss.
    AsymmetricHuber,
    /// Symmetric Huber loss.
    SymmetricHuber,
    /// Asymmetric Indec loss.
    AsymmetricIndec,
    /// Symmetric Indec loss.
    SymmetricIndec,
}

/// Parameters for penalized polynomial fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenalizedPolyParams {
    /// Polynomial order.
    pub order: usize,
    /// Maximum number of non-quadratic refinement iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
    /// Non-quadratic loss function.
    pub cost: PenalizedCost,
    /// Residual threshold separating quadratic and non-quadratic loss. If
    /// `None`, uses `std(y) / 10`.
    pub threshold: Option<f64>,
    /// Scale factor for the non-quadratic penalty in `(0, 1]`.
    pub alpha_factor: f64,
}

impl Default for PenalizedPolyParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 250,
            tol: 1.0e-3,
            cost: PenalizedCost::AsymmetricTruncatedQuadratic,
            threshold: None,
            alpha_factor: 0.99,
        }
    }
}

/// Parameters for LOESS baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoessParams {
    /// Local regression window size.
    pub window_size: usize,
}

impl Default for LoessParams {
    fn default() -> Self {
        Self { window_size: 31 }
    }
}

/// Parameters for quantile-regression polynomial fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantRegParams {
    /// Polynomial order.
    pub order: usize,
    /// Quantile in `(0, 1)` to fit.
    pub quantile: f64,
    /// Maximum number of iteratively reweighted least-squares iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
    /// Residual floor used to avoid singular weights. If `None`, a
    /// scale-aware default is used.
    pub epsilon: Option<f64>,
}

impl Default for QuantRegParams {
    fn default() -> Self {
        Self {
            order: 2,
            quantile: 0.05,
            max_iter: 250,
            tol: 1.0e-6,
            epsilon: None,
        }
    }
}

/// Parameters for Goldindec baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoldindecParams {
    /// Polynomial order.
    pub order: usize,
    /// Maximum number of non-quadratic fit iterations for each threshold.
    pub max_iter: usize,
    /// Relative baseline-change tolerance for each threshold fit.
    pub tol: f64,
    /// Asymmetric non-quadratic loss function.
    pub cost: PenalizedCost,
    /// Expected fraction of peak points in `(0, 1)`.
    pub peak_ratio: f64,
    /// Scale factor for the non-quadratic penalty in `(0, 1]`.
    pub alpha_factor: f64,
    /// Maximum number of threshold-search iterations.
    pub max_threshold_iter: usize,
    /// Tolerance for the up/down-ratio objective.
    pub ratio_tol: f64,
    /// Relative threshold-change tolerance.
    pub threshold_tol: f64,
}

impl Default for GoldindecParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 250,
            tol: 1.0e-3,
            cost: PenalizedCost::AsymmetricIndec,
            peak_ratio: 0.5,
            alpha_factor: 0.99,
            max_threshold_iter: 100,
            ratio_tol: 1.0e-3,
            threshold_tol: 1.0e-6,
        }
    }
}

/// Fits a penalized polynomial baseline.
///
/// # References
///
/// - V. Mazet et al., "Background removal from spectra by designing and
///   minimising a non-quadratic cost function", *Chemometrics and Intelligent
///   Laboratory Systems*, 2005.
/// - J. Liu et al., "Goldindec: A Novel Algorithm for Raman Spectrum Baseline
///   Correction", *Applied Spectroscopy*, 2015.
/// - `pybaselines.Baseline.penalized_poly` is used as a behavioral reference.
pub fn penalized_poly(y: &[f64], params: PenalizedPolyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = penalized_poly_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a robust local-constant LOESS baseline.
///
/// # References
///
/// - A. F. Ruckstuhl et al., "Baseline subtraction using robust local
///   regression estimation", *Journal of Quantitative Spectroscopy and
///   Radiative Transfer*, 2001.
/// - W. S. Cleveland, "Robust locally weighted regression and smoothing
///   scatterplots", *Journal of the American Statistical Association*, 1979.
/// - `pybaselines.Baseline.loess` is used as a behavioral reference.
pub fn loess(y: &[f64], params: LoessParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = loess_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a robust local-constant LOESS baseline into an existing output buffer.
pub fn loess_into(y: &[f64], params: LoessParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if params.window_size == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "window_size",
            reason: "must be greater than zero",
        });
    }
    if params.window_size > y.len() {
        return Err(BaselineError::InvalidParameter {
            name: "window_size",
            reason: "must not be greater than the input length",
        });
    }

    const LOESS_MAX_ITER: usize = 10;
    const LOESS_TOL: f64 = 1.0e-3;
    const TUKEY_SCALE: f64 = 3.0;

    let x: Vec<f64> = (0..y.len()).map(|index| scaled_x(index, y.len())).collect();
    let windows = loess_windows(&x, params.window_size);
    let mut robust_weights = vec![1.0; y.len()];
    let mut previous = y.to_vec();
    let mut tolerance = f64::INFINITY;

    for iter in 0..=LOESS_MAX_ITER {
        previous.copy_from_slice(if iter == 0 { y } else { baseline });
        fit_local_constant_loess(y, &x, &windows, &robust_weights, baseline);
        tolerance = relative_baseline_change(&previous, baseline);
        if tolerance < LOESS_TOL {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }

        update_loess_robust_weights(y, baseline, &mut robust_weights, TUKEY_SCALE);
    }

    Ok(FitReport::new(LOESS_MAX_ITER + 1, false, tolerance))
}

/// Fits a quantile-regression polynomial baseline.
///
/// # References
///
/// - R. Koenker and G. Bassett Jr., "Regression Quantiles", *Econometrica*,
///   1978.
/// - `pybaselines.Baseline.quant_reg` is used as a behavioral reference.
pub fn quant_reg(y: &[f64], params: QuantRegParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = quant_reg_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a Goldindec polynomial baseline.
///
/// # References
///
/// - J. Liu et al., "Goldindec: A Novel Algorithm for Raman Spectrum Baseline
///   Correction", *Applied Spectroscopy*, 2015.
/// - `pybaselines.Baseline.goldindec` is used as a behavioral reference.
pub fn goldindec(y: &[f64], params: GoldindecParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = goldindec_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a penalized polynomial baseline into an existing output buffer.
pub fn penalized_poly_into(
    y: &[f64],
    params: PenalizedPolyParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_penalized_poly_params(params)?;
    validate_poly_input(y, params.order, baseline)?;

    let threshold = params
        .threshold
        .unwrap_or_else(|| standard_deviation(y) / 10.0);
    let weights = vec![1.0; y.len()];
    fit_weighted_polynomial(y, &weights, params.order, baseline)?;
    fit_penalized_polynomial_with_threshold(
        y,
        PenalizedFitParams {
            order: params.order,
            threshold,
            alpha_factor: params.alpha_factor,
            cost: params.cost,
            max_iter: params.max_iter,
            tol: params.tol,
        },
        baseline,
    )
}

/// Fits a Goldindec baseline into an existing output buffer.
pub fn goldindec_into(
    y: &[f64],
    params: GoldindecParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_goldindec_params(params)?;
    validate_poly_input(y, params.order, baseline)?;

    let weights = vec![1.0; y.len()];
    fit_weighted_polynomial(y, &weights, params.order, baseline)?;
    let initial = baseline.to_vec();
    let goal = goldindec_up_down_ratio_goal(params.peak_ratio);
    let mut lower = 0.0;
    let mut upper = y
        .iter()
        .zip(&initial)
        .map(|(observed, fitted)| observed - fitted)
        .fold(f64::NEG_INFINITY, f64::max)
        .abs();
    if !upper.is_finite() || upper <= f64::EPSILON {
        return Ok(FitReport::new(1, true, 0.0));
    }

    let mut tolerance = f64::INFINITY;
    let mut threshold = lower + 0.618 * (upper - lower);
    for iter in 0..params.max_threshold_iter {
        baseline.copy_from_slice(&initial);
        fit_penalized_polynomial_with_threshold(
            y,
            PenalizedFitParams {
                order: params.order,
                threshold,
                alpha_factor: params.alpha_factor,
                cost: params.cost,
                max_iter: params.max_iter,
                tol: params.tol,
            },
            baseline,
        )?;

        let above = y
            .iter()
            .zip(baseline.iter())
            .filter(|(observed, fitted)| observed > fitted)
            .count();
        let below_or_equal = y.len().saturating_sub(above).max(1);
        tolerance = above as f64 / below_or_equal as f64 - goal;
        if tolerance > params.ratio_tol {
            lower = threshold;
        } else if tolerance < -params.ratio_tol {
            upper = threshold;
        } else {
            return Ok(FitReport::new(iter + 1, true, tolerance.abs()));
        }

        let previous_threshold = threshold;
        threshold = lower + 0.618 * (upper - lower);
        if relative_scalar_change(previous_threshold, threshold) < params.threshold_tol {
            return Ok(FitReport::new(iter + 1, true, tolerance.abs()));
        }
    }

    Ok(FitReport::new(
        params.max_threshold_iter,
        false,
        tolerance.abs(),
    ))
}

/// Fits an improved modified polynomial baseline into an existing output buffer.
pub fn imodpoly_into(y: &[f64], params: ImodPolyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_iter_params(params.max_iter, params.tol)?;
    validate_poly_input(y, params.order, baseline)?;
    let mut clipped = y.to_vec();
    let mut previous = vec![0.0; y.len()];
    let mut weights = vec![1.0; y.len()];
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(baseline);
        fit_weighted_polynomial(&clipped, &weights, params.order, baseline)?;
        tolerance = relative_baseline_change(&previous, baseline);
        let residual: Vec<f64> = y
            .iter()
            .zip(baseline.iter())
            .map(|(observed, fitted)| observed - fitted)
            .collect();
        let negative: Vec<f64> = residual
            .iter()
            .copied()
            .filter(|value| *value < 0.0)
            .collect();
        let spread = if negative.is_empty() {
            rms(&residual)
        } else {
            rms(&negative)
        };
        for (((target, weight), observed), fitted) in clipped
            .iter_mut()
            .zip(weights.iter_mut())
            .zip(y)
            .zip(baseline.iter())
        {
            let limit = fitted + spread;
            *target = observed.min(limit);
            *weight = if observed > &limit { 0.0 } else { 1.0 };
        }
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

/// Fits a quantile-regression polynomial baseline into an existing output buffer.
pub fn quant_reg_into(
    y: &[f64],
    params: QuantRegParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_quant_reg_params(params)?;
    validate_poly_input(y, params.order, baseline)?;

    let mut weights = vec![1.0; y.len()];
    let mut previous = vec![0.0; y.len()];
    let epsilon = params
        .epsilon
        .unwrap_or_else(|| f64::EPSILON.sqrt() * signal_scale(y).max(1.0));
    fit_weighted_polynomial(y, &weights, params.order, baseline)?;

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.max_iter {
        previous.copy_from_slice(baseline);
        for ((weight, observed), fitted) in weights.iter_mut().zip(y).zip(baseline.iter()) {
            let residual = observed - fitted;
            let quantile_weight = if residual >= 0.0 {
                params.quantile
            } else {
                1.0 - params.quantile
            };
            *weight = quantile_weight / residual.abs().max(epsilon);
        }

        fit_weighted_polynomial(y, &weights, params.order, baseline)?;
        tolerance = relative_baseline_change(&previous, baseline);
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

fn validate_quant_reg_params(params: QuantRegParams) -> Result<()> {
    validate_iter_params(params.max_iter, params.tol)?;
    if !params.quantile.is_finite() || params.quantile <= 0.0 || params.quantile >= 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "quantile",
            reason: "must be finite and between 0 and 1",
        });
    }
    if let Some(epsilon) = params.epsilon
        && (!epsilon.is_finite() || epsilon <= 0.0)
    {
        return Err(BaselineError::InvalidParameter {
            name: "epsilon",
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

fn validate_penalized_poly_params(params: PenalizedPolyParams) -> Result<()> {
    validate_iter_params(params.max_iter, params.tol)?;
    if !params.alpha_factor.is_finite() || params.alpha_factor <= 0.0 || params.alpha_factor > 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "alpha_factor",
            reason: "must be finite and in (0, 1]",
        });
    }
    if let Some(threshold) = params.threshold
        && (!threshold.is_finite() || threshold < 0.0)
    {
        return Err(BaselineError::InvalidParameter {
            name: "threshold",
            reason: "must be finite and non-negative",
        });
    }
    Ok(())
}

fn validate_goldindec_params(params: GoldindecParams) -> Result<()> {
    validate_iter_params(params.max_iter, params.tol)?;
    if params.max_threshold_iter == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "max_threshold_iter",
            reason: "must be greater than zero",
        });
    }
    if !params.peak_ratio.is_finite() || params.peak_ratio <= 0.0 || params.peak_ratio >= 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "peak_ratio",
            reason: "must be finite and between 0 and 1",
        });
    }
    if !params.alpha_factor.is_finite() || params.alpha_factor <= 0.0 || params.alpha_factor > 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "alpha_factor",
            reason: "must be finite and in (0, 1]",
        });
    }
    if !params.ratio_tol.is_finite() || params.ratio_tol <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "ratio_tol",
            reason: "must be finite and positive",
        });
    }
    if !params.threshold_tol.is_finite() || params.threshold_tol <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "threshold_tol",
            reason: "must be finite and positive",
        });
    }
    if !params.cost.is_asymmetric() {
        return Err(BaselineError::InvalidParameter {
            name: "cost",
            reason: "goldindec requires an asymmetric cost",
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct PenalizedFitParams {
    order: usize,
    threshold: f64,
    alpha_factor: f64,
    cost: PenalizedCost,
    max_iter: usize,
    tol: f64,
}

fn fit_penalized_polynomial_with_threshold(
    y: &[f64],
    params: PenalizedFitParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    let weights = vec![1.0; y.len()];
    let mut adjusted = y.to_vec();
    let mut previous = vec![0.0; y.len()];
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(baseline);
        for ((target, observed), fitted) in adjusted.iter_mut().zip(y).zip(baseline.iter()) {
            let residual = observed - fitted;
            *target = observed
                + non_quadratic_correction(
                    residual,
                    params.threshold,
                    params.alpha_factor,
                    params.cost,
                );
        }

        fit_weighted_polynomial(&adjusted, &weights, params.order, baseline)?;
        tolerance = relative_baseline_change(&previous, baseline);
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

fn goldindec_up_down_ratio_goal(peak_ratio: f64) -> f64 {
    0.7679 + 11.2358 * peak_ratio - 39.7064 * peak_ratio.powi(2) + 92.3583 * peak_ratio.powi(3)
}

fn relative_scalar_change(previous: f64, current: f64) -> f64 {
    (current - previous).abs() / previous.abs().max(f64::EPSILON)
}

impl PenalizedCost {
    fn is_asymmetric(self) -> bool {
        matches!(
            self,
            Self::AsymmetricTruncatedQuadratic | Self::AsymmetricHuber | Self::AsymmetricIndec
        )
    }
}

fn non_quadratic_correction(
    residual: f64,
    threshold: f64,
    alpha_factor: f64,
    cost: PenalizedCost,
) -> f64 {
    let alpha = 0.5 * alpha_factor;
    match cost {
        PenalizedCost::AsymmetricTruncatedQuadratic => {
            truncated_quadratic_correction(residual, threshold, alpha, false)
        }
        PenalizedCost::SymmetricTruncatedQuadratic => {
            truncated_quadratic_correction(residual, threshold, alpha, true)
        }
        PenalizedCost::AsymmetricHuber => huber_correction(residual, threshold, alpha, false),
        PenalizedCost::SymmetricHuber => huber_correction(residual, threshold, alpha, true),
        PenalizedCost::AsymmetricIndec => indec_correction(residual, threshold, alpha, false),
        PenalizedCost::SymmetricIndec => indec_correction(residual, threshold, alpha, true),
    }
}

fn truncated_quadratic_correction(
    residual: f64,
    threshold: f64,
    alpha: f64,
    symmetric: bool,
) -> f64 {
    let in_quadratic_region = if symmetric {
        residual.abs() < threshold
    } else {
        residual < threshold
    };
    if in_quadratic_region {
        residual * (2.0 * alpha - 1.0)
    } else {
        -residual
    }
}

fn huber_correction(residual: f64, threshold: f64, alpha: f64, symmetric: bool) -> f64 {
    if symmetric {
        if residual.abs() < threshold {
            residual * (2.0 * alpha - 1.0)
        } else {
            2.0 * alpha * threshold * residual.signum()
        }
    } else if residual < threshold {
        residual * (2.0 * alpha - 1.0)
    } else {
        2.0 * alpha * threshold - residual
    }
}

fn indec_correction(residual: f64, threshold: f64, alpha: f64, symmetric: bool) -> f64 {
    let in_quadratic_region = if symmetric {
        residual.abs() < threshold
    } else {
        residual < threshold
    };
    if in_quadratic_region {
        residual * (2.0 * alpha - 1.0)
    } else {
        let sign = if symmetric { residual.signum() } else { 1.0 };
        let denominator = (2.0 * residual * residual).max(f64::MIN_POSITIVE);
        -(residual + alpha * sign * threshold.powi(3) / denominator)
    }
}

fn loess_windows(x: &[f64], window_size: usize) -> Vec<(usize, usize)> {
    let len = x.len();
    let mut windows = Vec::with_capacity(len);
    let mut left = 0;
    let mut right = window_size;
    windows.push((0, window_size));

    for i in 1..len.saturating_sub(1) {
        while right < len && x[i] - x[left] > x[right] - x[i] {
            left += 1;
            right += 1;
        }
        windows.push((left, right));
    }

    if len > 1 {
        windows.push((len - window_size, len));
    }
    windows
}

fn fit_local_constant_loess(
    y: &[f64],
    x: &[f64],
    windows: &[(usize, usize)],
    robust_weights: &[f64],
    baseline: &mut [f64],
) {
    for (i, target) in baseline.iter_mut().enumerate() {
        let (left, right) = windows[i];
        let max_distance = (x[i] - x[left]).abs().max((x[right - 1] - x[i]).abs());
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;

        for j in left..right {
            let scaled = if max_distance > 0.0 {
                ((x[j] - x[i]).abs() / max_distance).min(1.0)
            } else {
                0.0
            };
            let distance_weight = (1.0 - scaled.powi(3)).powi(3);
            let weight = distance_weight * robust_weights[j];
            weight_sum += weight;
            value_sum += weight * y[j];
        }

        *target = if weight_sum > 0.0 {
            value_sum / weight_sum
        } else {
            y[i]
        };
    }
}

fn update_loess_robust_weights(
    y: &[f64],
    baseline: &[f64],
    robust_weights: &mut [f64],
    scale: f64,
) {
    let residuals: Vec<f64> = y
        .iter()
        .zip(baseline)
        .map(|(observed, fitted)| observed - fitted)
        .collect();
    let spread = median_absolute_value(&residuals);
    if spread <= f64::EPSILON {
        robust_weights.fill(1.0);
        return;
    }

    for (weight, residual) in robust_weights.iter_mut().zip(residuals) {
        if residual <= 0.0 {
            *weight = 1.0;
        } else {
            let inner = residual / (spread * scale);
            let sqrt_weight = (1.0 - inner * inner).max(0.0);
            *weight = sqrt_weight * sqrt_weight;
        }
    }
}

fn median_absolute_value(values: &[f64]) -> f64 {
    const NORMAL_MEDIAN_SCALE: f64 = 0.674_489_750_196_081_7;

    let mut absolute: Vec<f64> = values.iter().map(|value| value.abs()).collect();
    absolute.sort_by(|left, right| left.total_cmp(right));
    let median = if absolute.len().is_multiple_of(2) {
        let upper = absolute.len() / 2;
        0.5 * (absolute[upper - 1] + absolute[upper])
    } else {
        absolute[absolute.len() / 2]
    };
    median / NORMAL_MEDIAN_SCALE
}

fn signal_scale(y: &[f64]) -> f64 {
    let (min, max) = y
        .iter()
        .copied()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(value), max.max(value))
        });
    max - min
}

fn standard_deviation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn validate_poly_input(y: &[f64], order: usize, baseline: &[f64]) -> Result<()> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if order + 1 > y.len() {
        return Err(BaselineError::TooShort {
            algorithm: "polynomial",
            len: y.len(),
            min: order + 1,
        });
    }
    Ok(())
}

fn validate_iter_params(max_iter: usize, tol: f64) -> Result<()> {
    if max_iter == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "max_iter",
            reason: "must be greater than zero",
        });
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "tol",
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

/// Fits a weighted polynomial baseline into an existing output buffer.
pub(crate) fn fit_weighted_polynomial(
    y: &[f64],
    weights: &[f64],
    order: usize,
    baseline: &mut [f64],
) -> Result<()> {
    let coeffs = fit_weighted_polynomial_coefficients(y, weights, order)?;
    evaluate_polynomial_coefficients(&coeffs, baseline);
    Ok(())
}

/// Fits weighted polynomial coefficients in increasing order.
pub(crate) fn fit_weighted_polynomial_coefficients(
    y: &[f64],
    weights: &[f64],
    order: usize,
) -> Result<Vec<f64>> {
    let n_coeffs = order + 1;
    let mut normal = vec![vec![0.0; n_coeffs]; n_coeffs];
    let mut rhs = vec![0.0; n_coeffs];

    for (i, (observed, weight)) in y.iter().zip(weights).enumerate() {
        let x = scaled_x(i, y.len());
        let powers = powers(x, order);
        for row in 0..n_coeffs {
            rhs[row] += weight * observed * powers[row];
            for col in 0..n_coeffs {
                normal[row][col] += weight * powers[row] * powers[col];
            }
        }
    }

    solve_dense(normal, rhs)
}

/// Evaluates polynomial coefficients on the crate's standard scaled grid.
pub(crate) fn evaluate_polynomial_coefficients(coeffs: &[f64], baseline: &mut [f64]) {
    let len = baseline.len();
    for (i, fitted) in baseline.iter_mut().enumerate() {
        let x = scaled_x(i, len);
        *fitted = evaluate_polynomial(coeffs, x);
    }
}

fn scaled_x(index: usize, len: usize) -> f64 {
    if len == 1 {
        0.0
    } else {
        2.0 * index as f64 / (len - 1) as f64 - 1.0
    }
}

fn powers(x: f64, order: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(order + 1);
    let mut current = 1.0;
    for _ in 0..=order {
        values.push(current);
        current *= x;
    }
    values
}

fn evaluate_polynomial(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, coeff| acc * x + coeff)
}

fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if matrix[best][pivot].abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "singular polynomial normal equations",
            });
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);

        let pivot_row = matrix[pivot].clone();
        for row in pivot + 1..n {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for (entry, pivot_entry) in matrix[row][pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *entry -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let known = matrix[row][row + 1..]
            .iter()
            .zip(&solution[row + 1..])
            .map(|(coeff, value)| coeff * value)
            .sum::<f64>();
        solution[row] = (rhs[row] - known) / matrix[row][row];
    }
    Ok(solution)
}

fn relative_baseline_change(previous: &[f64], current: &[f64]) -> f64 {
    let numerator = previous
        .iter()
        .zip(current)
        .map(|(old, new)| {
            let diff = new - old;
            diff * diff
        })
        .sum::<f64>()
        .sqrt();
    let denominator = previous
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    numerator / denominator.max(f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_recovers_quadratic() {
        let y: Vec<f64> = (0..50)
            .map(|i| {
                let x = scaled_x(i, 50);
                1.0 + 0.5 * x + 2.0 * x * x
            })
            .collect();

        let fit = poly(&y, PolyParams { order: 2 }).unwrap();
        for (expected, actual) in y.iter().zip(&fit.baseline) {
            assert!((expected - actual).abs() < 1.0e-9);
        }
    }

    #[test]
    fn modpoly_returns_finite_values() {
        let y: Vec<f64> = (0..100)
            .map(|i| {
                let x = scaled_x(i, 100);
                1.0 + 0.3 * x + (-(x - 0.2).powi(2) / 0.01).exp()
            })
            .collect();

        let fit = modpoly(&y, ModPolyParams::default()).unwrap();
        assert!(fit.baseline.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn loess_preserves_constant_signal() {
        let y = vec![2.5; 64];
        let fit = loess(&y, LoessParams { window_size: 13 }).unwrap();

        for value in fit.baseline {
            assert!((value - 2.5).abs() < 1.0e-12);
        }
    }
}
