//! Additional Whittaker-family algorithm entry points.

use crate::fit::{Fit, FitHistory};
use crate::linalg::pentadiagonal::{
    GeneralPentadiagonalSystem, GeneralPentadiagonalWorkspace, solve_general_pentadiagonal,
    solve_second_order_with_first_order,
};
use crate::polynomial::fit_weighted_polynomial;
use crate::whittaker::ArPlsParams;
use crate::whittaker::engine::{
    Reweighter, WhittakerParams, active_at, fit_alloc, relative_change,
};
use crate::workspace::{logistic, validate_output, validate_signal};
use crate::{BaselineError, FitReport, Result};

/// Parameters for improved asymmetric least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IaslsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Smoothness penalty for the first derivative of the residual.
    pub lambda_1: f64,
}

impl Default for IaslsParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams::default(),
            p: 0.01,
            lambda_1: 1.0e-4,
        }
    }
}

impl IaslsParams {
    /// Validates IAsLS parameters.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        if !self.lambda_1.is_finite() || self.lambda_1 <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "lambda_1",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for doubly reweighted penalized least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrPlsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Penalty reweighting control in `[0, 1]`.
    pub eta: f64,
}

impl Default for DrPlsParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                max_iter: 50,
                tol: 1.0e-3,
            },
            eta: 0.5,
        }
    }
}

impl DrPlsParams {
    /// Validates drPLS parameters.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.eta.is_finite() || self.eta < 0.0 || self.eta > 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "eta",
                reason: "must be finite and between 0 and 1",
            });
        }
        Ok(())
    }
}

/// Parameters for improved asymmetrically reweighted penalized least squares.
pub type IarPlsParams = ArPlsParams;
/// Parameters for locally symmetric reweighted penalized least squares.
pub type LsrPlsParams = ArPlsParams;

/// Parameters for adaptive smoothness penalized least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsPlsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetric weighting coefficient.
    pub asymmetric_coef: f64,
}

impl Default for AsPlsParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                max_iter: 100,
                tol: 1.0e-3,
            },
            asymmetric_coef: 0.5,
        }
    }
}

impl AsPlsParams {
    /// Validates asPLS parameters.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.asymmetric_coef.is_finite() || self.asymmetric_coef <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "asymmetric_coef",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for peaked signal asymmetric least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsalsaParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Exponential decay scale. If `None`, uses `std(y) / 10`.
    pub k: Option<f64>,
}

impl Default for PsalsaParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                max_iter: 50,
                tol: 1.0e-3,
            },
            p: 0.5,
            k: None,
        }
    }
}

impl PsalsaParams {
    /// Validates psalsa parameters that do not depend on the input signal.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        if let Some(k) = self.k
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(BaselineError::InvalidParameter {
                name: "k",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for derivative peak-screening asymmetric least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerPsalsaParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Peak-height scale. If `None`, uses `std(y) / 10`.
    pub k: Option<f64>,
    /// Optional half-window for derivative smoothing. If `None`, uses
    /// `len(y) / 200`.
    pub smooth_half_window: Option<usize>,
    /// Number of mollifier smoothing passes before computing derivatives.
    pub num_smooths: usize,
}

impl Default for DerPsalsaParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams::default(),
            p: 0.01,
            k: None,
            smooth_half_window: None,
            num_smooths: 16,
        }
    }
}

impl DerPsalsaParams {
    /// Validates derpsalsa parameters that do not depend on the input signal.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        if let Some(k) = self.k
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(BaselineError::InvalidParameter {
                name: "k",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for Bayesian reweighted penalized least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrPlsParams {
    /// Shared Whittaker parameters for the inner baseline fit.
    pub whittaker: WhittakerParams,
    /// Maximum number of outer peak-proportion updates.
    pub max_iter_2: usize,
    /// Convergence tolerance for the outer peak-proportion update.
    pub tol_2: f64,
}

impl Default for BrPlsParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                max_iter: 50,
                tol: 1.0e-3,
            },
            max_iter_2: 50,
            tol_2: 1.0e-3,
        }
    }
}

impl BrPlsParams {
    /// Validates BrPLS parameters.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if self.max_iter_2 == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "max_iter_2",
                reason: "must be greater than zero",
            });
        }
        if !self.tol_2.is_finite() || self.tol_2 <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "tol_2",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Fits an IAsLS baseline.
///
/// # References
///
/// - S. He et al., "Baseline correction for Raman spectra using an improved
///   asymmetric least squares method", *Analytical Methods*, 2014.
/// - `pybaselines.Baseline.iasls` is used as a behavioral reference.
pub fn iasls(y: &[f64], params: IaslsParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = iasls_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits an IAsLS baseline into an existing output buffer.
pub fn iasls_into(y: &[f64], params: IaslsParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "iasls",
            len: y.len(),
            min: 3,
        });
    }
    params.validate()?;

    let n = y.len();
    let mut workspace = crate::whittaker::WhittakerWorkspace::new(n);
    workspace.iter.weights.fill(1.0);
    fit_weighted_polynomial(y, &workspace.iter.weights, 2, &mut workspace.iter.residual)?;
    for ((weight, observed), fitted) in workspace
        .iter
        .weights
        .iter_mut()
        .zip(y)
        .zip(&workspace.iter.residual)
    {
        *weight = asls_weight(*observed, *fitted, params.p);
    }
    let mut first_order_rhs = vec![0.0; n];
    first_order_penalty_rhs(y, params.lambda_1, &mut first_order_rhs);

    let mut tolerance = f64::INFINITY;
    for iter in 0..=params.whittaker.max_iter {
        workspace
            .iter
            .previous_weights
            .copy_from_slice(&workspace.iter.weights);

        for (((diagonal, rhs), weight), (observed, first_order_rhs)) in workspace
            .iter
            .residual
            .iter_mut()
            .zip(workspace.iter.rhs.iter_mut())
            .zip(&workspace.iter.weights)
            .zip(y.iter().zip(&first_order_rhs))
        {
            let weight_squared = weight * weight;
            *diagonal = weight_squared;
            *rhs = weight_squared * observed + first_order_rhs;
        }

        solve_second_order_with_first_order(
            &workspace.iter.residual,
            &workspace.iter.rhs,
            params.whittaker.lambda,
            params.lambda_1,
            baseline,
            &mut workspace.solver,
        )?;

        for ((weight, observed), fitted) in workspace
            .iter
            .weights
            .iter_mut()
            .zip(y)
            .zip(baseline.iter())
        {
            *weight = asls_weight(*observed, *fitted, params.p);
        }
        tolerance = relative_change(&workspace.iter.previous_weights, &workspace.iter.weights);
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(
        params.whittaker.max_iter + 1,
        false,
        tolerance,
    ))
}

/// Fits a drPLS baseline.
///
/// # References
///
/// - D. Xu et al., "Baseline correction method based on doubly reweighted
///   penalized least squares", *Applied Optics*, 2019.
/// - `pybaselines.Baseline.drpls` is used as a behavioral reference.
pub fn drpls(y: &[f64], params: DrPlsParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = drpls_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a drPLS baseline into an existing output buffer.
pub fn drpls_into(y: &[f64], params: DrPlsParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "drpls",
            len: y.len(),
            min: 3,
        });
    }
    params.validate()?;

    let n = y.len();
    let mut workspace = crate::whittaker::WhittakerWorkspace::new(n);
    let mut band_workspace = GeneralPentadiagonalWorkspace::new(n);
    let mut lower2 = vec![0.0; n - 2];
    let mut lower1 = vec![0.0; n - 1];
    let mut diag = vec![0.0; n];
    let mut upper1 = vec![0.0; n - 1];
    let mut upper2 = vec![0.0; n - 2];
    let mut rhs = vec![0.0; n];
    let mut new_weights = vec![1.0; n];
    workspace.iter.weights.fill(1.0);

    let mut tolerance = f64::INFINITY;
    for iter in 0..=params.whittaker.max_iter {
        fill_drpls_bands(
            &workspace.iter.weights,
            params.whittaker.lambda,
            params.eta,
            &mut lower2,
            &mut lower1,
            &mut diag,
            &mut upper1,
            &mut upper2,
        );
        for ((target, observed), weight) in rhs.iter_mut().zip(y).zip(&workspace.iter.weights) {
            *target = observed * weight;
        }

        solve_general_pentadiagonal(
            GeneralPentadiagonalSystem {
                lower2: &lower2,
                lower1: &lower1,
                diag: &diag,
                upper1: &upper1,
                upper2: &upper2,
            },
            &rhs,
            baseline,
            &mut band_workspace,
        )?;

        if !drpls_weights(y, baseline, iter + 1, &mut new_weights) {
            break;
        }
        tolerance = relative_change(&workspace.iter.weights, &new_weights);
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        workspace.iter.weights.copy_from_slice(&new_weights);
    }

    Ok(FitReport::new(
        params.whittaker.max_iter + 1,
        false,
        tolerance,
    ))
}

/// Fits an IarPLS baseline.
///
/// # References
///
/// - J. Ye et al., "Baseline correction method based on improved
///   asymmetrically reweighted penalized least squares for Raman spectrum",
///   *Applied Optics*, 2020.
/// - `pybaselines.Baseline.iarpls` is used as a behavioral reference.
pub fn iarpls(y: &[f64], params: IarPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    fit_alloc(y, params.whittaker, IarPlsWeights)
}

/// Fits an asPLS baseline.
///
/// # References
///
/// - F. Zhang et al., "Baseline correction for infrared spectra using
///   adaptive smoothness parameter penalized least squares method",
///   *Spectroscopy Letters*, 2020.
/// - `pybaselines.Baseline.aspls` is used as a behavioral reference.
pub fn aspls(y: &[f64], params: AsPlsParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = aspls_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits an asPLS baseline and returns per-iteration tolerance history.
///
/// # References
///
/// - `pybaselines.Baseline.aspls` returns `tol_history`; this function exposes
///   the same diagnostic information in a typed Rust result.
pub fn aspls_with_history(y: &[f64], params: AsPlsParams) -> Result<FitHistory> {
    let mut baseline = vec![0.0; y.len()];
    let mut tol_history = Vec::with_capacity(params.whittaker.max_iter);
    let report = aspls_into_with_history(y, params, &mut baseline, &mut tol_history)?;
    Ok(FitHistory {
        baseline,
        report,
        tol_history,
    })
}

/// Fits an asPLS baseline into an existing output buffer.
pub fn aspls_into(y: &[f64], params: AsPlsParams, baseline: &mut [f64]) -> Result<FitReport> {
    aspls_into_impl(y, params, baseline, None)
}

/// Fits an asPLS baseline into an existing output buffer and records tolerance history.
pub fn aspls_into_with_history(
    y: &[f64],
    params: AsPlsParams,
    baseline: &mut [f64],
    tol_history: &mut Vec<f64>,
) -> Result<FitReport> {
    aspls_into_impl(y, params, baseline, Some(tol_history))
}

fn aspls_into_impl(
    y: &[f64],
    params: AsPlsParams,
    baseline: &mut [f64],
    mut tol_history: Option<&mut Vec<f64>>,
) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "aspls",
            len: y.len(),
            min: 3,
        });
    }
    params.validate()?;

    let n = y.len();
    let mut workspace = crate::whittaker::WhittakerWorkspace::new(n);
    let mut band_workspace = GeneralPentadiagonalWorkspace::new(n);
    let mut alpha = vec![1.0; n];
    let mut lower2 = vec![0.0; n - 2];
    let mut lower1 = vec![0.0; n - 1];
    let mut diag = vec![0.0; n];
    let mut upper1 = vec![0.0; n - 1];
    let mut upper2 = vec![0.0; n - 2];
    let mut rhs = vec![0.0; n];
    let mut new_weights = vec![1.0; n];
    workspace.iter.weights.fill(1.0);
    if let Some(history) = tol_history.as_deref_mut() {
        history.clear();
    }

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.whittaker.max_iter {
        fill_aspls_bands(
            &workspace.iter.weights,
            &alpha,
            params.whittaker.lambda,
            &mut lower2,
            &mut lower1,
            &mut diag,
            &mut upper1,
            &mut upper2,
        );
        for ((target, observed), weight) in rhs.iter_mut().zip(y).zip(&workspace.iter.weights) {
            *target = observed * weight;
        }

        solve_general_pentadiagonal(
            GeneralPentadiagonalSystem {
                lower2: &lower2,
                lower1: &lower1,
                diag: &diag,
                upper1: &upper1,
                upper2: &upper2,
            },
            &rhs,
            baseline,
            &mut band_workspace,
        )?;

        if !aspls_weights(
            y,
            baseline,
            params.asymmetric_coef,
            &mut new_weights,
            &mut workspace.iter.residual,
        ) {
            break;
        }
        tolerance = relative_change(&workspace.iter.weights, &new_weights);
        if let Some(history) = tol_history.as_deref_mut() {
            history.push(tolerance);
        }
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        workspace.iter.weights.copy_from_slice(&new_weights);
        let max_abs = workspace
            .iter
            .residual
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        for (target, residual) in alpha.iter_mut().zip(&workspace.iter.residual) {
            *target = residual.abs() / max_abs;
        }
    }

    Ok(FitReport::new(params.whittaker.max_iter, false, tolerance))
}

/// Fits a psalsa baseline.
///
/// # References
///
/// - `pybaselines.Baseline.psalsa` is used as a behavioral reference.
pub fn psalsa(y: &[f64], params: PsalsaParams) -> Result<Fit> {
    params.validate()?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }
    fit_alloc(y, params.whittaker, PsalsaWeights { p: params.p, k })
}

/// Fits a derpsalsa baseline.
///
/// # References
///
/// - V. Korepanov, "Asymmetric least-squares baseline algorithm with peak
///   screening for automatic processing of the Raman spectra", *Journal of
///   Raman Spectroscopy*, 2020.
/// - `pybaselines.Baseline.derpsalsa` is used as a behavioral reference.
pub fn derpsalsa(y: &[f64], params: DerPsalsaParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }
    let partial_weights = derivative_peak_screening_weights(
        y,
        params.smooth_half_window.unwrap_or(y.len() / 200),
        params.num_smooths,
    );
    fit_alloc(
        y,
        params.whittaker,
        DerPsalsaWeights {
            p: params.p,
            k,
            partial_weights,
        },
    )
}

/// Fits a brPLS baseline.
///
/// # References
///
/// - Q. Wang et al., "Spectral baseline estimation using penalized least
///   squares with weights derived from the Bayesian method", *Nuclear Science
///   and Techniques*, 2022.
/// - `pybaselines.Baseline.brpls` is used as a behavioral reference.
pub fn brpls(y: &[f64], params: BrPlsParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = brpls_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits a brPLS baseline into an existing output buffer.
pub fn brpls_into(y: &[f64], params: BrPlsParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "brpls",
            len: y.len(),
            min: 3,
        });
    }
    params.validate()?;

    let mut workspace = crate::whittaker::WhittakerWorkspace::new(y.len());
    workspace.iter.weights.fill(1.0);
    let mut current_baseline = y.to_vec();
    let mut candidate = vec![0.0; y.len()];
    let mut new_weights = vec![1.0; y.len()];
    let mut beta = 0.5;
    let mut tolerance = f64::INFINITY;
    let mut outer_tolerance = f64::INFINITY;
    let mut total_iterations = 0usize;

    'outer: for outer in 0..=params.max_iter_2 {
        for inner in 0..=params.whittaker.max_iter {
            crate::linalg::pentadiagonal::solve_second_order(
                y,
                &workspace.iter.weights,
                params.whittaker.lambda,
                &mut candidate,
                &mut workspace.solver,
            )?;
            total_iterations += 1;

            if !brpls_weights(y, &candidate, beta, &mut new_weights) {
                break 'outer;
            }

            tolerance = relative_change(&current_baseline, &candidate);
            if tolerance < params.whittaker.tol {
                if outer == 0 && inner == 0 {
                    current_baseline.copy_from_slice(&candidate);
                }
                break;
            }

            workspace.iter.weights.copy_from_slice(&new_weights);
            current_baseline.copy_from_slice(&candidate);
        }

        workspace.iter.weights.copy_from_slice(&new_weights);
        let weight_mean =
            workspace.iter.weights.iter().sum::<f64>() / workspace.iter.weights.len() as f64;
        outer_tolerance = (beta + weight_mean - 1.0).abs();
        if outer_tolerance < params.tol_2 {
            baseline.copy_from_slice(&current_baseline);
            return Ok(FitReport::new(total_iterations, true, outer_tolerance));
        }
        beta = 1.0 - weight_mean;
    }

    baseline.copy_from_slice(&current_baseline);
    Ok(FitReport::new(
        total_iterations,
        outer_tolerance <= params.tol_2,
        outer_tolerance.max(tolerance),
    ))
}

/// Fits an lsrPLS baseline.
///
/// # References
///
/// - Z. Heng et al., "Baseline correction for Raman spectra based on locally
///   symmetric reweighted penalized least squares", *Chinese Journal of
///   Lasers*, 2018.
/// - `pybaselines.Baseline.lsrpls` is used as a behavioral reference.
pub fn lsrpls(y: &[f64], params: LsrPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    fit_alloc(y, params.whittaker, LsrPlsWeights)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PsalsaWeights {
    pub(crate) p: f64,
    pub(crate) k: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct DerPsalsaWeights {
    pub(crate) p: f64,
    pub(crate) k: f64,
    pub(crate) partial_weights: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct IarPlsWeights;

#[derive(Debug, Clone, Copy)]
pub(crate) struct LsrPlsWeights;

impl Reweighter for PsalsaWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], _iter: usize) -> f64 {
        self.update_masked(y, baseline, weights, 0, None)
    }

    fn update_masked(
        &self,
        y: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        active_mask: Option<&[bool]>,
    ) -> f64 {
        let previous = weights.to_vec();
        for (index, ((weight, observed), fitted)) in
            weights.iter_mut().zip(y).zip(baseline).enumerate()
        {
            if !active_at(active_mask, index) {
                *weight = 0.0;
                continue;
            }
            let residual = observed - fitted;
            *weight = if residual > 0.0 {
                self.p * (-residual / self.k).exp()
            } else {
                1.0 - self.p
            };
        }
        relative_change(&previous, weights)
    }
}

impl Reweighter for DerPsalsaWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], _iter: usize) -> f64 {
        self.update_masked(y, baseline, weights, 0, None)
    }

    fn update_masked(
        &self,
        y: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        active_mask: Option<&[bool]>,
    ) -> f64 {
        let previous = weights.to_vec();
        for (index, (((weight, observed), fitted), partial)) in weights
            .iter_mut()
            .zip(y)
            .zip(baseline)
            .zip(&self.partial_weights)
            .enumerate()
        {
            if !active_at(active_mask, index) {
                *weight = 0.0;
                continue;
            }
            let residual = observed - fitted;
            let asymmetric = if residual > 0.0 {
                self.p * (-0.5 * (residual / self.k).powi(2)).exp()
            } else {
                1.0 - self.p
            };
            *weight = asymmetric * partial;
        }
        relative_change(&previous, weights)
    }
}

impl Reweighter for IarPlsWeights {
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
        let Some((_mean, std)) = negative_residual_stats_masked(y, baseline, active_mask) else {
            return 0.0;
        };
        let factor = (iter + 1).min(100) as f64;
        let scale = factor.exp() / std.max(f64::EPSILON);

        for (index, ((weight, observed), fitted)) in
            weights.iter_mut().zip(y).zip(baseline).enumerate()
        {
            if !active_at(active_mask, index) {
                *weight = 0.0;
                continue;
            }
            let residual = observed - fitted;
            let inner = scale * (residual - 2.0 * std);
            *weight = 0.5 * (1.0 - inner / (1.0 + inner * inner).sqrt());
        }

        relative_change(&previous, weights)
    }
}

impl Reweighter for LsrPlsWeights {
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
        let Some((mean, std)) = negative_residual_stats_masked(y, baseline, active_mask) else {
            return 0.0;
        };
        let scale = 10f64.powi((iter + 1).min(100) as i32) / std.max(f64::EPSILON);

        for (index, ((weight, observed), fitted)) in
            weights.iter_mut().zip(y).zip(baseline).enumerate()
        {
            if !active_at(active_mask, index) {
                *weight = 0.0;
                continue;
            }
            let residual = observed - fitted;
            let inner = scale * (residual - (2.0 * std - mean));
            *weight = 0.5 * (1.0 - inner / (1.0 + inner.abs()));
        }

        relative_change(&previous, weights)
    }
}

pub(crate) fn standard_deviation(values: &[f64]) -> f64 {
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

pub(crate) fn negative_residual_stats_masked(
    y: &[f64],
    baseline: &[f64],
    active_mask: Option<&[bool]>,
) -> Option<(f64, f64)> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for (index, (observed, fitted)) in y.iter().zip(baseline).enumerate() {
        if !active_at(active_mask, index) {
            continue;
        }
        let residual = observed - fitted;
        if residual < 0.0 {
            count += 1;
            sum += residual;
        }
    }
    if count < 2 {
        return None;
    }

    let mean = sum / count as f64;
    let mut sum_squares = 0.0;
    for (index, (observed, fitted)) in y.iter().zip(baseline).enumerate() {
        if !active_at(active_mask, index) {
            continue;
        }
        let residual = observed - fitted;
        if residual < 0.0 {
            let centered = residual - mean;
            sum_squares += centered * centered;
        }
    }
    let std = (sum_squares / (count - 1) as f64).sqrt();
    Some((mean, std))
}

fn drpls_weights(y: &[f64], baseline: &[f64], iteration: usize, weights: &mut [f64]) -> bool {
    drpls_weights_masked(y, baseline, iteration, weights, None)
}

pub(crate) fn drpls_weights_masked(
    y: &[f64],
    baseline: &[f64],
    iteration: usize,
    weights: &mut [f64],
    active_mask: Option<&[bool]>,
) -> bool {
    let Some((mean, std)) = negative_residual_stats_masked(y, baseline, active_mask) else {
        weights.fill(0.0);
        return false;
    };
    let scale = ((iteration.min(100) as f64).exp()) / std.max(f64::MIN_POSITIVE);
    for (index, ((weight, observed), fitted)) in weights.iter_mut().zip(y).zip(baseline).enumerate()
    {
        if !active_at(active_mask, index) {
            *weight = 0.0;
            continue;
        }
        let residual = observed - fitted;
        let inner = scale * (residual - (2.0 * std - mean));
        *weight = 0.5 * (1.0 - inner / (1.0 + inner.abs()));
    }
    true
}

fn aspls_weights(
    y: &[f64],
    baseline: &[f64],
    asymmetric_coef: f64,
    weights: &mut [f64],
    residuals: &mut [f64],
) -> bool {
    aspls_weights_masked(y, baseline, asymmetric_coef, weights, residuals, None)
}

pub(crate) fn aspls_weights_masked(
    y: &[f64],
    baseline: &[f64],
    asymmetric_coef: f64,
    weights: &mut [f64],
    residuals: &mut [f64],
    active_mask: Option<&[bool]>,
) -> bool {
    for (index, ((residual, observed), fitted)) in
        residuals.iter_mut().zip(y).zip(baseline).enumerate()
    {
        *residual = if active_at(active_mask, index) {
            observed - fitted
        } else {
            0.0
        };
    }
    let Some(std) = negative_residual_std_masked(residuals, active_mask) else {
        weights.fill(0.0);
        return false;
    };
    let scale = asymmetric_coef / std.max(f64::MIN_POSITIVE);
    for (index, (weight, residual)) in weights.iter_mut().zip(residuals).enumerate() {
        *weight = if active_at(active_mask, index) {
            logistic(-scale * (*residual - std))
        } else {
            0.0
        };
    }
    true
}

pub(crate) fn negative_residual_std_masked(
    residuals: &[f64],
    active_mask: Option<&[bool]>,
) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for (index, residual) in residuals.iter().enumerate() {
        if !active_at(active_mask, index) {
            continue;
        }
        if *residual < 0.0 {
            count += 1;
            sum += residual;
        }
    }
    if count < 2 {
        return None;
    }
    let mean = sum / count as f64;
    let mut sum_squares = 0.0;
    for (index, residual) in residuals.iter().enumerate() {
        if !active_at(active_mask, index) {
            continue;
        }
        if *residual < 0.0 {
            let centered = residual - mean;
            sum_squares += centered * centered;
        }
    }
    Some((sum_squares / (count - 1) as f64).sqrt())
}

pub(crate) fn asls_weight(observed: f64, fitted: f64, p: f64) -> f64 {
    if observed > fitted { p } else { 1.0 - p }
}

fn first_order_penalty_rhs(y: &[f64], lambda_1: f64, output: &mut [f64]) {
    output[0] = lambda_1 * (y[0] - y[1]);
    for i in 1..y.len() - 1 {
        output[i] = lambda_1 * (2.0 * y[i] - y[i - 1] - y[i + 1]);
    }
    let last = y.len() - 1;
    output[last] = lambda_1 * (y[last] - y[last - 1]);
}

#[allow(clippy::too_many_arguments)]
fn fill_drpls_bands(
    weights: &[f64],
    lambda: f64,
    eta: f64,
    lower2: &mut [f64],
    lower1: &mut [f64],
    diag: &mut [f64],
    upper1: &mut [f64],
    upper2: &mut [f64],
) {
    let n = weights.len();
    for (i, target) in diag.iter_mut().enumerate() {
        let second = second_order_diag(i, n, lambda);
        *target = first_order_diag(i, n) + second * (1.0 - eta * weights[i]) + weights[i];
    }
    for i in 0..n - 1 {
        let first = -1.0;
        let second = second_order_off1(i, n, lambda);
        upper1[i] = first + second * (1.0 - eta * weights[i]);
        lower1[i] = first + second * (1.0 - eta * weights[i + 1]);
    }
    for i in 0..n - 2 {
        upper2[i] = lambda * (1.0 - eta * weights[i]);
        lower2[i] = lambda * (1.0 - eta * weights[i + 2]);
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_aspls_bands(
    weights: &[f64],
    alpha: &[f64],
    lambda: f64,
    lower2: &mut [f64],
    lower1: &mut [f64],
    diag: &mut [f64],
    upper1: &mut [f64],
    upper2: &mut [f64],
) {
    let n = weights.len();
    for (i, target) in diag.iter_mut().enumerate() {
        *target = weights[i] + alpha[i] * second_order_diag(i, n, lambda);
    }
    for i in 0..n - 1 {
        let second = second_order_off1(i, n, lambda);
        upper1[i] = alpha[i] * second;
        lower1[i] = alpha[i + 1] * second;
    }
    for i in 0..n - 2 {
        upper2[i] = alpha[i] * lambda;
        lower2[i] = alpha[i + 2] * lambda;
    }
}

fn first_order_diag(index: usize, len: usize) -> f64 {
    if index == 0 || index + 1 == len {
        1.0
    } else {
        2.0
    }
}

fn second_order_diag(index: usize, len: usize, lambda: f64) -> f64 {
    if index == 0 || index + 1 == len {
        lambda
    } else if index == 1 || index + 2 == len {
        5.0 * lambda
    } else {
        6.0 * lambda
    }
}

fn second_order_off1(index: usize, len: usize, lambda: f64) -> f64 {
    if index == 0 || index + 2 == len {
        -2.0 * lambda
    } else {
        -4.0 * lambda
    }
}

fn brpls_weights(y: &[f64], baseline: &[f64], beta: f64, weights: &mut [f64]) -> bool {
    brpls_weights_masked(y, baseline, beta, weights, None)
}

pub(crate) fn brpls_weights_masked(
    y: &[f64],
    baseline: &[f64],
    beta: f64,
    weights: &mut [f64],
    active_mask: Option<&[bool]>,
) -> bool {
    let mut positive_count = 0usize;
    let mut positive_sum = 0.0;
    let mut negative_count = 0usize;
    let mut negative_sum_squares = 0.0;

    for (index, (observed, fitted)) in y.iter().zip(baseline).enumerate() {
        if !active_at(active_mask, index) {
            continue;
        }
        let residual = observed - fitted;
        if residual > 0.0 {
            positive_count += 1;
            positive_sum += residual;
        } else if residual < 0.0 {
            negative_count += 1;
            negative_sum_squares += residual * residual;
        }
    }

    if positive_count < 2 || negative_count < 2 {
        weights.fill(0.0);
        return false;
    }

    let mean = positive_sum / positive_count as f64;
    let sigma = (negative_sum_squares / negative_count as f64)
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let denominator = (1.0 - beta).max(f64::MIN_POSITIVE);
    let multiplier = (beta * (0.5 * std::f64::consts::PI).sqrt() / denominator) * (sigma / mean);
    let max_inner = f64::MAX.ln().sqrt();
    let sqrt_two = std::f64::consts::SQRT_2;

    for (index, ((weight, observed), fitted)) in weights.iter_mut().zip(y).zip(baseline).enumerate()
    {
        if !active_at(active_mask, index) {
            *weight = 0.0;
            continue;
        }
        let residual = observed - fitted;
        let inner = residual / (sigma * sqrt_two) - sigma / (mean * sqrt_two);
        let clipped_inner = inner.clamp(-max_inner, max_inner);
        let mut partial = (clipped_inner * clipped_inner).exp();
        if multiplier >= 0.5 {
            partial = partial.min(f64::MAX / (2.0 * multiplier));
        }
        *weight = 1.0 / (1.0 + multiplier * (1.0 + libm::erf(inner)) * partial);
    }
    true
}

pub(crate) fn derivative_peak_screening_weights(
    y: &[f64],
    smooth_half_window: usize,
    num_smooths: usize,
) -> Vec<f64> {
    let smoothed = smooth_for_derivatives(y, smooth_half_window, num_smooths);
    let first = gradient(&smoothed);
    let second = gradient(&first);
    let first_rms = root_mean_square(&first).max(f64::MIN_POSITIVE);
    let second_rms = root_mean_square(&second).max(f64::MIN_POSITIVE);

    first
        .iter()
        .zip(&second)
        .map(|(first_deriv, second_deriv)| {
            (-0.5 * (first_deriv / first_rms).powi(2)).exp()
                * (-0.5 * (second_deriv / second_rms).powi(2)).exp()
        })
        .collect()
}

fn smooth_for_derivatives(y: &[f64], smooth_half_window: usize, num_smooths: usize) -> Vec<f64> {
    if smooth_half_window == 0 || num_smooths == 0 {
        return y.to_vec();
    }

    let kernel = mollifier_kernel(smooth_half_window);
    let mut current = extrapolate_pad(y, smooth_half_window);
    for _ in 0..num_smooths {
        current = convolve_reflect_same(&current, &kernel);
    }
    current[smooth_half_window..smooth_half_window + y.len()].to_vec()
}

fn mollifier_kernel(half_window: usize) -> Vec<f64> {
    if half_window == 0 {
        return vec![1.0];
    }
    let mut kernel = Vec::with_capacity(2 * half_window + 1);
    for index in 0..=2 * half_window {
        if index == 0 || index == 2 * half_window {
            kernel.push(0.0);
        } else {
            let x = (index as f64 - half_window as f64) / half_window as f64;
            kernel.push((-1.0 / (1.0 - x * x)).exp());
        }
    }
    let sum = kernel.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn extrapolate_pad(y: &[f64], pad: usize) -> Vec<f64> {
    if pad == 0 {
        return y.to_vec();
    }
    let left_slope = if y.len() > 1 { y[1] - y[0] } else { 0.0 };
    let right_slope = if y.len() > 1 {
        y[y.len() - 1] - y[y.len() - 2]
    } else {
        0.0
    };
    let mut output = Vec::with_capacity(y.len() + 2 * pad);
    for i in (1..=pad).rev() {
        output.push(y[0] - left_slope * i as f64);
    }
    output.extend_from_slice(y);
    let last = *y.last().unwrap_or(&0.0);
    for i in 1..=pad {
        output.push(last + right_slope * i as f64);
    }
    output
}

fn convolve_reflect_same(y: &[f64], kernel: &[f64]) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let mut output = vec![0.0; y.len()];
    for (i, target) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, weight) in kernel.iter().enumerate() {
            let offset = j as isize - radius as isize;
            let index = reflect_index(i as isize + offset, y.len());
            sum += weight * y[index];
        }
        *target = sum;
    }
    output
}

fn reflect_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len as isize - 2;
    let mut value = index.rem_euclid(period);
    if value >= len as isize {
        value = period - value;
    }
    value as usize
}

fn gradient(values: &[f64]) -> Vec<f64> {
    match values.len() {
        0 => Vec::new(),
        1 => vec![0.0],
        len => {
            let mut output = vec![0.0; len];
            output[0] = values[1] - values[0];
            output[len - 1] = values[len - 1] - values[len - 2];
            for i in 1..len - 1 {
                output[i] = 0.5 * (values[i + 1] - values[i - 1]);
            }
            output
        }
    }
}

fn root_mean_square(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum = values.iter().map(|value| value * value).sum::<f64>();
    (sum / values.len() as f64).sqrt()
}
