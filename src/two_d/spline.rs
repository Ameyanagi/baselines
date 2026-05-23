//! Two-dimensional penalized-spline baseline algorithms.
//!
//! These methods use separable row/column P-spline passes with iterative
//! reweighting. This keeps the dense linear solves limited to compact one-axis
//! spline bases instead of building a full image-sized tensor-product system.
//!
//! # References
//!
//! - P. H. C. Eilers, "A Perfect Smoother", *Analytical Chemistry*, 2003.
//! - P. H. C. Eilers and B. D. Marx, "Flexible smoothing with B-splines and
//!   penalties", *Statistical Science*, 1996.
//! - P. H. C. Eilers, I. D. Currie, and M. Durban, "Fast and compact smoothing
//!   on large multidimensional grids", *Computational Statistics & Data
//!   Analysis*, 2006.
//! - `pybaselines.Baseline2D` penalized-spline methods are used as behavioral
//!   references.

use crate::data::{MatrixView, MatrixViewMut};
use crate::fit::{Fit2D, FitReport};
use crate::linalg::pspline::PenalizedSpline;
use crate::workspace::logistic;
use crate::{BaselineError, Result};

const PSPLINE_DEGREE: usize = 3;
const PSPLINE_DIFF_ORDER: usize = 2;
const IRSQR_DIFF_ORDER: usize = 3;
const MIN_WEIGHT: f64 = 1.0e-8;

/// Common parameters for two-dimensional penalized-spline algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spline2DParams {
    /// Smoothness penalty. Larger values produce smoother baselines.
    pub lambda: f64,
    /// Maximum number of reweighting iterations.
    pub max_iter: usize,
    /// Relative weight-change tolerance.
    pub tol: f64,
    /// Number of row-axis knots used by the separable P-spline pass.
    pub num_knots_rows: usize,
    /// Number of column-axis knots used by the separable P-spline pass.
    pub num_knots_cols: usize,
}

impl Default for Spline2DParams {
    fn default() -> Self {
        Self {
            lambda: 1.0e3,
            max_iter: 50,
            tol: 1.0e-3,
            num_knots_rows: 8,
            num_knots_cols: 8,
        }
    }
}

impl Spline2DParams {
    /// Validates common spline parameters.
    pub fn validate(self) -> Result<()> {
        if !self.lambda.is_finite() || self.lambda <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "lambda",
                reason: "must be finite and positive",
            });
        }
        if self.max_iter == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "max_iter",
                reason: "must be greater than zero",
            });
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "tol",
                reason: "must be finite and positive",
            });
        }
        if self.num_knots_rows < 2 {
            return Err(BaselineError::InvalidParameter {
                name: "num_knots_rows",
                reason: "must be at least two",
            });
        }
        if self.num_knots_cols < 2 {
            return Err(BaselineError::InvalidParameter {
                name: "num_knots_cols",
                reason: "must be at least two",
            });
        }
        Ok(())
    }
}

/// Workspace for two-dimensional penalized-spline algorithms.
#[derive(Debug, Clone)]
pub struct Spline2DWorkspace {
    weights: Vec<f64>,
    previous_weights: Vec<f64>,
    residuals: Vec<f64>,
    temp: Vec<f64>,
    row_weights: Vec<f64>,
    column_values: Vec<f64>,
    column_weights: Vec<f64>,
}

impl Spline2DWorkspace {
    /// Creates a workspace sized for a `rows` by `cols` matrix.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        let len = rows.saturating_mul(cols);
        Self {
            weights: vec![1.0; len],
            previous_weights: vec![1.0; len],
            residuals: vec![0.0; len],
            temp: vec![0.0; len],
            row_weights: vec![1.0; cols],
            column_values: vec![0.0; rows],
            column_weights: vec![1.0; rows],
        }
    }

    /// Resizes all buffers to a `rows` by `cols` matrix.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        let len = rows * cols;
        self.weights.resize(len, 1.0);
        self.previous_weights.resize(len, 1.0);
        self.residuals.resize(len, 0.0);
        self.temp.resize(len, 0.0);
        self.row_weights.resize(cols, 1.0);
        self.column_values.resize(rows, 0.0);
        self.column_weights.resize(rows, 1.0);
    }
}

/// Parameters for [`pspline_asls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsplineAsls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
}

impl Default for PsplineAsls2DParams {
    fn default() -> Self {
        Self {
            spline: Spline2DParams::default(),
            p: 0.01,
        }
    }
}

/// Parameters for [`pspline_iasls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsplineIasls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// First-difference penalty for the residual.
    pub lambda_1: f64,
}

impl Default for PsplineIasls2DParams {
    fn default() -> Self {
        Self {
            spline: Spline2DParams::default(),
            p: 0.01,
            lambda_1: 1.0e-4,
        }
    }
}

/// Parameters for [`pspline_airpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PsplineAirPls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
}

/// Parameters for [`pspline_arpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PsplineArPls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
}

/// Parameters for [`pspline_iarpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PsplineIarPls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
}

/// Parameters for [`pspline_psalsa`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsplinePsalsa2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Exponential decay scale. If `None`, uses `std(data) / 10`.
    pub k: Option<f64>,
}

impl Default for PsplinePsalsa2DParams {
    fn default() -> Self {
        Self {
            spline: Spline2DParams::default(),
            p: 0.5,
            k: None,
        }
    }
}

/// Parameters for [`pspline_brpls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsplineBrPls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
    /// Maximum outer beta updates.
    pub max_iter_2: usize,
    /// Outer beta convergence tolerance.
    pub tol_2: f64,
}

impl Default for PsplineBrPls2DParams {
    fn default() -> Self {
        Self {
            spline: Spline2DParams {
                max_iter: 20,
                ..Spline2DParams::default()
            },
            max_iter_2: 10,
            tol_2: 1.0e-3,
        }
    }
}

/// Parameters for [`pspline_lsrpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PsplineLsrPls2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
}

/// Parameters for [`irsqr`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Irsqr2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
    /// Quantile in `(0, 1)` to fit.
    pub quantile: f64,
    /// Residual floor used to avoid singular weights. If `None`, a
    /// scale-aware default is used.
    pub epsilon: Option<f64>,
}

impl Default for Irsqr2DParams {
    fn default() -> Self {
        Self {
            spline: Spline2DParams {
                max_iter: 20,
                ..Spline2DParams::default()
            },
            quantile: 0.05,
            epsilon: None,
        }
    }
}

/// Parameters for [`mixture_model`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixtureModel2DParams {
    /// Shared spline parameters.
    pub spline: Spline2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
}

impl Default for MixtureModel2DParams {
    fn default() -> Self {
        Self {
            spline: Spline2DParams {
                max_iter: 20,
                ..Spline2DParams::default()
            },
            p: 0.01,
        }
    }
}

/// Fits a 2D penalized-spline AsLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_asls` is used as a behavioral reference.
pub fn pspline_asls(input: MatrixView<'_>, params: PsplineAsls2DParams) -> Result<Fit2D> {
    validate_asymmetry(params.p)?;
    fit_alloc(
        input,
        params.spline,
        AslsWeights { p: params.p },
        PSPLINE_DIFF_ORDER,
        0.0,
    )
}

/// Fits a 2D penalized-spline AsLS baseline into an existing output buffer.
pub fn pspline_asls_into(
    input: MatrixView<'_>,
    params: PsplineAsls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    fit_with_policy(
        input,
        params.spline,
        AslsWeights { p: params.p },
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline IAsLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_iasls` is used as a behavioral reference.
pub fn pspline_iasls(input: MatrixView<'_>, params: PsplineIasls2DParams) -> Result<Fit2D> {
    validate_asymmetry(params.p)?;
    validate_lambda_1(params.lambda_1)?;
    fit_alloc(
        input,
        params.spline,
        AslsWeights { p: params.p },
        PSPLINE_DIFF_ORDER,
        params.lambda_1,
    )
}

/// Fits a 2D penalized-spline IAsLS baseline into an existing output buffer.
pub fn pspline_iasls_into(
    input: MatrixView<'_>,
    params: PsplineIasls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    validate_lambda_1(params.lambda_1)?;
    fit_with_policy(
        input,
        params.spline,
        AslsWeights { p: params.p },
        PSPLINE_DIFF_ORDER,
        params.lambda_1,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline airPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_airpls` is used as a behavioral reference.
pub fn pspline_airpls(input: MatrixView<'_>, params: PsplineAirPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.spline, AirPlsWeights, PSPLINE_DIFF_ORDER, 0.0)
}

/// Fits a 2D penalized-spline airPLS baseline into an existing output buffer.
pub fn pspline_airpls_into(
    input: MatrixView<'_>,
    params: PsplineAirPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(
        input,
        params.spline,
        AirPlsWeights,
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline arPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_arpls` is used as a behavioral reference.
pub fn pspline_arpls(input: MatrixView<'_>, params: PsplineArPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.spline, ArPlsWeights, PSPLINE_DIFF_ORDER, 0.0)
}

/// Fits a 2D penalized-spline arPLS baseline into an existing output buffer.
pub fn pspline_arpls_into(
    input: MatrixView<'_>,
    params: PsplineArPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(
        input,
        params.spline,
        ArPlsWeights,
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline IarPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_iarpls` is used as a behavioral reference.
pub fn pspline_iarpls(input: MatrixView<'_>, params: PsplineIarPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.spline, IarPlsWeights, PSPLINE_DIFF_ORDER, 0.0)
}

/// Fits a 2D penalized-spline IarPLS baseline into an existing output buffer.
pub fn pspline_iarpls_into(
    input: MatrixView<'_>,
    params: PsplineIarPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(
        input,
        params.spline,
        IarPlsWeights,
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline psalsa baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_psalsa` is used as a behavioral reference.
pub fn pspline_psalsa(input: MatrixView<'_>, params: PsplinePsalsa2DParams) -> Result<Fit2D> {
    validate_asymmetry(params.p)?;
    let k = psalsa_k(input.as_slice(), params.k)?;
    fit_alloc(
        input,
        params.spline,
        PsalsaWeights { p: params.p, k },
        PSPLINE_DIFF_ORDER,
        0.0,
    )
}

/// Fits a 2D penalized-spline psalsa baseline into an existing output buffer.
pub fn pspline_psalsa_into(
    input: MatrixView<'_>,
    params: PsplinePsalsa2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    let k = psalsa_k(input.as_slice(), params.k)?;
    fit_with_policy(
        input,
        params.spline,
        PsalsaWeights { p: params.p, k },
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline brPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_brpls` is used as a behavioral reference.
pub fn pspline_brpls(input: MatrixView<'_>, params: PsplineBrPls2DParams) -> Result<Fit2D> {
    validate_brpls_params(params)?;
    fit_alloc(
        input,
        params.spline,
        BrPlsWeights { beta: 0.5 },
        PSPLINE_DIFF_ORDER,
        0.0,
    )
}

/// Fits a 2D penalized-spline brPLS baseline into an existing output buffer.
pub fn pspline_brpls_into(
    input: MatrixView<'_>,
    params: PsplineBrPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_brpls_params(params)?;
    fit_with_policy(
        input,
        params.spline,
        BrPlsWeights { beta: 0.5 },
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D penalized-spline lsrPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.pspline_lsrpls` is used as a behavioral reference.
pub fn pspline_lsrpls(input: MatrixView<'_>, params: PsplineLsrPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.spline, LsrPlsWeights, PSPLINE_DIFF_ORDER, 0.0)
}

/// Fits a 2D penalized-spline lsrPLS baseline into an existing output buffer.
pub fn pspline_lsrpls_into(
    input: MatrixView<'_>,
    params: PsplineLsrPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(
        input,
        params.spline,
        LsrPlsWeights,
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D iterative reweighted spline quantile-regression baseline.
///
/// # References
///
/// - Q. Han et al., "Iterative Reweighted Quantile Regression Using Augmented
///   Lagrangian Optimization for Baseline Correction", ICISCE, 2018.
/// - `pybaselines.Baseline2D.irsqr` is used as a behavioral reference.
pub fn irsqr(input: MatrixView<'_>, params: Irsqr2DParams) -> Result<Fit2D> {
    validate_quantile(params.quantile)?;
    fit_alloc(
        input,
        params.spline,
        IrsqrWeights {
            quantile: params.quantile,
            epsilon: params.epsilon,
        },
        IRSQR_DIFF_ORDER,
        0.0,
    )
}

/// Fits a 2D IRSQR baseline into an existing output buffer.
pub fn irsqr_into(
    input: MatrixView<'_>,
    params: Irsqr2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_quantile(params.quantile)?;
    fit_with_policy(
        input,
        params.spline,
        IrsqrWeights {
            quantile: params.quantile,
            epsilon: params.epsilon,
        },
        IRSQR_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

/// Fits a 2D spline mixture-model baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.mixture_model` is used as a behavioral reference.
pub fn mixture_model(input: MatrixView<'_>, params: MixtureModel2DParams) -> Result<Fit2D> {
    validate_asymmetry(params.p)?;
    fit_alloc(
        input,
        params.spline,
        AslsWeights { p: params.p },
        PSPLINE_DIFF_ORDER,
        0.0,
    )
}

/// Fits a 2D spline mixture-model baseline into an existing output buffer.
pub fn mixture_model_into(
    input: MatrixView<'_>,
    params: MixtureModel2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    fit_with_policy(
        input,
        params.spline,
        AslsWeights { p: params.p },
        PSPLINE_DIFF_ORDER,
        0.0,
        output,
        workspace,
    )
}

fn fit_alloc<P: ReweightPolicy>(
    input: MatrixView<'_>,
    params: Spline2DParams,
    policy: P,
    diff_order: usize,
    first_difference_lambda: f64,
) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Spline2DWorkspace::new(input.rows(), input.cols());
    let report = fit_with_policy(
        input,
        params,
        policy,
        diff_order,
        first_difference_lambda,
        output,
        &mut workspace,
    )?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

fn fit_with_policy<P: ReweightPolicy>(
    input: MatrixView<'_>,
    params: Spline2DParams,
    mut policy: P,
    diff_order: usize,
    first_difference_lambda: f64,
    mut output: MatrixViewMut<'_>,
    workspace: &mut Spline2DWorkspace,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    validate_first_difference_lambda(first_difference_lambda)?;
    workspace.resize(input.rows(), input.cols());
    output.as_mut_slice().copy_from_slice(input.as_slice());
    policy.initialize(input.as_slice(), &mut workspace.weights);

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.max_iter {
        workspace
            .previous_weights
            .copy_from_slice(&workspace.weights);
        solve_separable_pspline(
            input.rows(),
            input.cols(),
            params,
            diff_order,
            first_difference_lambda,
            input.as_slice(),
            &workspace.weights,
            output.as_mut_slice(),
            &mut workspace.temp,
            &mut workspace.row_weights,
            &mut workspace.column_values,
            &mut workspace.column_weights,
        )?;
        if !policy.update(
            input.as_slice(),
            output.as_slice(),
            &mut workspace.weights,
            iter,
            &mut workspace.residuals,
        ) {
            break;
        }
        tolerance = relative_change(&workspace.previous_weights, &workspace.weights);
        if tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

#[allow(clippy::too_many_arguments)]
fn solve_separable_pspline(
    rows: usize,
    cols: usize,
    params: Spline2DParams,
    diff_order: usize,
    first_difference_lambda: f64,
    data: &[f64],
    weights: &[f64],
    output: &mut [f64],
    temp: &mut [f64],
    row_weights: &mut [f64],
    column_values: &mut [f64],
    column_weights: &mut [f64],
) -> Result<()> {
    let row_spline = PenalizedSpline::new(
        cols,
        params.num_knots_cols.min(cols).max(2),
        PSPLINE_DEGREE,
        diff_order,
    );
    for row in 0..rows {
        let start = row * cols;
        for (target, weight) in row_weights.iter_mut().zip(&weights[start..start + cols]) {
            *target = weight.max(MIN_WEIGHT);
        }
        let smoothed = if first_difference_lambda > 0.0 {
            row_spline.solve_with_first_difference_penalty(
                &data[start..start + cols],
                row_weights,
                params.lambda,
                first_difference_lambda,
            )?
        } else {
            row_spline.solve(&data[start..start + cols], row_weights, params.lambda)?
        };
        temp[start..start + cols].copy_from_slice(&smoothed);
    }

    let column_spline = PenalizedSpline::new(
        rows,
        params.num_knots_rows.min(rows).max(2),
        PSPLINE_DEGREE,
        diff_order,
    );
    for col in 0..cols {
        for row in 0..rows {
            let index = row * cols + col;
            column_values[row] = temp[index];
            column_weights[row] = weights[index].max(MIN_WEIGHT);
        }
        let smoothed = if first_difference_lambda > 0.0 {
            column_spline.solve_with_first_difference_penalty(
                column_values,
                column_weights,
                params.lambda,
                first_difference_lambda,
            )?
        } else {
            column_spline.solve(column_values, column_weights, params.lambda)?
        };
        for (row, value) in smoothed.iter().enumerate() {
            output[row * cols + col] = *value;
        }
    }

    Ok(())
}

fn validate_input_output(
    input: MatrixView<'_>,
    output: &MatrixViewMut<'_>,
    params: Spline2DParams,
) -> Result<()> {
    params.validate()?;
    if input.shape() != output.shape() {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: input.len(),
            actual: output.len(),
        });
    }
    let min = PSPLINE_DEGREE + 2;
    if input.rows() < min || input.cols() < min {
        return Err(BaselineError::TooShort {
            algorithm: "two_d_spline",
            len: input.len(),
            min: min * min,
        });
    }
    Ok(())
}

trait ReweightPolicy {
    fn initialize(&mut self, _data: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        iter: usize,
        residuals: &mut [f64],
    ) -> bool;
}

#[derive(Debug, Clone, Copy)]
struct AslsWeights {
    p: f64,
}

impl ReweightPolicy for AslsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            *weight = if observed > fitted {
                self.p
            } else {
                1.0 - self.p
            };
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct AirPlsWeights;

impl ReweightPolicy for AirPlsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        let negative_sum = data
            .iter()
            .zip(baseline)
            .map(|(observed, fitted)| observed - fitted)
            .filter(|residual| *residual < 0.0)
            .map(f64::abs)
            .sum::<f64>();
        if negative_sum <= f64::EPSILON {
            weights.fill(1.0);
            return true;
        }
        let scale = (iter + 1) as f64 / negative_sum;
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            *weight = if residual >= 0.0 {
                0.0
            } else {
                (scale * residual.abs()).exp().min(1.0e12)
            };
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct ArPlsWeights;

impl ReweightPolicy for ArPlsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        let Some((mean, std)) = negative_residual_stats(data, baseline) else {
            weights.fill(1.0);
            return true;
        };
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            let exponent = 2.0 * (residual - (2.0 * std - mean)) / std.max(f64::EPSILON);
            *weight = 1.0 - logistic(exponent);
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct IarPlsWeights;

impl ReweightPolicy for IarPlsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        let Some((_mean, std)) = negative_residual_stats(data, baseline) else {
            weights.fill(0.0);
            return false;
        };
        let scale = ((iter + 1).min(100) as f64).exp() / std.max(f64::EPSILON);
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            let inner = scale * (residual - 2.0 * std);
            *weight = 0.5 * (1.0 - inner / (1.0 + inner * inner).sqrt());
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct PsalsaWeights {
    p: f64,
    k: f64,
}

impl ReweightPolicy for PsalsaWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            *weight = if residual > 0.0 {
                self.p * (-residual / self.k).exp()
            } else {
                1.0 - self.p
            };
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct BrPlsWeights {
    beta: f64,
}

impl ReweightPolicy for BrPlsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        let mut positive_count = 0usize;
        let mut positive_sum = 0.0;
        let mut negative_count = 0usize;
        let mut negative_sum_squares = 0.0;
        for (observed, fitted) in data.iter().zip(baseline) {
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
        let denominator = (1.0 - self.beta).max(f64::MIN_POSITIVE);
        let multiplier =
            (self.beta * (0.5 * std::f64::consts::PI).sqrt() / denominator) * (sigma / mean);
        let max_inner = f64::MAX.ln().sqrt();
        let sqrt_two = std::f64::consts::SQRT_2;
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
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
}

#[derive(Debug, Clone, Copy)]
struct LsrPlsWeights;

impl ReweightPolicy for LsrPlsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        let Some((mean, std)) = negative_residual_stats(data, baseline) else {
            weights.fill(0.0);
            return false;
        };
        let scale = 10f64.powi((iter + 1).min(100) as i32) / std.max(f64::EPSILON);
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            let inner = scale * (residual - (2.0 * std - mean));
            *weight = 0.5 * (1.0 - inner / (1.0 + inner.abs()));
        }
        true
    }
}

#[derive(Debug, Clone, Copy)]
struct IrsqrWeights {
    quantile: f64,
    epsilon: Option<f64>,
}

impl ReweightPolicy for IrsqrWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        _residuals: &mut [f64],
    ) -> bool {
        let epsilon = self
            .epsilon
            .unwrap_or_else(|| f64::EPSILON.sqrt() * signal_scale(data).max(1.0));
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            let quantile_weight = if residual >= 0.0 {
                self.quantile
            } else {
                1.0 - self.quantile
            };
            *weight = quantile_weight / residual.abs().max(epsilon);
        }
        true
    }
}

fn validate_asymmetry(p: f64) -> Result<()> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "p",
            reason: "must be finite and between 0 and 1",
        });
    }
    Ok(())
}

fn validate_quantile(quantile: f64) -> Result<()> {
    if !quantile.is_finite() || quantile <= 0.0 || quantile >= 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "quantile",
            reason: "must be finite and between 0 and 1",
        });
    }
    Ok(())
}

fn validate_lambda_1(lambda_1: f64) -> Result<()> {
    if !lambda_1.is_finite() || lambda_1 < 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "lambda_1",
            reason: "must be finite and non-negative",
        });
    }
    Ok(())
}

fn validate_first_difference_lambda(lambda: f64) -> Result<()> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "first_difference_lambda",
            reason: "must be finite and non-negative",
        });
    }
    Ok(())
}

fn validate_brpls_params(params: PsplineBrPls2DParams) -> Result<()> {
    if params.max_iter_2 == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "max_iter_2",
            reason: "must be greater than zero",
        });
    }
    if !params.tol_2.is_finite() || params.tol_2 <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "tol_2",
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

fn psalsa_k(data: &[f64], configured: Option<f64>) -> Result<f64> {
    let k = configured.unwrap_or_else(|| standard_deviation(data) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(data) / 10 must be finite and positive",
        });
    }
    Ok(k)
}

fn negative_residual_stats(data: &[f64], baseline: &[f64]) -> Option<(f64, f64)> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for (observed, fitted) in data.iter().zip(baseline) {
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
    for (observed, fitted) in data.iter().zip(baseline) {
        let residual = observed - fitted;
        if residual < 0.0 {
            let centered = residual - mean;
            sum_squares += centered * centered;
        }
    }
    Some((mean, (sum_squares / (count - 1) as f64).sqrt()))
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

fn signal_scale(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn relative_change(previous: &[f64], current: &[f64]) -> f64 {
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
    use super::{PsplineAsls2DParams, pspline_asls};
    use crate::MatrixView;

    #[test]
    fn pspline_preserves_constant_surface() {
        let data = vec![2.0; 30];
        let input = MatrixView::row_major(&data, 5, 6).unwrap();
        let fit = pspline_asls(input, PsplineAsls2DParams::default()).unwrap();
        assert!(fit.baseline.iter().all(|value| (*value - 2.0).abs() < 1e-6));
    }
}
