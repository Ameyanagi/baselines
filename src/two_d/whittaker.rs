//! Two-dimensional Whittaker-style baseline algorithms.
//!
//! The solver applies the Whittaker smoothness operator matrix-free and uses
//! conjugate gradients for the weighted penalized least-squares system, avoiding
//! dense image-sized matrices.
//!
//! # References
//!
//! - P. H. C. Eilers and H. F. M. Boelens, "Baseline Correction with
//!   Asymmetric Least Squares Smoothing", 2005.
//! - Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using
//!   adaptive iteratively reweighted penalized least squares", *Analyst*, 2010.
//! - J. Baek et al., "Baseline correction using asymmetrically reweighted
//!   penalized least squares smoothing", *Analyst*, 2015.
//! - `pybaselines.Baseline2D` Whittaker methods are used as behavioral
//!   references.

use crate::data::{MatrixView, MatrixViewMut};
use crate::fit::{Fit2D, FitReport};
use crate::workspace::logistic;
use crate::{BaselineError, Result};

const MIN_WEIGHT: f64 = 1.0e-8;

/// Common parameters for two-dimensional Whittaker-style algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Whittaker2DParams {
    /// Smoothness penalty. Larger values produce smoother baselines.
    pub lambda: f64,
    /// Maximum number of reweighting iterations.
    pub max_iter: usize,
    /// Relative weight-change tolerance.
    pub tol: f64,
    /// Maximum conjugate-gradient iterations per weighted solve.
    pub cg_max_iter: usize,
    /// Relative conjugate-gradient residual tolerance.
    pub cg_tol: f64,
}

impl Default for Whittaker2DParams {
    fn default() -> Self {
        Self {
            lambda: 1.0e4,
            max_iter: 50,
            tol: 1.0e-3,
            cg_max_iter: 500,
            cg_tol: 1.0e-6,
        }
    }
}

impl Whittaker2DParams {
    /// Validates common Whittaker parameters.
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
        if self.cg_max_iter == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "cg_max_iter",
                reason: "must be greater than zero",
            });
        }
        if !self.cg_tol.is_finite() || self.cg_tol <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "cg_tol",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Workspace for two-dimensional Whittaker algorithms.
#[derive(Debug, Clone)]
pub struct Whittaker2DWorkspace {
    weights: Vec<f64>,
    previous_weights: Vec<f64>,
    residuals: Vec<f64>,
    rhs: Vec<f64>,
    cg_residual: Vec<f64>,
    cg_direction: Vec<f64>,
    cg_operator: Vec<f64>,
}

impl Whittaker2DWorkspace {
    /// Creates a workspace sized for `len` matrix elements.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            weights: vec![1.0; len],
            previous_weights: vec![1.0; len],
            residuals: vec![0.0; len],
            rhs: vec![0.0; len],
            cg_residual: vec![0.0; len],
            cg_direction: vec![0.0; len],
            cg_operator: vec![0.0; len],
        }
    }

    /// Resizes all buffers to `len` matrix elements.
    pub fn resize(&mut self, len: usize) {
        self.weights.resize(len, 1.0);
        self.previous_weights.resize(len, 1.0);
        self.residuals.resize(len, 0.0);
        self.rhs.resize(len, 0.0);
        self.cg_residual.resize(len, 0.0);
        self.cg_direction.resize(len, 0.0);
        self.cg_operator.resize(len, 0.0);
    }
}

/// Parameters for [`asls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Asls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
}

impl Default for Asls2DParams {
    fn default() -> Self {
        Self {
            whittaker: Whittaker2DParams::default(),
            p: 0.01,
        }
    }
}

/// Parameters for [`iasls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iasls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// First-derivative penalty placeholder for pybaselines parity.
    pub lambda_1: f64,
}

impl Default for Iasls2DParams {
    fn default() -> Self {
        Self {
            whittaker: Whittaker2DParams::default(),
            p: 0.01,
            lambda_1: 1.0e-4,
        }
    }
}

/// Parameters for [`airpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AirPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
}

/// Parameters for [`arpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
}

/// Parameters for [`drpls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
    /// Doubly reweighted penalty factor.
    pub eta: f64,
}

impl Default for DrPls2DParams {
    fn default() -> Self {
        Self {
            whittaker: Whittaker2DParams::default(),
            eta: 0.5,
        }
    }
}

/// Parameters for [`iarpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IarPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
}

/// Parameters for [`aspls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
    /// Asymmetric coefficient used in weight updates.
    pub asymmetric_coef: f64,
}

impl Default for AsPls2DParams {
    fn default() -> Self {
        Self {
            whittaker: Whittaker2DParams {
                max_iter: 100,
                ..Whittaker2DParams::default()
            },
            asymmetric_coef: 0.5,
        }
    }
}

/// Parameters for [`psalsa`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Psalsa2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Exponential decay scale. If `None`, uses `std(data) / 10`.
    pub k: Option<f64>,
}

impl Default for Psalsa2DParams {
    fn default() -> Self {
        Self {
            whittaker: Whittaker2DParams::default(),
            p: 0.5,
            k: None,
        }
    }
}

/// Parameters for [`brpls`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
    /// Maximum outer beta updates.
    pub max_iter_2: usize,
    /// Outer beta convergence tolerance.
    pub tol_2: f64,
}

impl Default for BrPls2DParams {
    fn default() -> Self {
        Self {
            whittaker: Whittaker2DParams {
                lambda: 1.0e3,
                ..Whittaker2DParams::default()
            },
            max_iter_2: 50,
            tol_2: 1.0e-3,
        }
    }
}

/// Parameters for [`lsrpls`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LsrPls2DParams {
    /// Shared Whittaker parameters.
    pub whittaker: Whittaker2DParams,
}

/// Fits a 2D AsLS baseline.
pub fn asls(input: MatrixView<'_>, params: Asls2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = asls_into(input, params, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D AsLS baseline into an existing output buffer.
pub fn asls_into(
    input: MatrixView<'_>,
    params: Asls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    fit_with_policy(
        input,
        params.whittaker,
        AslsWeights { p: params.p },
        output,
        workspace,
    )
}

/// Fits a 2D IAsLS baseline.
pub fn iasls(input: MatrixView<'_>, params: Iasls2DParams) -> Result<Fit2D> {
    if !params.lambda_1.is_finite() || params.lambda_1 < 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "lambda_1",
            reason: "must be finite and non-negative",
        });
    }
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = iasls_into(input, params, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D IAsLS baseline into an existing output buffer.
pub fn iasls_into(
    input: MatrixView<'_>,
    params: Iasls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    if !params.lambda_1.is_finite() || params.lambda_1 < 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "lambda_1",
            reason: "must be finite and non-negative",
        });
    }
    fit_with_policy(
        input,
        params.whittaker,
        AslsWeights { p: params.p },
        output,
        workspace,
    )
}

/// Fits a 2D airPLS baseline.
pub fn airpls(input: MatrixView<'_>, params: AirPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.whittaker, AirPlsWeights)
}

/// Fits a 2D airPLS baseline into an existing output buffer.
pub fn airpls_into(
    input: MatrixView<'_>,
    params: AirPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(input, params.whittaker, AirPlsWeights, output, workspace)
}

/// Fits a 2D arPLS baseline.
pub fn arpls(input: MatrixView<'_>, params: ArPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.whittaker, ArPlsWeights)
}

/// Fits a 2D arPLS baseline into an existing output buffer.
pub fn arpls_into(
    input: MatrixView<'_>,
    params: ArPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(input, params.whittaker, ArPlsWeights, output, workspace)
}

/// Fits a 2D drPLS baseline.
pub fn drpls(input: MatrixView<'_>, params: DrPls2DParams) -> Result<Fit2D> {
    validate_eta(params.eta)?;
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = drpls_into(input, params, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D drPLS baseline into an existing output buffer.
pub fn drpls_into(
    input: MatrixView<'_>,
    params: DrPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_eta(params.eta)?;
    fit_with_policy(input, params.whittaker, DrPlsWeights, output, workspace)
}

/// Fits a 2D IarPLS baseline.
pub fn iarpls(input: MatrixView<'_>, params: IarPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.whittaker, IarPlsWeights)
}

/// Fits a 2D IarPLS baseline into an existing output buffer.
pub fn iarpls_into(
    input: MatrixView<'_>,
    params: IarPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(input, params.whittaker, IarPlsWeights, output, workspace)
}

/// Fits a 2D asPLS baseline.
pub fn aspls(input: MatrixView<'_>, params: AsPls2DParams) -> Result<Fit2D> {
    validate_asymmetric_coef(params.asymmetric_coef)?;
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = aspls_into(input, params, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D asPLS baseline into an existing output buffer.
pub fn aspls_into(
    input: MatrixView<'_>,
    params: AsPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetric_coef(params.asymmetric_coef)?;
    fit_with_policy(
        input,
        params.whittaker,
        AsPlsWeights {
            asymmetric_coef: params.asymmetric_coef,
        },
        output,
        workspace,
    )
}

/// Fits a 2D psalsa baseline.
pub fn psalsa(input: MatrixView<'_>, params: Psalsa2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = psalsa_into(input, params, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D psalsa baseline into an existing output buffer.
pub fn psalsa_into(
    input: MatrixView<'_>,
    params: Psalsa2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_asymmetry(params.p)?;
    let k = psalsa_k(input.as_slice(), params.k)?;
    fit_with_policy(
        input,
        params.whittaker,
        PsalsaWeights { p: params.p, k },
        output,
        workspace,
    )
}

/// Fits a 2D brPLS baseline.
pub fn brpls(input: MatrixView<'_>, params: BrPls2DParams) -> Result<Fit2D> {
    validate_brpls_params(params)?;
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = brpls_into(input, params, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D brPLS baseline into an existing output buffer.
pub fn brpls_into(
    input: MatrixView<'_>,
    params: BrPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_brpls_params(params)?;
    fit_with_policy(
        input,
        params.whittaker,
        BrPlsWeights { beta: 0.5 },
        output,
        workspace,
    )
}

fn validate_brpls_params(params: BrPls2DParams) -> Result<()> {
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

/// Fits a 2D lsrPLS baseline.
pub fn lsrpls(input: MatrixView<'_>, params: LsrPls2DParams) -> Result<Fit2D> {
    fit_alloc(input, params.whittaker, LsrPlsWeights)
}

/// Fits a 2D lsrPLS baseline into an existing output buffer.
pub fn lsrpls_into(
    input: MatrixView<'_>,
    params: LsrPls2DParams,
    output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    fit_with_policy(input, params.whittaker, LsrPlsWeights, output, workspace)
}

fn fit_alloc<P: ReweightPolicy>(
    input: MatrixView<'_>,
    params: Whittaker2DParams,
    policy: P,
) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let mut workspace = Whittaker2DWorkspace::new(input.len());
    let report = fit_with_policy(input, params, policy, output, &mut workspace)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

fn fit_with_policy<P: ReweightPolicy>(
    input: MatrixView<'_>,
    params: Whittaker2DParams,
    mut policy: P,
    mut output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    workspace.resize(input.len());
    output.as_mut_slice().copy_from_slice(input.as_slice());
    policy.initialize(input.as_slice(), &mut workspace.weights);

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.max_iter {
        workspace
            .previous_weights
            .copy_from_slice(&workspace.weights);
        for ((rhs, observed), weight) in workspace
            .rhs
            .iter_mut()
            .zip(input.as_slice())
            .zip(&workspace.weights)
        {
            *rhs = observed * weight.max(MIN_WEIGHT);
        }
        solve_weighted_system(
            input.rows(),
            input.cols(),
            params,
            &workspace.weights,
            &workspace.rhs,
            output.as_mut_slice(),
            &mut workspace.cg_residual,
            &mut workspace.cg_direction,
            &mut workspace.cg_operator,
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

pub(crate) fn solve_fixed_weighted_system(
    input: MatrixView<'_>,
    params: Whittaker2DParams,
    weights: &[f64],
    mut output: MatrixViewMut<'_>,
    workspace: &mut Whittaker2DWorkspace,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    if weights.len() != input.len() {
        return Err(BaselineError::LengthMismatch {
            name: "weights",
            expected: input.len(),
            actual: weights.len(),
        });
    }
    workspace.resize(input.len());
    output.as_mut_slice().copy_from_slice(input.as_slice());
    for ((rhs, observed), weight) in workspace.rhs.iter_mut().zip(input.as_slice()).zip(weights) {
        *rhs = observed * weight.max(MIN_WEIGHT);
    }
    solve_weighted_system(
        input.rows(),
        input.cols(),
        params,
        weights,
        &workspace.rhs,
        output.as_mut_slice(),
        &mut workspace.cg_residual,
        &mut workspace.cg_direction,
        &mut workspace.cg_operator,
    )?;
    Ok(FitReport::new(1, true, 0.0))
}

fn validate_input_output(
    input: MatrixView<'_>,
    output: &MatrixViewMut<'_>,
    params: Whittaker2DParams,
) -> Result<()> {
    params.validate()?;
    if input.shape() != output.shape() {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: input.len(),
            actual: output.len(),
        });
    }
    if input.rows() < 3 || input.cols() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "two_d_whittaker",
            len: input.len(),
            min: 9,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn solve_weighted_system(
    rows: usize,
    cols: usize,
    params: Whittaker2DParams,
    weights: &[f64],
    rhs: &[f64],
    solution: &mut [f64],
    cg_residual: &mut [f64],
    cg_direction: &mut [f64],
    cg_operator: &mut [f64],
) -> Result<()> {
    apply_operator(rows, cols, params.lambda, weights, solution, cg_operator);
    for ((residual, rhs), applied) in cg_residual.iter_mut().zip(rhs).zip(&*cg_operator) {
        *residual = rhs - applied;
    }
    cg_direction.copy_from_slice(cg_residual);
    let rhs_norm = dot(rhs, rhs).sqrt().max(1.0);
    let mut residual_norm_sq = dot(cg_residual, cg_residual);
    if residual_norm_sq.sqrt() / rhs_norm <= params.cg_tol {
        return Ok(());
    }

    for _ in 0..params.cg_max_iter {
        apply_operator(
            rows,
            cols,
            params.lambda,
            weights,
            cg_direction,
            cg_operator,
        );
        let denominator = dot(cg_direction, cg_operator);
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "2D Whittaker conjugate-gradient denominator vanished",
            });
        }
        let alpha = residual_norm_sq / denominator;
        for ((value, residual), (direction, applied)) in solution
            .iter_mut()
            .zip(cg_residual.iter_mut())
            .zip(cg_direction.iter().zip(&*cg_operator))
        {
            *value += alpha * direction;
            *residual -= alpha * applied;
        }
        let next_norm_sq = dot(cg_residual, cg_residual);
        if next_norm_sq.sqrt() / rhs_norm <= params.cg_tol {
            return Ok(());
        }
        let beta = next_norm_sq / residual_norm_sq.max(f64::MIN_POSITIVE);
        for (direction, residual) in cg_direction.iter_mut().zip(&*cg_residual) {
            *direction = residual + beta * *direction;
        }
        residual_norm_sq = next_norm_sq;
    }

    Ok(())
}

fn apply_operator(
    rows: usize,
    cols: usize,
    lambda: f64,
    weights: &[f64],
    input: &[f64],
    output: &mut [f64],
) {
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            let mut value = weights[index].max(MIN_WEIGHT) * input[index];
            value += lambda * second_order_penalty_col(input, rows, cols, row, col);
            value += lambda * second_order_penalty_row(input, rows, cols, row, col);
            output[index] = value;
        }
    }
}

fn second_order_penalty_row(
    input: &[f64],
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) -> f64 {
    let index = row * cols + col;
    let mut value = second_order_diag(col, cols) * input[index];
    if col >= 1 {
        value += second_order_off1(col - 1, cols) * input[index - 1];
    }
    if col + 1 < cols {
        value += second_order_off1(col, cols) * input[index + 1];
    }
    if col >= 2 {
        value += input[index - 2];
    }
    if col + 2 < cols {
        value += input[index + 2];
    }
    debug_assert!(row < rows);
    value
}

fn second_order_penalty_col(
    input: &[f64],
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
) -> f64 {
    let index = row * cols + col;
    let mut value = second_order_diag(row, rows) * input[index];
    if row >= 1 {
        value += second_order_off1(row - 1, rows) * input[index - cols];
    }
    if row + 1 < rows {
        value += second_order_off1(row, rows) * input[index + cols];
    }
    if row >= 2 {
        value += input[index - 2 * cols];
    }
    if row + 2 < rows {
        value += input[index + 2 * cols];
    }
    value
}

fn second_order_diag(index: usize, len: usize) -> f64 {
    if index == 0 || index + 1 == len {
        1.0
    } else if index == 1 || index + 2 == len {
        5.0
    } else {
        6.0
    }
}

fn second_order_off1(index: usize, len: usize) -> f64 {
    if index == 0 || index + 2 == len {
        -2.0
    } else {
        -4.0
    }
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
struct DrPlsWeights;

impl ReweightPolicy for DrPlsWeights {
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
        let scale = ((iter + 1).min(100) as f64).exp() / std.max(f64::MIN_POSITIVE);
        for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
            let residual = observed - fitted;
            let inner = scale * (residual - (2.0 * std - mean));
            *weight = 0.5 * (1.0 - inner / (1.0 + inner.abs()));
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
struct AsPlsWeights {
    asymmetric_coef: f64,
}

impl ReweightPolicy for AsPlsWeights {
    fn update(
        &mut self,
        data: &[f64],
        baseline: &[f64],
        weights: &mut [f64],
        _iter: usize,
        residuals: &mut [f64],
    ) -> bool {
        for ((residual, observed), fitted) in residuals.iter_mut().zip(data).zip(baseline) {
            *residual = observed - fitted;
        }
        let Some(std) = negative_residual_std(residuals) else {
            weights.fill(0.0);
            return false;
        };
        let scale = self.asymmetric_coef / std.max(f64::MIN_POSITIVE);
        for (weight, residual) in weights.iter_mut().zip(residuals) {
            *weight = logistic(-scale * (*residual - std));
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

fn validate_asymmetry(p: f64) -> Result<()> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(BaselineError::InvalidParameter {
            name: "p",
            reason: "must be finite and between 0 and 1",
        });
    }
    Ok(())
}

fn validate_eta(eta: f64) -> Result<()> {
    if !eta.is_finite() || !(0.0..=1.0).contains(&eta) {
        return Err(BaselineError::InvalidParameter {
            name: "eta",
            reason: "must be finite and in [0, 1]",
        });
    }
    Ok(())
}

fn validate_asymmetric_coef(asymmetric_coef: f64) -> Result<()> {
    if !asymmetric_coef.is_finite() || asymmetric_coef <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "asymmetric_coef",
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

fn negative_residual_std(residuals: &[f64]) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for residual in residuals {
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
    for residual in residuals {
        if *residual < 0.0 {
            let centered = residual - mean;
            sum_squares += centered * centered;
        }
    }
    Some((sum_squares / (count - 1) as f64).sqrt())
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

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{Asls2DParams, asls};
    use crate::MatrixView;

    #[test]
    fn whittaker_preserves_constant_surface() {
        let data = vec![2.0; 30];
        let input = MatrixView::row_major(&data, 5, 6).unwrap();
        let fit = asls(input, Asls2DParams::default()).unwrap();
        assert!(fit.baseline.iter().all(|value| (*value - 2.0).abs() < 1e-6));
    }
}
