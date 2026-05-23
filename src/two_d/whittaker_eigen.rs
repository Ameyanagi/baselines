//! Reduced eigenspace solvers for two-dimensional Whittaker baselines.
//!
//! These routines mirror pybaselines' `num_eigens` path for 2D Whittaker
//! methods while keeping the public API Rust-native. The row and column
//! finite-difference penalties are diagonalized, and the weighted problem is
//! solved by conjugate gradients in the tensor-product eigenbasis.
//!
//! # References
//!
//! - G. Biessy, "Revisiting Whittaker-Henderson Smoothing", 2023.
//! - P. H. C. Eilers et al., "Fast and compact smoothing on large
//!   multidimensional grids", *Computational Statistics & Data Analysis*,
//!   2006.
//! - `pybaselines.Baseline2D.arpls` and its `num_eigens`/`return_dof`
//!   behavior are used as behavioral references.

use crate::data::MatrixView;
use crate::fit::{Fit2D, FitReport};
use crate::workspace::{logistic, validate_output};
use crate::{BaselineError, Result};

const MIN_WEIGHT: f64 = 1.0e-8;
const DEFAULT_CG_MAX_ITER: usize = 500;
const DEFAULT_CG_TOL: f64 = 1.0e-7;

/// Shared parameters for reduced eigenspace 2D Whittaker fits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Whittaker2DEigenParams {
    /// Smoothness penalty used for both axes unless axis-specific values are set.
    pub lambda: f64,
    /// Optional smoothness penalty along rows.
    pub lambda_rows: Option<f64>,
    /// Optional smoothness penalty along columns.
    pub lambda_cols: Option<f64>,
    /// Difference orders for rows and columns.
    ///
    /// The current implementation supports `(2, 2)`, matching the
    /// pybaselines eigendecomposition gallery example.
    pub diff_order: (usize, usize),
    /// Number of row and column eigenvectors to keep.
    pub num_eigens: (usize, usize),
    /// Whether to return effective degrees-of-freedom estimates.
    pub return_dof: bool,
    /// Maximum number of reweighting iterations.
    pub max_iter: usize,
    /// Relative weight-change tolerance.
    pub tol: f64,
    /// Maximum conjugate-gradient iterations for each reduced solve.
    pub cg_max_iter: usize,
    /// Relative conjugate-gradient residual tolerance for each reduced solve.
    pub cg_tol: f64,
}

impl Default for Whittaker2DEigenParams {
    fn default() -> Self {
        Self {
            lambda: 1.0e3,
            lambda_rows: None,
            lambda_cols: None,
            diff_order: (2, 2),
            num_eigens: (10, 10),
            return_dof: false,
            max_iter: 50,
            tol: 1.0e-3,
            cg_max_iter: DEFAULT_CG_MAX_ITER,
            cg_tol: DEFAULT_CG_TOL,
        }
    }
}

impl Whittaker2DEigenParams {
    fn validate(self, rows: usize, cols: usize) -> Result<()> {
        validate_positive("lambda", self.lambda)?;
        if let Some(lambda_rows) = self.lambda_rows {
            validate_positive("lambda_rows", lambda_rows)?;
        }
        if let Some(lambda_cols) = self.lambda_cols {
            validate_positive("lambda_cols", lambda_cols)?;
        }
        if self.diff_order != (2, 2) {
            return Err(BaselineError::Unsupported {
                feature: "two_d_whittaker_eigen",
                reason: "only second-order differences are currently supported",
            });
        }
        validate_num_eigens(
            "row eigenvalues",
            self.num_eigens.0,
            rows,
            self.diff_order.0,
        )?;
        validate_num_eigens(
            "column eigenvalues",
            self.num_eigens.1,
            cols,
            self.diff_order.1,
        )?;
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

    fn lambda_rows(self) -> f64 {
        self.lambda_rows.unwrap_or(self.lambda)
    }

    fn lambda_cols(self) -> f64 {
        self.lambda_cols.unwrap_or(self.lambda)
    }
}

/// Parameters for [`arpls_eigen`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArPls2DEigenParams {
    /// Shared reduced-eigenspace Whittaker parameters.
    pub whittaker: Whittaker2DEigenParams,
}

/// Two-dimensional eigenspace baseline output with convergence metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Whittaker2DEigenFit {
    /// Estimated row-major baseline.
    pub baseline: Vec<f64>,
    /// Number of matrix rows.
    pub rows: usize,
    /// Number of matrix columns.
    pub cols: usize,
    /// Fit metadata.
    pub report: FitReport,
    /// Final row-major weights.
    pub weights: Vec<f64>,
    /// Optional row-major effective degrees-of-freedom estimates in eigen space.
    pub dof: Option<Vec<f64>>,
    /// Number of row and column eigenvectors used for the reduced basis.
    pub num_eigens: (usize, usize),
}

impl Whittaker2DEigenFit {
    fn new(
        baseline: Vec<f64>,
        rows: usize,
        cols: usize,
        report: FitReport,
        weights: Vec<f64>,
        dof: Option<Vec<f64>>,
        num_eigens: (usize, usize),
    ) -> Result<Self> {
        Fit2D::new(baseline.clone(), rows, cols, report)?;
        validate_output("weights", rows * cols, weights.len())?;
        if let Some(dof) = &dof {
            validate_output("dof", num_eigens.0 * num_eigens.1, dof.len())?;
        }
        Ok(Self {
            baseline,
            rows,
            cols,
            report,
            weights,
            dof,
            num_eigens,
        })
    }

    /// Returns the matrix shape as `(rows, cols)`.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the eigenspace degrees-of-freedom shape.
    #[must_use]
    pub fn dof_shape(&self) -> (usize, usize) {
        self.num_eigens
    }

    /// Returns `data - baseline`.
    pub fn corrected(&self, data: &[f64]) -> Result<Vec<f64>> {
        validate_output("data", self.baseline.len(), data.len())?;
        Ok(data
            .iter()
            .zip(&self.baseline)
            .map(|(observed, baseline)| observed - baseline)
            .collect())
    }
}

/// Fits a 2D arPLS baseline in a reduced Whittaker eigenspace.
///
/// This is the Rust analogue of pybaselines' `Baseline2D.arpls(...,
/// num_eigens=..., return_dof=...)` path. The degrees of freedom are returned
/// as a diagonal estimate in the reduced basis to keep the Rust implementation
/// allocation-bounded for larger eigen counts.
///
/// # References
///
/// - S.-J. Baek et al., "Baseline correction using asymmetrically reweighted
///   penalized least squares smoothing", *Analyst*, 2015.
/// - G. Biessy, "Revisiting Whittaker-Henderson Smoothing", 2023.
/// - `pybaselines.Baseline2D.arpls` is used as a behavioral reference.
pub fn arpls_eigen(
    input: MatrixView<'_>,
    params: ArPls2DEigenParams,
) -> Result<Whittaker2DEigenFit> {
    let shared = params.whittaker;
    validate_input(input, shared)?;

    let system = ReducedWhittakerSystem::new(input.rows(), input.cols(), shared)?;
    let mut buffers = ReducedSolveBuffers::new(
        system.coeff_len(),
        input.len(),
        system.rows,
        system.cols_eigens,
    );
    let mut weights = vec![1.0; input.len()];
    let mut previous_weights = vec![1.0; input.len()];
    let mut baseline = input.as_slice().to_vec();
    let mut tolerance = f64::INFINITY;

    for iter in 0..shared.max_iter {
        previous_weights.copy_from_slice(&weights);
        solve_reduced_system(
            input.as_slice(),
            &weights,
            &system,
            shared,
            &mut buffers,
            &mut baseline,
        )?;
        update_arpls_weights(input.as_slice(), &baseline, &mut weights);
        tolerance = relative_change(&previous_weights, &weights);
        if tolerance <= shared.tol {
            let dof = shared
                .return_dof
                .then(|| diagonal_dof_estimate(&weights, &system, &mut buffers));
            return Whittaker2DEigenFit::new(
                baseline,
                input.rows(),
                input.cols(),
                FitReport::new(iter + 1, true, tolerance),
                weights,
                dof,
                shared.num_eigens,
            );
        }
    }

    let dof = shared
        .return_dof
        .then(|| diagonal_dof_estimate(&weights, &system, &mut buffers));
    Whittaker2DEigenFit::new(
        baseline,
        input.rows(),
        input.cols(),
        FitReport::new(shared.max_iter, false, tolerance),
        weights,
        dof,
        shared.num_eigens,
    )
}

struct ReducedWhittakerSystem {
    rows: usize,
    cols: usize,
    rows_eigens: usize,
    cols_eigens: usize,
    row_basis: Vec<f64>,
    col_basis: Vec<f64>,
    penalty: Vec<f64>,
}

impl ReducedWhittakerSystem {
    fn new(rows: usize, cols: usize, params: Whittaker2DEigenParams) -> Result<Self> {
        let row_basis = EigenBasis::second_order(rows, params.num_eigens.0)?;
        let col_basis = EigenBasis::second_order(cols, params.num_eigens.1)?;
        let mut penalty = vec![0.0; params.num_eigens.0 * params.num_eigens.1];
        for row_eigen in 0..params.num_eigens.0 {
            for col_eigen in 0..params.num_eigens.1 {
                penalty[row_eigen * params.num_eigens.1 + col_eigen] = params.lambda_rows()
                    * row_basis.values[row_eigen]
                    + params.lambda_cols() * col_basis.values[col_eigen];
            }
        }
        Ok(Self {
            rows,
            cols,
            rows_eigens: params.num_eigens.0,
            cols_eigens: params.num_eigens.1,
            row_basis: row_basis.vectors,
            col_basis: col_basis.vectors,
            penalty,
        })
    }

    fn coeff_len(&self) -> usize {
        self.rows_eigens * self.cols_eigens
    }
}

struct EigenBasis {
    values: Vec<f64>,
    vectors: Vec<f64>,
}

impl EigenBasis {
    fn second_order(points: usize, count: usize) -> Result<Self> {
        let matrix = second_order_penalty_matrix(points);
        let (values, vectors) = jacobi_eigendecomposition(matrix, points)?;
        let mut order: Vec<usize> = (0..points).collect();
        order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));

        let mut selected_values = vec![0.0; count];
        let mut selected_vectors = vec![0.0; points * count];
        for (target, source) in order.into_iter().take(count).enumerate() {
            selected_values[target] = if target < 2 {
                0.0
            } else {
                values[source].max(0.0)
            };
            for point in 0..points {
                selected_vectors[point * count + target] = vectors[point * points + source];
            }
        }

        Ok(Self {
            values: selected_values,
            vectors: selected_vectors,
        })
    }
}

struct ReducedSolveBuffers {
    coeff: Vec<f64>,
    rhs: Vec<f64>,
    residual: Vec<f64>,
    preconditioned: Vec<f64>,
    preconditioner: Vec<f64>,
    direction: Vec<f64>,
    operator: Vec<f64>,
    matrix: Vec<f64>,
    row_by_col_eigen: Vec<f64>,
}

impl ReducedSolveBuffers {
    fn new(coeff_len: usize, matrix_len: usize, rows: usize, cols_eigens: usize) -> Self {
        Self {
            coeff: vec![0.0; coeff_len],
            rhs: vec![0.0; coeff_len],
            residual: vec![0.0; coeff_len],
            preconditioned: vec![0.0; coeff_len],
            preconditioner: vec![1.0; coeff_len],
            direction: vec![0.0; coeff_len],
            operator: vec![0.0; coeff_len],
            matrix: vec![0.0; matrix_len],
            row_by_col_eigen: vec![0.0; rows * cols_eigens],
        }
    }
}

fn validate_input(input: MatrixView<'_>, params: Whittaker2DEigenParams) -> Result<()> {
    params.validate(input.rows(), input.cols())?;
    if input.rows() < 3 || input.cols() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "two_d_whittaker_eigen",
            len: input.len(),
            min: 9,
        });
    }
    Ok(())
}

fn validate_positive(name: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name,
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

fn validate_num_eigens(
    name: &'static str,
    value: usize,
    points: usize,
    diff_order: usize,
) -> Result<()> {
    if value > points {
        return Err(BaselineError::InvalidParameter {
            name,
            reason: "must not be greater than the number of points",
        });
    }
    if value <= diff_order {
        return Err(BaselineError::InvalidParameter {
            name,
            reason: "must be greater than the difference order",
        });
    }
    Ok(())
}

fn solve_reduced_system(
    data: &[f64],
    weights: &[f64],
    system: &ReducedWhittakerSystem,
    params: Whittaker2DEigenParams,
    buffers: &mut ReducedSolveBuffers,
    baseline: &mut [f64],
) -> Result<()> {
    project_weighted_data(
        data,
        weights,
        system,
        &mut buffers.rhs,
        &mut buffers.row_by_col_eigen,
    );
    reduced_diagonal(
        weights,
        system,
        &mut buffers.preconditioner,
        IncludePenalty::Yes,
    );
    apply_reduced_operator(
        &buffers.coeff,
        weights,
        system,
        &mut buffers.operator,
        &mut buffers.matrix,
        &mut buffers.row_by_col_eigen,
    );
    for ((residual, rhs), applied) in buffers
        .residual
        .iter_mut()
        .zip(&buffers.rhs)
        .zip(&buffers.operator)
    {
        *residual = rhs - applied;
    }
    for ((preconditioned, residual), diagonal) in buffers
        .preconditioned
        .iter_mut()
        .zip(&buffers.residual)
        .zip(&buffers.preconditioner)
    {
        *preconditioned = residual / diagonal.max(f64::MIN_POSITIVE);
    }
    buffers.direction.copy_from_slice(&buffers.preconditioned);
    let rhs_norm = dot(&buffers.rhs, &buffers.rhs).sqrt().max(1.0);
    let residual_norm_sq = dot(&buffers.residual, &buffers.residual);
    let mut residual_preconditioned_dot = dot(&buffers.residual, &buffers.preconditioned);

    if residual_norm_sq.sqrt() / rhs_norm <= params.cg_tol {
        reconstruct(
            system,
            &buffers.coeff,
            baseline,
            &mut buffers.row_by_col_eigen,
        );
        return Ok(());
    }

    for _ in 0..params.cg_max_iter {
        apply_reduced_operator(
            &buffers.direction,
            weights,
            system,
            &mut buffers.operator,
            &mut buffers.matrix,
            &mut buffers.row_by_col_eigen,
        );
        let denominator = dot(&buffers.direction, &buffers.operator);
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "2D Whittaker eigenspace conjugate-gradient denominator vanished",
            });
        }
        let alpha = residual_preconditioned_dot / denominator;
        for ((coeff, residual), (direction, applied)) in buffers
            .coeff
            .iter_mut()
            .zip(buffers.residual.iter_mut())
            .zip(buffers.direction.iter().zip(&buffers.operator))
        {
            *coeff += alpha * direction;
            *residual -= alpha * applied;
        }
        let next_norm_sq = dot(&buffers.residual, &buffers.residual);
        if next_norm_sq.sqrt() / rhs_norm <= params.cg_tol {
            reconstruct(
                system,
                &buffers.coeff,
                baseline,
                &mut buffers.row_by_col_eigen,
            );
            return Ok(());
        }
        for ((preconditioned, residual), diagonal) in buffers
            .preconditioned
            .iter_mut()
            .zip(&buffers.residual)
            .zip(&buffers.preconditioner)
        {
            *preconditioned = residual / diagonal.max(f64::MIN_POSITIVE);
        }
        let next_residual_preconditioned_dot = dot(&buffers.residual, &buffers.preconditioned);
        let beta =
            next_residual_preconditioned_dot / residual_preconditioned_dot.max(f64::MIN_POSITIVE);
        for (direction, preconditioned) in buffers.direction.iter_mut().zip(&buffers.preconditioned)
        {
            *direction = preconditioned + beta * *direction;
        }
        residual_preconditioned_dot = next_residual_preconditioned_dot;
    }

    reconstruct(
        system,
        &buffers.coeff,
        baseline,
        &mut buffers.row_by_col_eigen,
    );
    Ok(())
}

fn apply_reduced_operator(
    coeff: &[f64],
    weights: &[f64],
    system: &ReducedWhittakerSystem,
    output: &mut [f64],
    matrix: &mut [f64],
    row_by_col_eigen: &mut [f64],
) {
    reconstruct(system, coeff, matrix, row_by_col_eigen);
    for (value, weight) in matrix.iter_mut().zip(weights) {
        *value *= weight.max(MIN_WEIGHT);
    }
    project_matrix(matrix, system, output, row_by_col_eigen);
    for (index, value) in output.iter_mut().enumerate() {
        *value += system.penalty[index] * coeff[index];
    }
}

fn project_weighted_data(
    data: &[f64],
    weights: &[f64],
    system: &ReducedWhittakerSystem,
    output: &mut [f64],
    row_by_col_eigen: &mut [f64],
) {
    row_by_col_eigen.fill(0.0);
    for row in 0..system.rows {
        for col_eigen in 0..system.cols_eigens {
            let mut sum = 0.0;
            for col in 0..system.cols {
                let index = row * system.cols + col;
                sum += data[index]
                    * weights[index].max(MIN_WEIGHT)
                    * system.col_basis[col * system.cols_eigens + col_eigen];
            }
            row_by_col_eigen[row * system.cols_eigens + col_eigen] = sum;
        }
    }
    finish_projection(system, output, row_by_col_eigen);
}

fn project_matrix(
    matrix: &[f64],
    system: &ReducedWhittakerSystem,
    output: &mut [f64],
    row_by_col_eigen: &mut [f64],
) {
    row_by_col_eigen.fill(0.0);
    for row in 0..system.rows {
        for col_eigen in 0..system.cols_eigens {
            let mut sum = 0.0;
            for col in 0..system.cols {
                sum += matrix[row * system.cols + col]
                    * system.col_basis[col * system.cols_eigens + col_eigen];
            }
            row_by_col_eigen[row * system.cols_eigens + col_eigen] = sum;
        }
    }
    finish_projection(system, output, row_by_col_eigen);
}

fn finish_projection(
    system: &ReducedWhittakerSystem,
    output: &mut [f64],
    row_by_col_eigen: &[f64],
) {
    output.fill(0.0);
    for row_eigen in 0..system.rows_eigens {
        for col_eigen in 0..system.cols_eigens {
            let mut sum = 0.0;
            for row in 0..system.rows {
                sum += system.row_basis[row * system.rows_eigens + row_eigen]
                    * row_by_col_eigen[row * system.cols_eigens + col_eigen];
            }
            output[row_eigen * system.cols_eigens + col_eigen] = sum;
        }
    }
}

fn reconstruct(
    system: &ReducedWhittakerSystem,
    coeff: &[f64],
    output: &mut [f64],
    row_by_col_eigen: &mut [f64],
) {
    row_by_col_eigen.fill(0.0);
    for row in 0..system.rows {
        for col_eigen in 0..system.cols_eigens {
            let mut sum = 0.0;
            for row_eigen in 0..system.rows_eigens {
                sum += system.row_basis[row * system.rows_eigens + row_eigen]
                    * coeff[row_eigen * system.cols_eigens + col_eigen];
            }
            row_by_col_eigen[row * system.cols_eigens + col_eigen] = sum;
        }
    }
    for row in 0..system.rows {
        for col in 0..system.cols {
            let mut sum = 0.0;
            for col_eigen in 0..system.cols_eigens {
                sum += row_by_col_eigen[row * system.cols_eigens + col_eigen]
                    * system.col_basis[col * system.cols_eigens + col_eigen];
            }
            output[row * system.cols + col] = sum;
        }
    }
}

fn diagonal_dof_estimate(
    weights: &[f64],
    system: &ReducedWhittakerSystem,
    buffers: &mut ReducedSolveBuffers,
) -> Vec<f64> {
    let mut hat_diagonal = vec![0.0; system.coeff_len()];
    reduced_diagonal(weights, system, &mut hat_diagonal, IncludePenalty::No);
    reduced_diagonal(
        weights,
        system,
        &mut buffers.preconditioner,
        IncludePenalty::Yes,
    );
    let mut dof = vec![0.0; system.coeff_len()];
    for ((dof_value, hat_value), diagonal) in dof
        .iter_mut()
        .zip(&hat_diagonal)
        .zip(&buffers.preconditioner)
    {
        *dof_value = hat_value / diagonal.max(f64::MIN_POSITIVE);
    }
    buffers.matrix.fill(0.0);
    dof
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IncludePenalty {
    No,
    Yes,
}

fn reduced_diagonal(
    weights: &[f64],
    system: &ReducedWhittakerSystem,
    output: &mut [f64],
    include_penalty: IncludePenalty,
) {
    output.fill(0.0);
    for row_eigen in 0..system.rows_eigens {
        for col_eigen in 0..system.cols_eigens {
            let mut hat_diagonal = 0.0;
            for row in 0..system.rows {
                let row_value = system.row_basis[row * system.rows_eigens + row_eigen];
                let row_sq = row_value * row_value;
                for col in 0..system.cols {
                    let col_value = system.col_basis[col * system.cols_eigens + col_eigen];
                    let index = row * system.cols + col;
                    hat_diagonal += weights[index].max(MIN_WEIGHT) * row_sq * col_value * col_value;
                }
            }
            let coeff_index = row_eigen * system.cols_eigens + col_eigen;
            output[coeff_index] = if include_penalty == IncludePenalty::Yes {
                hat_diagonal + system.penalty[coeff_index]
            } else {
                hat_diagonal
            };
        }
    }
}

fn update_arpls_weights(data: &[f64], baseline: &[f64], weights: &mut [f64]) {
    let Some((mean, std)) = negative_residual_stats(data, baseline) else {
        weights.fill(1.0);
        return;
    };
    let denominator = std.max(f64::EPSILON);
    for ((weight, observed), fitted) in weights.iter_mut().zip(data).zip(baseline) {
        let residual = observed - fitted;
        let exponent = 2.0 * (residual - (2.0 * std - mean)) / denominator;
        *weight = 1.0 - logistic(exponent);
    }
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

fn second_order_penalty_matrix(points: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; points * points];
    for index in 0..points {
        matrix[index * points + index] = second_order_diag(index, points);
    }
    for index in 0..points.saturating_sub(1) {
        let value = second_order_off1(index, points);
        matrix[index * points + index + 1] = value;
        matrix[(index + 1) * points + index] = value;
    }
    for index in 0..points.saturating_sub(2) {
        matrix[index * points + index + 2] = 1.0;
        matrix[(index + 2) * points + index] = 1.0;
    }
    matrix
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

fn jacobi_eigendecomposition(mut matrix: Vec<f64>, size: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut vectors = vec![0.0; size * size];
    for index in 0..size {
        vectors[index * size + index] = 1.0;
    }
    let tolerance = 1.0e-12;
    let max_sweeps = 80;

    for _ in 0..max_sweeps {
        let mut max_off_diag = 0.0_f64;
        for row in 0..size.saturating_sub(1) {
            for col in row + 1..size {
                let value = matrix[row * size + col];
                max_off_diag = max_off_diag.max(value.abs());
                if value.abs() <= tolerance {
                    continue;
                }
                rotate_jacobi(&mut matrix, &mut vectors, size, row, col);
            }
        }
        if max_off_diag <= tolerance {
            let values = (0..size)
                .map(|index| matrix[index * size + index])
                .collect();
            return Ok((values, vectors));
        }
    }

    let values = (0..size)
        .map(|index| matrix[index * size + index])
        .collect();
    Ok((values, vectors))
}

fn rotate_jacobi(matrix: &mut [f64], vectors: &mut [f64], size: usize, row: usize, col: usize) {
    let off_diag = matrix[row * size + col];
    if off_diag.abs() <= f64::EPSILON {
        return;
    }
    let row_diag = matrix[row * size + row];
    let col_diag = matrix[col * size + col];
    let tau = (col_diag - row_diag) / (2.0 * off_diag);
    let tangent = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
    let sine = tangent * cosine;

    for index in 0..size {
        if index != row && index != col {
            let left = matrix[index * size + row];
            let right = matrix[index * size + col];
            let new_left = cosine * left - sine * right;
            let new_right = sine * left + cosine * right;
            matrix[index * size + row] = new_left;
            matrix[row * size + index] = new_left;
            matrix[index * size + col] = new_right;
            matrix[col * size + index] = new_right;
        }
    }

    matrix[row * size + row] = row_diag - tangent * off_diag;
    matrix[col * size + col] = col_diag + tangent * off_diag;
    matrix[row * size + col] = 0.0;
    matrix[col * size + row] = 0.0;

    for index in 0..size {
        let left = vectors[index * size + row];
        let right = vectors[index * size + col];
        vectors[index * size + row] = cosine * left - sine * right;
        vectors[index * size + col] = sine * left + cosine * right;
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}
