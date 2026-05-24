//! Two-dimensional polynomial baseline algorithms.
//!
//! # References
//!
//! - C. A. Lieber and A. Mahadevan-Jansen, "Automated Method for Subtraction
//!   of Fluorescence from Biological Raman Spectra", *Applied Spectroscopy*,
//!   2003.
//! - R. Koenker and G. Bassett Jr., "Regression Quantiles", *Econometrica*,
//!   1978.
//! - V. Mazet et al., "Background removal from spectra by designing and
//!   minimising a non-quadratic cost function", *Chemometrics and Intelligent
//!   Laboratory Systems*, 2005.
//! - `pybaselines.Baseline2D` polynomial methods are used as behavioral
//!   references.

use crate::data::{MatrixView, MatrixViewMut};
use crate::fit::{Fit2D, FitReport};
use crate::linalg::dense::solve_dense_in_place;
use crate::workspace::rms;
use crate::{BaselineError, Result};

/// Parameters for a direct two-dimensional polynomial baseline fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poly2DParams {
    /// Maximum polynomial degree on each axis.
    pub order: usize,
}

impl Default for Poly2DParams {
    fn default() -> Self {
        Self { order: 2 }
    }
}

/// Parameters for two-dimensional modified polynomial baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModPoly2DParams {
    /// Maximum polynomial degree on each axis.
    pub order: usize,
    /// Maximum number of clipped least-squares iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
}

impl Default for ModPoly2DParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 100,
            tol: 1.0e-3,
        }
    }
}

/// Parameters for two-dimensional improved modified polynomial baseline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImodPoly2DParams {
    /// Maximum polynomial degree on each axis.
    pub order: usize,
    /// Maximum number of clipped least-squares iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
}

impl Default for ImodPoly2DParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 100,
            tol: 1.0e-3,
        }
    }
}

/// Parameters for two-dimensional penalized polynomial fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PenalizedPoly2DParams {
    /// Maximum polynomial degree on each axis.
    pub order: usize,
    /// Maximum number of non-quadratic refinement iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
    /// Residual threshold. If `None`, uses `std(data) / 10`.
    pub threshold: Option<f64>,
    /// Scale factor for the asymmetric truncated quadratic penalty in `(0, 1]`.
    pub alpha_factor: f64,
}

impl Default for PenalizedPoly2DParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 250,
            tol: 1.0e-3,
            threshold: None,
            alpha_factor: 0.99,
        }
    }
}

/// Parameters for two-dimensional quantile-regression polynomial fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantReg2DParams {
    /// Maximum polynomial degree on each axis.
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

impl Default for QuantReg2DParams {
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

/// Fits a two-dimensional least-squares polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.poly` is used as a behavioral reference.
pub fn poly(input: MatrixView<'_>, params: Poly2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = poly_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a two-dimensional least-squares polynomial baseline into an output buffer.
pub fn poly_into(
    input: MatrixView<'_>,
    params: Poly2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_poly_input(input, &output, params.order)?;
    let weights = vec![1.0; input.len()];
    let mut workspace = SurfacePolynomialWorkspace::new(input.rows(), input.cols(), params.order);
    fit_weighted_polynomial_with_workspace(
        input,
        &weights,
        params.order,
        output.as_mut_slice(),
        &mut workspace,
    )?;
    Ok(FitReport::new(1, true, 0.0))
}

/// Fits a two-dimensional modified polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.modpoly` is used as a behavioral reference.
pub fn modpoly(input: MatrixView<'_>, params: ModPoly2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = modpoly_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a two-dimensional modified polynomial baseline into an output buffer.
pub fn modpoly_into(
    input: MatrixView<'_>,
    params: ModPoly2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_iter_params(params.max_iter, params.tol)?;
    validate_poly_input(input, &output, params.order)?;
    let mut clipped = input.as_slice().to_vec();
    let weights = vec![1.0; input.len()];
    let mut previous = vec![0.0; input.len()];
    let mut workspace = SurfacePolynomialWorkspace::new(input.rows(), input.cols(), params.order);
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(output.as_slice());
        fit_weighted_surface_with_workspace(
            &clipped,
            input.rows(),
            input.cols(),
            &weights,
            params.order,
            output.as_mut_slice(),
            &mut workspace,
        )?;
        tolerance = relative_change(&previous, output.as_slice());
        let residual: Vec<f64> = clipped
            .iter()
            .zip(output.as_slice())
            .map(|(observed, fitted)| observed - fitted)
            .collect();
        let spread = rms(&residual);
        for ((target, observed), fitted) in clipped
            .iter_mut()
            .zip(input.as_slice())
            .zip(output.as_slice())
        {
            *target = observed.min(fitted + spread);
        }
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

/// Fits a two-dimensional improved modified polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.imodpoly` is used as a behavioral reference.
pub fn imodpoly(input: MatrixView<'_>, params: ImodPoly2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = imodpoly_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a two-dimensional improved modified polynomial baseline into an output buffer.
pub fn imodpoly_into(
    input: MatrixView<'_>,
    params: ImodPoly2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_iter_params(params.max_iter, params.tol)?;
    validate_poly_input(input, &output, params.order)?;
    let mut clipped = input.as_slice().to_vec();
    let mut weights = vec![1.0; input.len()];
    let mut previous = vec![0.0; input.len()];
    let mut workspace = SurfacePolynomialWorkspace::new(input.rows(), input.cols(), params.order);
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(output.as_slice());
        fit_weighted_surface_with_workspace(
            &clipped,
            input.rows(),
            input.cols(),
            &weights,
            params.order,
            output.as_mut_slice(),
            &mut workspace,
        )?;
        tolerance = relative_change(&previous, output.as_slice());
        let residual: Vec<f64> = input
            .as_slice()
            .iter()
            .zip(output.as_slice())
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
            .zip(&mut weights)
            .zip(input.as_slice())
            .zip(output.as_slice())
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

/// Fits a two-dimensional penalized polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.penalized_poly` is used as a behavioral reference.
pub fn penalized_poly(input: MatrixView<'_>, params: PenalizedPoly2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = penalized_poly_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a two-dimensional penalized polynomial baseline into an output buffer.
pub fn penalized_poly_into(
    input: MatrixView<'_>,
    params: PenalizedPoly2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_penalized_params(params)?;
    validate_poly_input(input, &output, params.order)?;
    let threshold = params
        .threshold
        .unwrap_or_else(|| standard_deviation(input.as_slice()) / 10.0)
        .max(f64::EPSILON);
    let weights = vec![1.0; input.len()];
    let mut adjusted = input.as_slice().to_vec();
    let mut previous = vec![0.0; input.len()];
    let mut workspace = SurfacePolynomialWorkspace::new(input.rows(), input.cols(), params.order);
    let mut tolerance = f64::INFINITY;
    fit_weighted_polynomial_with_workspace(
        input,
        &weights,
        params.order,
        output.as_mut_slice(),
        &mut workspace,
    )?;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(output.as_slice());
        for ((target, observed), fitted) in adjusted
            .iter_mut()
            .zip(input.as_slice())
            .zip(output.as_slice())
        {
            let residual = observed - fitted;
            *target = observed
                + asymmetric_truncated_quadratic_correction(
                    residual,
                    threshold,
                    params.alpha_factor,
                );
        }
        fit_weighted_surface_with_workspace(
            &adjusted,
            input.rows(),
            input.cols(),
            &weights,
            params.order,
            output.as_mut_slice(),
            &mut workspace,
        )?;
        tolerance = relative_change(&previous, output.as_slice());
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

/// Fits a two-dimensional quantile-regression polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.quant_reg` is used as a behavioral reference.
pub fn quant_reg(input: MatrixView<'_>, params: QuantReg2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = quant_reg_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a two-dimensional quantile-regression polynomial baseline into an output buffer.
pub fn quant_reg_into(
    input: MatrixView<'_>,
    params: QuantReg2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_quant_reg_params(params)?;
    validate_poly_input(input, &output, params.order)?;
    let mut weights = vec![1.0; input.len()];
    let mut previous = vec![0.0; input.len()];
    let epsilon = params
        .epsilon
        .unwrap_or_else(|| f64::EPSILON.sqrt() * signal_scale(input.as_slice()).max(1.0));
    let mut workspace = SurfacePolynomialWorkspace::new(input.rows(), input.cols(), params.order);

    fit_weighted_polynomial_with_workspace(
        input,
        &weights,
        params.order,
        output.as_mut_slice(),
        &mut workspace,
    )?;
    let mut tolerance = f64::INFINITY;
    for iter in 0..params.max_iter {
        previous.copy_from_slice(output.as_slice());
        for ((weight, observed), fitted) in weights
            .iter_mut()
            .zip(input.as_slice())
            .zip(output.as_slice())
        {
            let residual = observed - fitted;
            let quantile_weight = if residual >= 0.0 {
                params.quantile
            } else {
                1.0 - params.quantile
            };
            *weight = quantile_weight / residual.abs().max(epsilon);
        }

        fit_weighted_polynomial_with_workspace(
            input,
            &weights,
            params.order,
            output.as_mut_slice(),
            &mut workspace,
        )?;
        tolerance = relative_change(&previous, output.as_slice());
        if iter > 0 && tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

fn validate_poly_input(
    input: MatrixView<'_>,
    output: &MatrixViewMut<'_>,
    order: usize,
) -> Result<()> {
    if input.shape() != output.shape() {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: input.len(),
            actual: output.len(),
        });
    }
    let basis_len = basis_terms(order).len();
    if basis_len > input.len() {
        return Err(BaselineError::TooShort {
            algorithm: "two_d_polynomial",
            len: input.len(),
            min: basis_len,
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

fn validate_penalized_params(params: PenalizedPoly2DParams) -> Result<()> {
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

fn validate_quant_reg_params(params: QuantReg2DParams) -> Result<()> {
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

fn fit_weighted_polynomial_with_workspace(
    input: MatrixView<'_>,
    weights: &[f64],
    order: usize,
    baseline: &mut [f64],
    workspace: &mut SurfacePolynomialWorkspace,
) -> Result<()> {
    fit_weighted_surface_with_workspace(
        input.as_slice(),
        input.rows(),
        input.cols(),
        weights,
        order,
        baseline,
        workspace,
    )
}

fn fit_weighted_surface_with_workspace(
    data: &[f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
    order: usize,
    baseline: &mut [f64],
    workspace: &mut SurfacePolynomialWorkspace,
) -> Result<()> {
    fit_weighted_coefficients_with_workspace(data, rows, cols, weights, order, workspace)?;
    evaluate_coefficients_with_workspace(baseline, workspace);
    Ok(())
}

fn fit_weighted_coefficients_with_workspace(
    data: &[f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
    order: usize,
    workspace: &mut SurfacePolynomialWorkspace,
) -> Result<()> {
    if weights.len() != data.len() {
        return Err(BaselineError::LengthMismatch {
            name: "weights",
            expected: data.len(),
            actual: weights.len(),
        });
    }
    workspace.ensure_basis(rows, cols, order);
    let n_coeffs = workspace.terms.len();
    workspace.normal.fill(0.0);
    workspace.rhs.fill(0.0);

    {
        let basis_values = &workspace.basis_values;
        let normal = &mut workspace.normal;
        let rhs = &mut workspace.rhs;

        for (index, (observed, weight)) in data.iter().zip(weights).enumerate() {
            let weight = (*weight).max(0.0);
            if weight > f64::EPSILON {
                let basis_offset = index * n_coeffs;
                let basis = &basis_values[basis_offset..basis_offset + n_coeffs];
                let weighted_observed = weight * *observed;
                for basis_row in 0..n_coeffs {
                    let row_value = basis[basis_row];
                    rhs[basis_row] += weighted_observed * row_value;
                    let weighted_row = weight * row_value;
                    for basis_col in 0..n_coeffs {
                        normal[basis_row * n_coeffs + basis_col] += weighted_row * basis[basis_col];
                    }
                }
            }
        }
    }

    solve_dense_in_place(
        &mut workspace.normal,
        &mut workspace.rhs,
        &mut workspace.coeffs,
        &mut workspace.pivot_row,
    )
}

fn evaluate_coefficients_with_workspace(
    baseline: &mut [f64],
    workspace: &SurfacePolynomialWorkspace,
) {
    let n_coeffs = workspace.terms.len();
    for (index, fitted) in baseline.iter_mut().enumerate() {
        let offset = index * n_coeffs;
        *fitted = workspace.coeffs[..n_coeffs]
            .iter()
            .zip(&workspace.basis_values[offset..offset + n_coeffs])
            .map(|(coefficient, basis)| coefficient * basis)
            .sum();
    }
}

fn basis_terms(order: usize) -> Vec<(usize, usize)> {
    let mut terms = Vec::with_capacity((order + 1) * (order + 1));
    for x_degree in 0..=order {
        for y_degree in 0..=order {
            terms.push((x_degree, y_degree));
        }
    }
    terms
}

fn scaled_axis(index: usize, len: usize) -> f64 {
    if len <= 1 {
        0.0
    } else {
        -1.0 + 2.0 * index as f64 / (len - 1) as f64
    }
}

#[derive(Debug, Default)]
struct SurfacePolynomialWorkspace {
    cached_basis: Option<(usize, usize, usize)>,
    terms: Vec<(usize, usize)>,
    basis_values: Vec<f64>,
    x_powers: Vec<f64>,
    y_powers: Vec<f64>,
    normal: Vec<f64>,
    rhs: Vec<f64>,
    coeffs: Vec<f64>,
    pivot_row: Vec<f64>,
}

impl SurfacePolynomialWorkspace {
    fn new(rows: usize, cols: usize, order: usize) -> Self {
        let mut workspace = Self::default();
        workspace.ensure_basis(rows, cols, order);
        workspace
    }

    fn ensure_basis(&mut self, rows: usize, cols: usize, order: usize) {
        if self.cached_basis == Some((rows, cols, order)) {
            return;
        }

        self.terms = basis_terms(order);
        let n_coeffs = self.terms.len();
        self.basis_values.resize(rows * cols * n_coeffs, 0.0);
        self.x_powers.resize(order + 1, 0.0);
        self.y_powers.resize(order + 1, 0.0);
        self.normal.resize(n_coeffs * n_coeffs, 0.0);
        self.rhs.resize(n_coeffs, 0.0);
        self.coeffs.resize(n_coeffs, 0.0);
        self.pivot_row.resize(n_coeffs, 0.0);

        for row in 0..rows {
            fill_axis_powers(scaled_axis(row, rows), &mut self.y_powers);
            for col in 0..cols {
                fill_axis_powers(scaled_axis(col, cols), &mut self.x_powers);
                let pixel_offset = (row * cols + col) * n_coeffs;
                for (term_index, (x_degree, y_degree)) in self.terms.iter().enumerate() {
                    self.basis_values[pixel_offset + term_index] =
                        self.x_powers[*x_degree] * self.y_powers[*y_degree];
                }
            }
        }

        self.cached_basis = Some((rows, cols, order));
    }
}

fn fill_axis_powers(value: f64, powers: &mut [f64]) {
    if powers.is_empty() {
        return;
    }
    powers[0] = 1.0;
    for degree in 1..powers.len() {
        powers[degree] = powers[degree - 1] * value;
    }
}

fn asymmetric_truncated_quadratic_correction(
    residual: f64,
    threshold: f64,
    alpha_factor: f64,
) -> f64 {
    let alpha = 0.5 * alpha_factor;
    if residual < threshold {
        residual * (2.0 * alpha - 1.0)
    } else {
        -residual
    }
}

fn signal_scale(values: &[f64]) -> f64 {
    let (min, max) = values
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

fn relative_change(previous: &[f64], next: &[f64]) -> f64 {
    let numerator = previous
        .iter()
        .zip(next)
        .map(|(left, right)| {
            let diff = right - left;
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
    use super::{Poly2DParams, poly};
    use crate::MatrixView;

    #[test]
    fn polynomial_recovers_planar_surface() {
        let rows = 4;
        let cols = 5;
        let data = (0..rows)
            .flat_map(|row| {
                (0..cols).map(move |col| {
                    let x = col as f64;
                    let y = row as f64;
                    1.0 + 0.5 * x - 0.25 * y
                })
            })
            .collect::<Vec<_>>();
        let input = MatrixView::row_major(&data, rows, cols).unwrap();
        let fit = poly(input, Poly2DParams { order: 1 }).unwrap();
        for (actual, expected) in fit.baseline.iter().zip(data) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}
