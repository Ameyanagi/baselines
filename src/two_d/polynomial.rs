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
use crate::linalg::dense::solve_dense;
use crate::workspace::rms;
use crate::{BaselineError, Result};

/// Parameters for a direct two-dimensional polynomial baseline fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poly2DParams {
    /// Maximum total polynomial degree.
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
    /// Maximum total polynomial degree.
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
    /// Maximum total polynomial degree.
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
    /// Maximum total polynomial degree.
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
    /// Maximum total polynomial degree.
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
    fit_weighted_polynomial(input, &weights, params.order, output.as_mut_slice())?;
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
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(output.as_slice());
        fit_weighted_surface(
            &clipped,
            input.rows(),
            input.cols(),
            &weights,
            params.order,
            output.as_mut_slice(),
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
    let mut tolerance = f64::INFINITY;

    for iter in 0..params.max_iter {
        previous.copy_from_slice(output.as_slice());
        fit_weighted_surface(
            &clipped,
            input.rows(),
            input.cols(),
            &weights,
            params.order,
            output.as_mut_slice(),
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
    let mut tolerance = f64::INFINITY;
    fit_weighted_polynomial(input, &weights, params.order, output.as_mut_slice())?;

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
        fit_weighted_surface(
            &adjusted,
            input.rows(),
            input.cols(),
            &weights,
            params.order,
            output.as_mut_slice(),
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

    fit_weighted_polynomial(input, &weights, params.order, output.as_mut_slice())?;
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

        fit_weighted_polynomial(input, &weights, params.order, output.as_mut_slice())?;
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

fn fit_weighted_polynomial(
    input: MatrixView<'_>,
    weights: &[f64],
    order: usize,
    baseline: &mut [f64],
) -> Result<()> {
    fit_weighted_surface(
        input.as_slice(),
        input.rows(),
        input.cols(),
        weights,
        order,
        baseline,
    )
}

fn fit_weighted_surface(
    data: &[f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
    order: usize,
    baseline: &mut [f64],
) -> Result<()> {
    let terms = basis_terms(order);
    let coeffs = fit_weighted_coefficients(data, rows, cols, weights, &terms)?;
    evaluate_coefficients(rows, cols, &terms, &coeffs, baseline);
    Ok(())
}

fn fit_weighted_coefficients(
    data: &[f64],
    rows: usize,
    cols: usize,
    weights: &[f64],
    terms: &[(usize, usize)],
) -> Result<Vec<f64>> {
    if weights.len() != data.len() {
        return Err(BaselineError::LengthMismatch {
            name: "weights",
            expected: data.len(),
            actual: weights.len(),
        });
    }
    let n_coeffs = terms.len();
    let mut normal = vec![vec![0.0; n_coeffs]; n_coeffs];
    let mut rhs = vec![0.0; n_coeffs];
    let mut basis = vec![0.0; n_coeffs];

    for row in 0..rows {
        let y = scaled_axis(row, rows);
        for col in 0..cols {
            let index = row * cols + col;
            let weight = weights[index].max(0.0);
            if weight <= f64::EPSILON {
                continue;
            }
            let x = scaled_axis(col, cols);
            fill_basis(x, y, terms, &mut basis);
            for basis_row in 0..n_coeffs {
                rhs[basis_row] += weight * data[index] * basis[basis_row];
                for basis_col in 0..n_coeffs {
                    normal[basis_row][basis_col] += weight * basis[basis_row] * basis[basis_col];
                }
            }
        }
    }

    solve_dense(normal, rhs)
}

fn evaluate_coefficients(
    rows: usize,
    cols: usize,
    terms: &[(usize, usize)],
    coeffs: &[f64],
    baseline: &mut [f64],
) {
    let mut basis = vec![0.0; terms.len()];
    for row in 0..rows {
        let y = scaled_axis(row, rows);
        for col in 0..cols {
            let x = scaled_axis(col, cols);
            fill_basis(x, y, terms, &mut basis);
            baseline[row * cols + col] = coeffs.iter().zip(&basis).map(|(c, b)| c * b).sum();
        }
    }
}

fn basis_terms(order: usize) -> Vec<(usize, usize)> {
    let mut terms = Vec::with_capacity((order + 1) * (order + 2) / 2);
    for total_degree in 0..=order {
        for x_degree in (0..=total_degree).rev() {
            terms.push((x_degree, total_degree - x_degree));
        }
    }
    terms
}

fn fill_basis(x: f64, y: f64, terms: &[(usize, usize)], output: &mut [f64]) {
    for ((x_degree, y_degree), target) in terms.iter().zip(output) {
        *target = x.powi(*x_degree as i32) * y.powi(*y_degree as i32);
    }
}

fn scaled_axis(index: usize, len: usize) -> f64 {
    if len <= 1 {
        0.0
    } else {
        -1.0 + 2.0 * index as f64 / (len - 1) as f64
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
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>();
    let denominator = previous.iter().map(|value| value.abs()).sum::<f64>();
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
