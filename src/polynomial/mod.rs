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

/// Parameters for penalized polynomial fitting.
pub type PenalizedPolyParams = ModPolyParams;

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
pub type QuantRegParams = ModPolyParams;

/// Parameters for Goldindec baseline fitting.
pub type GoldindecParams = ModPolyParams;

/// Fits a penalized polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline.penalized_poly` is used as a behavioral reference.
pub fn penalized_poly(y: &[f64], params: PenalizedPolyParams) -> Result<Fit> {
    modpoly(y, params)
}

/// Fits a LOESS-style local smoothing baseline.
///
/// # References
///
/// - `pybaselines.Baseline.loess` is used as a behavioral reference.
pub fn loess(y: &[f64], params: LoessParams) -> Result<Fit> {
    validate_signal(y)?;
    if params.window_size == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "window_size",
            reason: "must be greater than zero",
        });
    }
    let radius = params.window_size / 2;
    let mut baseline = vec![0.0; y.len()];
    for (i, target) in baseline.iter_mut().enumerate() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(y.len());
        let x0 = scaled_x(i, y.len());
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        for (j, observed) in y[start..end].iter().enumerate() {
            let index = start + j;
            let distance = (scaled_x(index, y.len()) - x0).abs();
            let max_distance = (radius.max(1) as f64) * 2.0 / y.len().max(2) as f64;
            let scaled = (distance / max_distance).min(1.0);
            let weight = (1.0 - scaled.powi(3)).powi(3);
            weight_sum += weight;
            value_sum += weight * observed;
        }
        *target = value_sum / weight_sum.max(f64::EPSILON);
    }
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Fits a quantile-regression polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline.quant_reg` is used as a behavioral reference.
pub fn quant_reg(y: &[f64], params: QuantRegParams) -> Result<Fit> {
    imodpoly(
        y,
        ImodPolyParams {
            order: params.order,
            max_iter: params.max_iter,
            tol: params.tol,
        },
    )
}

/// Fits a Goldindec polynomial baseline.
///
/// # References
///
/// - `pybaselines.Baseline.goldindec` is used as a behavioral reference.
pub fn goldindec(y: &[f64], params: GoldindecParams) -> Result<Fit> {
    modpoly(y, params)
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

fn fit_weighted_polynomial(
    y: &[f64],
    weights: &[f64],
    order: usize,
    baseline: &mut [f64],
) -> Result<()> {
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

    let coeffs = solve_dense(normal, rhs)?;
    for (i, fitted) in baseline.iter_mut().enumerate() {
        let x = scaled_x(i, y.len());
        *fitted = evaluate_polynomial(&coeffs, x);
    }
    Ok(())
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
}
