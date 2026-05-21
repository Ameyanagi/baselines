//! Dense P-spline helper for one-dimensional baseline methods.

use crate::linalg::dense::solve_dense;
use crate::{BaselineError, Result};

/// Dense penalized B-spline basis and solver.
#[derive(Debug, Clone)]
pub(crate) struct PenalizedSpline {
    basis: Vec<Vec<f64>>,
    basis_midpoints: Vec<f64>,
    first_order_penalty: Vec<Vec<f64>>,
    penalty: Vec<Vec<f64>>,
}

impl PenalizedSpline {
    /// Creates a cubic P-spline basis over `n` equally spaced points in `[-1, 1]`.
    pub(crate) fn new(n: usize, num_knots: usize, degree: usize, diff_order: usize) -> Self {
        let x = scaled_domain(n);
        let knots = spline_knots(&x, num_knots, degree);
        let n_bases = knots.len() - degree - 1;
        let basis_midpoints = basis_midpoints(&knots, degree);
        let basis = x
            .iter()
            .map(|value| basis_row(*value, &knots, degree, n_bases))
            .collect();
        let first_order_penalty = difference_penalty(n_bases, 1);
        let penalty = difference_penalty(n_bases, diff_order);
        Self {
            basis,
            basis_midpoints,
            first_order_penalty,
            penalty,
        }
    }

    /// Fits a weighted penalized spline and returns the evaluated baseline.
    pub(crate) fn solve(&self, y: &[f64], weights: &[f64], lambda: f64) -> Result<Vec<f64>> {
        self.solve_with_options(y, weights, lambda, None, 0.0, 0.0)
    }

    /// Fits a weighted penalized spline with row-scaled smoothness penalties.
    pub(crate) fn solve_with_row_scaled_penalty(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        row_scales: &[f64],
    ) -> Result<Vec<f64>> {
        self.solve_with_options(y, weights, lambda, Some(row_scales), 0.0, 0.0)
    }

    /// Fits a weighted penalized spline with drPLS basis penalties.
    pub(crate) fn solve_with_drpls_penalty(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        eta: f64,
        basis_weights: &[f64],
    ) -> Result<Vec<f64>> {
        let n_bases = self.penalty.len();
        if basis_weights.len() != n_bases {
            return Err(BaselineError::LengthMismatch {
                name: "basis_weights",
                expected: n_bases,
                actual: basis_weights.len(),
            });
        }

        let row_scales: Vec<f64> = basis_weights
            .iter()
            .map(|weight| 1.0 - eta * weight)
            .collect();
        self.solve_with_options(y, weights, lambda, Some(&row_scales), 1.0, 0.0)
    }

    /// Fits a weighted penalized spline with an added data-domain first-difference penalty.
    pub(crate) fn solve_with_first_difference_penalty(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        first_difference_lambda: f64,
    ) -> Result<Vec<f64>> {
        self.solve_with_options(y, weights, lambda, None, 0.0, first_difference_lambda)
    }

    /// Interpolates sample-domain values onto the spline basis midpoints.
    pub(crate) fn interpolate_to_basis(&self, values: &[f64]) -> Vec<f64> {
        match values.len() {
            0 => vec![0.0; self.basis_midpoints.len()],
            1 => vec![values[0]; self.basis_midpoints.len()],
            len => self
                .basis_midpoints
                .iter()
                .map(|point| interpolate_sample(values, *point, len))
                .collect(),
        }
    }

    fn solve_with_options(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        row_scales: Option<&[f64]>,
        basis_first_difference_lambda: f64,
        data_first_difference_lambda: f64,
    ) -> Result<Vec<f64>> {
        let n_bases = self.penalty.len();
        if let Some(scales) = row_scales
            && scales.len() != n_bases
        {
            return Err(BaselineError::LengthMismatch {
                name: "row_scales",
                expected: n_bases,
                actual: scales.len(),
            });
        }

        let mut normal = vec![vec![0.0; n_bases]; n_bases];
        let mut rhs = vec![0.0; n_bases];

        for ((basis_row, observed), weight) in self.basis.iter().zip(y).zip(weights) {
            for row in 0..n_bases {
                rhs[row] += basis_row[row] * weight * observed;
                for col in 0..n_bases {
                    normal[row][col] += basis_row[row] * weight * basis_row[col];
                }
            }
        }

        for (row, (normal_row, penalty_row)) in normal.iter_mut().zip(&self.penalty).enumerate() {
            let scale = row_scales.map_or(1.0, |scales| scales[row]);
            for (normal_value, penalty_value) in normal_row.iter_mut().zip(penalty_row) {
                *normal_value += lambda * scale * penalty_value;
            }
        }

        if basis_first_difference_lambda > 0.0 {
            for (normal_row, penalty_row) in normal.iter_mut().zip(&self.first_order_penalty) {
                for (normal_value, penalty_value) in normal_row.iter_mut().zip(penalty_row) {
                    *normal_value += basis_first_difference_lambda * penalty_value;
                }
            }
        }

        if data_first_difference_lambda > 0.0 {
            for (basis_pair, observed_pair) in self.basis.windows(2).zip(y.windows(2)) {
                let observed_difference = observed_pair[1] - observed_pair[0];
                for row in 0..n_bases {
                    let basis_row_difference = basis_pair[1][row] - basis_pair[0][row];
                    rhs[row] +=
                        data_first_difference_lambda * basis_row_difference * observed_difference;
                    for col in 0..n_bases {
                        normal[row][col] += data_first_difference_lambda
                            * basis_row_difference
                            * (basis_pair[1][col] - basis_pair[0][col]);
                    }
                }
            }
        }

        let coef = solve_dense(normal, rhs)?;
        Ok(self
            .basis
            .iter()
            .map(|basis_row| basis_row.iter().zip(&coef).map(|(b, c)| b * c).sum())
            .collect())
    }
}

fn scaled_domain(n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => (0..n)
            .map(|index| 2.0 * index as f64 / (n - 1) as f64 - 1.0)
            .collect(),
    }
}

fn spline_knots(x: &[f64], num_knots: usize, degree: usize) -> Vec<f64> {
    let num_knots = num_knots.max(2);
    let x_min = *x.first().unwrap_or(&-1.0);
    let x_max = *x.last().unwrap_or(&1.0);
    let dx = (x_max - x_min) / (num_knots - 1) as f64;
    let mut knots = Vec::with_capacity(num_knots + 2 * degree);
    for index in (1..=degree).rev() {
        knots.push(x_min - index as f64 * dx);
    }
    for index in 0..num_knots {
        knots.push(x_min + index as f64 * dx);
    }
    for index in 1..=degree {
        knots.push(x_max + index as f64 * dx);
    }
    knots
}

fn basis_midpoints(knots: &[f64], degree: usize) -> Vec<f64> {
    if degree % 2 == 1 {
        let start = 1 + degree / 2;
        let end = knots.len() - (degree - degree / 2);
        knots[start..end].to_vec()
    } else {
        let midpoints: Vec<f64> = knots
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        let start = degree / 2;
        let end = midpoints.len() - degree / 2;
        midpoints[start..end].to_vec()
    }
}

fn interpolate_sample(values: &[f64], point: f64, len: usize) -> f64 {
    if point <= -1.0 {
        return values[0];
    }
    if point >= 1.0 {
        return values[len - 1];
    }

    let position = 0.5 * (point + 1.0) * (len - 1) as f64;
    let left = position.floor() as usize;
    let right = (left + 1).min(len - 1);
    let fraction = position - left as f64;
    values[left] * (1.0 - fraction) + values[right] * fraction
}

fn basis_row(x: f64, knots: &[f64], degree: usize, n_bases: usize) -> Vec<f64> {
    let mut values = vec![0.0; n_bases];
    for (index, value) in values.iter_mut().enumerate() {
        *value = basis_value(index, degree, x, knots);
    }
    values
}

fn basis_value(index: usize, degree: usize, x: f64, knots: &[f64]) -> f64 {
    if degree == 0 {
        let left = knots[index];
        let right = knots[index + 1];
        if (left <= x && x < right) || (x == *knots.last().unwrap_or(&right) && x == right) {
            1.0
        } else {
            0.0
        }
    } else {
        let left_denominator = knots[index + degree] - knots[index];
        let left = if left_denominator.abs() <= f64::EPSILON {
            0.0
        } else {
            (x - knots[index]) / left_denominator * basis_value(index, degree - 1, x, knots)
        };
        let right_denominator = knots[index + degree + 1] - knots[index + 1];
        let right = if right_denominator.abs() <= f64::EPSILON {
            0.0
        } else {
            (knots[index + degree + 1] - x) / right_denominator
                * basis_value(index + 1, degree - 1, x, knots)
        };
        left + right
    }
}

fn difference_penalty(n_bases: usize, diff_order: usize) -> Vec<Vec<f64>> {
    let rows = n_bases.saturating_sub(diff_order);
    let mut difference = vec![vec![0.0; n_bases]; rows];
    let coefficients = difference_coefficients(diff_order);
    for row in 0..rows {
        for (offset, coefficient) in coefficients.iter().enumerate() {
            difference[row][row + offset] = *coefficient;
        }
    }

    let mut penalty = vec![vec![0.0; n_bases]; n_bases];
    for row in &difference {
        for i in 0..n_bases {
            for j in 0..n_bases {
                penalty[i][j] += row[i] * row[j];
            }
        }
    }
    penalty
}

fn difference_coefficients(order: usize) -> Vec<f64> {
    let mut coefficients = vec![1.0];
    for _ in 0..order {
        let mut next = vec![0.0; coefficients.len() + 1];
        for (index, coefficient) in coefficients.iter().enumerate() {
            next[index] -= coefficient;
            next[index + 1] += coefficient;
        }
        coefficients = next;
    }
    coefficients
}
