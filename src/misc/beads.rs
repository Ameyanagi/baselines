//! BEADS baseline estimation.

use crate::fit::{Fit, FitReport};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

const BEADS_SYSTEM_BANDWIDTH: usize = 4;
const BEADS_PENALTY_BANDWIDTH: usize = 2;
const BEADS_DENSE_COMPAT_THRESHOLD: usize = 256;

#[derive(Debug, Clone, Copy)]
struct TridiagonalCoefficients {
    offdiag: f64,
    diag: f64,
}

#[derive(Debug, Clone, Copy)]
struct BeadsSystemCoefficients {
    a: TridiagonalCoefficients,
    b: TridiagonalCoefficients,
}

struct BeadsPenaltyInputs<'a> {
    d1_x: &'a [f64],
    d2_x: &'a [f64],
    diagonal: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
struct BeadsPenaltyParams {
    lam_1: f64,
    lam_2: f64,
    cost_function: BeadsCostFunction,
    eps_1: f64,
}

/// Approximation used for the BEADS sparse absolute-value penalty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeadsCostFunction {
    /// `sqrt(x^2 + eps_1)` approximation.
    L1V1,
    /// `abs(x) - eps_1 * log(abs(x) + eps_1)` approximation.
    L1V2,
}

/// Parameters for BEADS baseline estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadsParams {
    /// Normalized high-pass cutoff frequency in `(0, 0.5)`.
    pub freq_cutoff: f64,
    /// Sparse penalty for signal values.
    pub lam_0: f64,
    /// Sparse penalty for the first derivative of the signal.
    pub lam_1: f64,
    /// Sparse penalty for the second derivative of the signal.
    pub lam_2: f64,
    /// Asymmetric penalty for negative signal values.
    pub asymmetry: f64,
    /// High-pass filter type. Currently only `1` is supported.
    pub filter_type: usize,
    /// Absolute-value penalty approximation.
    pub cost_function: BeadsCostFunction,
    /// Maximum optimization iterations.
    pub max_iter: usize,
    /// Relative cost-change tolerance.
    pub tol: f64,
    /// Threshold between absolute and quadratic asymmetric loss.
    pub eps_0: f64,
    /// Small positive value for sparse derivative penalties.
    pub eps_1: f64,
    /// Whether to subtract and restore an endpoint-matching parabola.
    pub fit_parabola: bool,
    /// Optional half-window for smoothing derivative estimates.
    pub smooth_half_window: Option<usize>,
}

impl Default for BeadsParams {
    fn default() -> Self {
        Self {
            freq_cutoff: 0.005,
            lam_0: 1.0,
            lam_1: 1.0,
            lam_2: 1.0,
            asymmetry: 6.0,
            filter_type: 1,
            cost_function: BeadsCostFunction::L1V2,
            max_iter: 50,
            tol: 1.0e-2,
            eps_0: 1.0e-6,
            eps_1: 1.0e-6,
            fit_parabola: true,
            smooth_half_window: None,
        }
    }
}

impl BeadsParams {
    fn validate(&self) -> Result<()> {
        if !self.freq_cutoff.is_finite() || self.freq_cutoff <= 0.0 || self.freq_cutoff >= 0.5 {
            return Err(BaselineError::InvalidParameter {
                name: "freq_cutoff",
                reason: "must be finite and between 0 and 0.5",
            });
        }
        for (name, value) in [
            ("lam_0", self.lam_0),
            ("lam_1", self.lam_1),
            ("lam_2", self.lam_2),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(BaselineError::InvalidParameter {
                    name,
                    reason: "must be finite and non-negative",
                });
            }
        }
        if !self.asymmetry.is_finite() || self.asymmetry <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "asymmetry",
                reason: "must be finite and positive",
            });
        }
        if self.filter_type != 1 {
            return Err(BaselineError::Unsupported {
                feature: "beads.filter_type",
                reason: "only filter_type = 1 is currently implemented",
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
        if !self.eps_0.is_finite() || self.eps_0 < 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "eps_0",
                reason: "must be finite and non-negative",
            });
        }
        if !self.eps_1.is_finite() || self.eps_1 < 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "eps_1",
                reason: "must be finite and non-negative",
            });
        }
        Ok(())
    }
}

/// Estimates a baseline with BEADS.
///
/// # References
///
/// - X. Ning et al., "Chromatogram baseline estimation and denoising using
///   sparsity (BEADS)", *Chemometrics and Intelligent Laboratory Systems*,
///   2014.
/// - J. A. Navarro-Huerta et al., "Assisted baseline subtraction in complex
///   chromatograms using the BEADS algorithm", *Journal of Chromatography A*,
///   2017.
/// - `pybaselines.Baseline.beads` is used as a behavioral reference.
pub fn beads(y: &[f64], params: BeadsParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "beads",
            len: y.len(),
            min: 3,
        });
    }

    let parabola = if params.fit_parabola {
        endpoint_parabola(y)
    } else {
        vec![0.0; y.len()]
    };
    let centered: Vec<f64> = y
        .iter()
        .zip(&parabola)
        .map(|(observed, fitted)| observed - fitted)
        .collect();
    let eps_0 = params.eps_0.max(f64::MIN_POSITIVE);
    let eps_1 = params.eps_1.max(f64::MIN_POSITIVE);
    let (mut baseline, report) = beads_filter_type_one(
        &centered,
        params,
        eps_0,
        eps_1,
        params.smooth_half_window.unwrap_or(0),
    )?;

    for (value, offset) in baseline.iter_mut().zip(parabola) {
        *value += offset;
    }
    Ok(Fit { baseline, report })
}

fn beads_filter_type_one(
    y: &[f64],
    params: BeadsParams,
    eps_0: f64,
    eps_1: f64,
    smooth_half_window: usize,
) -> Result<(Vec<f64>, FitReport)> {
    let n = y.len();
    let (a_offdiag, a_diag, b_offdiag, b_diag) = filter_coefficients(params.freq_cutoff);
    let coefficients = BeadsSystemCoefficients {
        a: TridiagonalCoefficients {
            offdiag: a_offdiag,
            diag: a_diag,
        },
        b: TridiagonalCoefficients {
            offdiag: b_offdiag,
            diag: b_diag,
        },
    };

    let a_inv_y = solve_tridiagonal(a_offdiag, a_diag, y)?;
    let mut d = apply_btb_tridiagonal(b_offdiag, b_diag, &a_inv_y);
    let asymmetry_shift = params.lam_0 * (1.0 - params.asymmetry) / 2.0;
    let shifted = vec![asymmetry_shift; n];
    for (target, value) in d
        .iter_mut()
        .zip(apply_tridiagonal(a_offdiag, a_diag, &shifted))
    {
        *target -= value;
    }

    let gamma_factor = params.lam_0 * (1.0 + params.asymmetry) / 2.0;
    let mut x = y.to_vec();
    let (mut d1_x, mut d2_x) = abs_diff(&x, smooth_half_window);
    let mut abs_x: Vec<f64> = x.iter().map(|value| value.abs()).collect();
    let mut big_x: Vec<bool> = abs_x.iter().map(|value| *value > eps_0).collect();
    let mut cost_old = 0.0;
    let mut tolerance = f64::INFINITY;
    let mut iterations = 0usize;

    for iter in 0..=params.max_iter {
        let diagonal_penalty: Vec<f64> = abs_x
            .iter()
            .zip(&big_x)
            .map(|(abs_value, is_big)| {
                if *is_big {
                    gamma_factor / abs_value
                } else {
                    gamma_factor / eps_0
                }
            })
            .collect();
        let mut system = beads_system_bands(
            n,
            coefficients,
            BeadsPenaltyInputs {
                d1_x: &d1_x,
                d2_x: &d2_x,
                diagonal: &diagonal_penalty,
            },
            BeadsPenaltyParams {
                lam_1: params.lam_1,
                lam_2: params.lam_2,
                cost_function: params.cost_function,
                eps_1,
            },
        );

        let solved = if n <= BEADS_DENSE_COMPAT_THRESHOLD {
            solve_dense(symmetric_bands_to_dense(&system), d.clone())?
        } else {
            solve_spd_banded(&mut system, &d)?
        };
        x = apply_tridiagonal(a_offdiag, a_diag, &solved);

        let diff: Vec<f64> = y
            .iter()
            .zip(&x)
            .map(|(observed, signal)| observed - signal)
            .collect();
        let h = apply_tridiagonal(
            b_offdiag,
            b_diag,
            &solve_tridiagonal(a_offdiag, a_diag, &diff)?,
        );
        let diffs = abs_diff(&x, smooth_half_window);
        d1_x = diffs.0;
        d2_x = diffs.1;
        let theta = beads_theta(&x, params.asymmetry, eps_0);
        abs_x = x.iter().map(|value| value.abs()).collect();
        big_x = abs_x.iter().map(|value| *value > eps_0).collect();
        let cost = 0.5 * dot(&h, &h)
            + params.lam_0 * theta
            + params.lam_1 * beads_loss_sum(&d1_x, params.cost_function, eps_1)
            + params.lam_2 * beads_loss_sum(&d2_x, params.cost_function, eps_1);
        tolerance = relative_difference_scalar(cost_old, cost);
        iterations = iter + 1;
        if tolerance < params.tol {
            break;
        }
        cost_old = cost;
    }

    let diff: Vec<f64> = y
        .iter()
        .zip(&x)
        .map(|(observed, signal)| observed - signal)
        .collect();
    let high_pass = apply_tridiagonal(
        b_offdiag,
        b_diag,
        &solve_tridiagonal(a_offdiag, a_diag, &diff)?,
    );
    let baseline: Vec<f64> = diff
        .iter()
        .zip(high_pass)
        .map(|(value, high)| value - high)
        .collect();
    Ok((
        baseline,
        FitReport::new(iterations, tolerance < params.tol, tolerance),
    ))
}

fn endpoint_parabola(y: &[f64]) -> Vec<f64> {
    let min_y = y.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let y1 = y[0] - min_y;
    let y2 = y[y.len() - 1] - min_y;
    let c = 0.5 * (y2 + y1);
    let b = c - y1;
    (0..y.len())
        .map(|index| {
            let x = if y.len() == 1 {
                0.0
            } else {
                2.0 * index as f64 / (y.len() - 1) as f64 - 1.0
            };
            min_y + b * x + c * x * x
        })
        .collect()
}

fn filter_coefficients(freq_cutoff: f64) -> (f64, f64, f64, f64) {
    let cos_freq = (2.0 * std::f64::consts::PI * freq_cutoff).cos();
    let t = ((1.0 - cos_freq) / (1.0 + cos_freq).max(f64::MIN_POSITIVE)).max(0.0);
    (-1.0 + t, 2.0 + 2.0 * t, -1.0, 2.0)
}

fn beads_system_bands(
    n: usize,
    coefficients: BeadsSystemCoefficients,
    inputs: BeadsPenaltyInputs<'_>,
    params: BeadsPenaltyParams,
) -> Vec<Vec<f64>> {
    let mut penalty = zero_symmetric_bands(n, BEADS_PENALTY_BANDWIDTH);
    for (index, value) in inputs.diagonal.iter().enumerate() {
        add_symmetric_band_value(&mut penalty, index, index, *value);
    }
    for (index, value) in inputs.d1_x.iter().enumerate() {
        let weight = params.lam_1 * beads_weighting(*value, params.cost_function, params.eps_1);
        add_symmetric_band_value(&mut penalty, index, index, weight);
        add_symmetric_band_value(&mut penalty, index + 1, index + 1, weight);
        add_symmetric_band_value(&mut penalty, index + 1, index, -weight);
    }
    for (index, value) in inputs.d2_x.iter().enumerate() {
        let weight = params.lam_2 * beads_weighting(*value, params.cost_function, params.eps_1);
        let coeffs = [1.0, -2.0, 1.0];
        for row in 0..3 {
            for col in 0..=row {
                add_symmetric_band_value(
                    &mut penalty,
                    index + row,
                    index + col,
                    weight * coeffs[row] * coeffs[col],
                );
            }
        }
    }

    let mut system = zero_symmetric_bands(n, BEADS_SYSTEM_BANDWIDTH);
    add_btb_tridiagonal_bands(&mut system, coefficients.b);
    add_a_penalty_a_bands(&mut system, coefficients.a, &penalty);
    system
}

fn zero_symmetric_bands(n: usize, bandwidth: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; bandwidth + 1]
}

fn add_symmetric_band_value(bands: &mut [Vec<f64>], row: usize, col: usize, value: f64) {
    let (lower, upper) = if row >= col { (row, col) } else { (col, row) };
    let offset = lower - upper;
    debug_assert!(
        offset < bands.len(),
        "band offset {offset} exceeds bandwidth {}",
        bands.len() - 1
    );
    if offset < bands.len() {
        bands[offset][lower] += value;
    }
}

fn set_symmetric_band_value(bands: &mut [Vec<f64>], row: usize, col: usize, value: f64) {
    let (lower, upper) = if row >= col { (row, col) } else { (col, row) };
    let offset = lower - upper;
    debug_assert!(
        offset < bands.len(),
        "band offset {offset} exceeds bandwidth {}",
        bands.len() - 1
    );
    if offset < bands.len() {
        bands[offset][lower] = value;
    }
}

fn symmetric_band_value(bands: &[Vec<f64>], row: usize, col: usize) -> f64 {
    let (lower, upper) = if row >= col { (row, col) } else { (col, row) };
    let offset = lower - upper;
    if offset < bands.len() {
        bands[offset][lower]
    } else {
        0.0
    }
}

fn add_btb_tridiagonal_bands(system: &mut [Vec<f64>], coefficients: TridiagonalCoefficients) {
    let n = system[0].len();
    for row in 0..n {
        for col in row.saturating_sub(2)..=row {
            let mut value = 0.0;
            for mid in row.saturating_sub(1)..=(row + 1).min(n - 1) {
                value += tridiagonal_value(coefficients, row, mid)
                    * tridiagonal_value(coefficients, mid, col);
            }
            add_symmetric_band_value(system, row, col, value);
        }
    }
}

fn add_a_penalty_a_bands(
    system: &mut [Vec<f64>],
    coefficients: TridiagonalCoefficients,
    penalty: &[Vec<f64>],
) {
    let n = system[0].len();
    for row in 0..n {
        for col in row.saturating_sub(BEADS_SYSTEM_BANDWIDTH)..=row {
            let mut value = 0.0;
            for left in row.saturating_sub(1)..=(row + 1).min(n - 1) {
                let a_left = tridiagonal_value(coefficients, row, left);
                if a_left == 0.0 {
                    continue;
                }
                for right in col.saturating_sub(1)..=(col + 1).min(n - 1) {
                    let a_right = tridiagonal_value(coefficients, right, col);
                    if a_right == 0.0 {
                        continue;
                    }
                    value += a_left * symmetric_band_value(penalty, left, right) * a_right;
                }
            }
            add_symmetric_band_value(system, row, col, value);
        }
    }
}

fn tridiagonal_value(coefficients: TridiagonalCoefficients, row: usize, col: usize) -> f64 {
    if row == col {
        coefficients.diag
    } else if row.abs_diff(col) == 1 {
        coefficients.offdiag
    } else {
        0.0
    }
}

fn abs_diff(x: &[f64], smooth_half_window: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d1: Vec<f64> = x.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let mut d2: Vec<f64> = d1.windows(2).map(|pair| pair[1] - pair[0]).collect();
    if smooth_half_window > 0 {
        d2 = moving_average_reflect(&d2, smooth_half_window);
        d1 = moving_average_reflect(&d1, smooth_half_window);
    }
    for value in &mut d1 {
        *value = value.abs();
    }
    for value in &mut d2 {
        *value = value.abs();
    }
    (d1, d2)
}

fn moving_average_reflect(values: &[f64], radius: usize) -> Vec<f64> {
    let width = 2 * radius + 1;
    (0..values.len())
        .map(|index| {
            (index as isize - radius as isize..=index as isize + radius as isize)
                .map(|candidate| values[reflect_index(candidate, values.len())])
                .sum::<f64>()
                / width as f64
        })
        .collect()
}

fn beads_theta(x: &[f64], asymmetry: f64, eps_0: f64) -> f64 {
    x.iter()
        .map(|value| {
            if *value > eps_0 {
                *value
            } else if *value < -eps_0 {
                -asymmetry * value
            } else {
                ((1.0 + asymmetry) / (4.0 * eps_0)) * value * value
                    + ((1.0 - asymmetry) / 2.0) * value
                    + eps_0 * (1.0 + asymmetry) / 4.0
            }
        })
        .sum()
}

fn beads_loss_sum(x: &[f64], cost_function: BeadsCostFunction, eps_1: f64) -> f64 {
    x.iter()
        .map(|value| match cost_function {
            BeadsCostFunction::L1V1 => (value * value + eps_1).sqrt(),
            BeadsCostFunction::L1V2 => value - eps_1 * (value + eps_1).ln(),
        })
        .sum()
}

fn beads_weighting(x: f64, cost_function: BeadsCostFunction, eps_1: f64) -> f64 {
    match cost_function {
        BeadsCostFunction::L1V1 => 1.0 / (x * x + eps_1).sqrt(),
        BeadsCostFunction::L1V2 => 1.0 / (x + eps_1),
    }
}

fn apply_tridiagonal(offdiag: f64, diag: f64, x: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; x.len()];
    for (index, target) in output.iter_mut().enumerate() {
        *target = diag * x[index];
        if index > 0 {
            *target += offdiag * x[index - 1];
        }
        if index + 1 < x.len() {
            *target += offdiag * x[index + 1];
        }
    }
    output
}

fn apply_btb_tridiagonal(offdiag: f64, diag: f64, x: &[f64]) -> Vec<f64> {
    apply_tridiagonal(offdiag, diag, &apply_tridiagonal(offdiag, diag, x))
}

fn solve_tridiagonal(offdiag: f64, diag: f64, rhs: &[f64]) -> Result<Vec<f64>> {
    let n = rhs.len();
    let mut c_prime = vec![0.0; n.saturating_sub(1)];
    let mut d_prime = vec![0.0; n];
    let mut denominator = diag;
    if denominator.abs() <= f64::EPSILON {
        return Err(BaselineError::LinearSolve {
            reason: "singular tridiagonal system",
        });
    }
    if n > 1 {
        c_prime[0] = offdiag / denominator;
    }
    d_prime[0] = rhs[0] / denominator;
    for index in 1..n {
        denominator = diag - offdiag * c_prime[index - 1];
        if denominator.abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "singular tridiagonal system",
            });
        }
        if index + 1 < n {
            c_prime[index] = offdiag / denominator;
        }
        d_prime[index] = (rhs[index] - offdiag * d_prime[index - 1]) / denominator;
    }

    let mut output = vec![0.0; n];
    output[n - 1] = d_prime[n - 1];
    for index in (0..n - 1).rev() {
        output[index] = d_prime[index] - c_prime[index] * output[index + 1];
    }
    Ok(output)
}

fn solve_spd_banded(bands: &mut [Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = rhs.len();
    let bandwidth = bands.len() - 1;
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        for col in start..row {
            let mut value = symmetric_band_value(bands, row, col);
            let sum_start = start.max(col.saturating_sub(bandwidth));
            for mid in sum_start..col {
                value -=
                    symmetric_band_value(bands, row, mid) * symmetric_band_value(bands, col, mid);
            }
            let col_diag = symmetric_band_value(bands, col, col);
            if col_diag.abs() <= f64::EPSILON {
                return Err(BaselineError::LinearSolve {
                    reason: "singular banded Cholesky factor",
                });
            }
            set_symmetric_band_value(bands, row, col, value / col_diag);
        }

        let mut diag = symmetric_band_value(bands, row, row);
        for col in start..row {
            let value = symmetric_band_value(bands, row, col);
            diag -= value * value;
        }
        if diag <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "matrix was not positive definite",
            });
        }
        set_symmetric_band_value(bands, row, row, diag.sqrt());
    }

    let mut intermediate = vec![0.0; n];
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        let tail = (start..row)
            .map(|col| symmetric_band_value(bands, row, col) * intermediate[col])
            .sum::<f64>();
        intermediate[row] = (rhs[row] - tail) / symmetric_band_value(bands, row, row);
    }

    let mut output = vec![0.0; n];
    for row in (0..n).rev() {
        let end = (row + bandwidth).min(n - 1);
        let tail = (row + 1..=end)
            .map(|lower| symmetric_band_value(bands, lower, row) * output[lower])
            .sum::<f64>();
        output[row] = (intermediate[row] - tail) / symmetric_band_value(bands, row, row);
    }
    Ok(output)
}

fn symmetric_bands_to_dense(bands: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = bands[0].len();
    let mut matrix = vec![vec![0.0; n]; n];
    let mut entries = Vec::with_capacity(n * bands.len());
    for row in 0..n {
        for col in row.saturating_sub(bands.len() - 1)..=row {
            entries.push((row, col, symmetric_band_value(bands, row, col)));
        }
    }
    for (row, col, value) in entries {
        matrix[row][col] = value;
        matrix[col][row] = value;
    }
    matrix
}

fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let max_row = (pivot..n)
            .max_by(|left, right| {
                matrix[*left][pivot]
                    .abs()
                    .total_cmp(&matrix[*right][pivot].abs())
            })
            .expect("pivot range is non-empty");
        if matrix[max_row][pivot].abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "singular dense system",
            });
        }
        matrix.swap(pivot, max_row);
        rhs.swap(pivot, max_row);
        let pivot_row = matrix[pivot].clone();
        for row in pivot + 1..n {
            let factor = matrix[row][pivot] / pivot_row[pivot];
            matrix[row][pivot] = 0.0;
            for (col, value) in matrix[row].iter_mut().enumerate().skip(pivot + 1) {
                *value -= factor * pivot_row[col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut output = vec![0.0; n];
    for row in (0..n).rev() {
        let tail = (row + 1..n)
            .map(|col| matrix[row][col] * output[col])
            .sum::<f64>();
        output[row] = (rhs[row] - tail) / matrix[row][row];
    }
    Ok(output)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn relative_difference_scalar(old: f64, new: f64) -> f64 {
    (new - old).abs() / old.abs().max(f64::MIN_POSITIVE)
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
