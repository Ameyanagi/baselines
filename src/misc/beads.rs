//! BEADS baseline estimation.

use crate::fit::{Fit, FitReport};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

const BEADS_SYSTEM_BANDWIDTH: usize = 4;
const BEADS_PENALTY_BANDWIDTH: usize = 2;
// Keep the dense path for small fixture-compatible cases; larger inputs use the
// banded solver to avoid cubic work and repeated dense matrix allocation.
const BEADS_DENSE_COMPAT_THRESHOLD: usize = 192;

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
    let mut next_x = vec![0.0; n];
    let mut diff = vec![0.0; n];
    let mut h = vec![0.0; n];
    let (mut d1_x, mut d2_x) = abs_diff(&x, smooth_half_window);
    let mut abs_x = vec![0.0; n];
    fill_abs(&x, &mut abs_x);
    let mut diagonal_penalty = vec![0.0; n];
    let mut system = zero_symmetric_bands(n, BEADS_SYSTEM_BANDWIDTH);
    let mut cost_old = 0.0;
    let mut tolerance = f64::INFINITY;
    let mut iterations = 0usize;

    for iter in 0..=params.max_iter {
        for (target, abs_value) in diagonal_penalty.iter_mut().zip(&abs_x) {
            *target = if *abs_value > eps_0 {
                gamma_factor / abs_value
            } else {
                gamma_factor / eps_0
            };
        }
        beads_system_bands_into(
            &mut system,
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
        apply_tridiagonal_into(a_offdiag, a_diag, &solved, &mut next_x);
        std::mem::swap(&mut x, &mut next_x);

        fill_difference(y, &x, &mut diff);
        let a_inv_diff = solve_tridiagonal(a_offdiag, a_diag, &diff)?;
        apply_tridiagonal_into(b_offdiag, b_diag, &a_inv_diff, &mut h);
        let diffs = abs_diff(&x, smooth_half_window);
        d1_x = diffs.0;
        d2_x = diffs.1;
        let theta = beads_theta(&x, params.asymmetry, eps_0);
        fill_abs(&x, &mut abs_x);
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

    fill_difference(y, &x, &mut diff);
    let a_inv_diff = solve_tridiagonal(a_offdiag, a_diag, &diff)?;
    apply_tridiagonal_into(b_offdiag, b_diag, &a_inv_diff, &mut h);
    let baseline: Vec<f64> = diff
        .iter()
        .zip(&h)
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

#[cfg(test)]
fn beads_system_bands(
    n: usize,
    coefficients: BeadsSystemCoefficients,
    inputs: BeadsPenaltyInputs<'_>,
    params: BeadsPenaltyParams,
) -> Vec<Vec<f64>> {
    let mut system = zero_symmetric_bands(n, BEADS_SYSTEM_BANDWIDTH);
    beads_system_bands_into(&mut system, n, coefficients, inputs, params);
    system
}

fn beads_system_bands_into(
    system: &mut [Vec<f64>],
    n: usize,
    coefficients: BeadsSystemCoefficients,
    inputs: BeadsPenaltyInputs<'_>,
    params: BeadsPenaltyParams,
) {
    reset_symmetric_bands(system);
    let mut penalty = zero_symmetric_bands(n, BEADS_PENALTY_BANDWIDTH);
    for (index, value) in inputs.diagonal.iter().enumerate() {
        penalty[0][index] += *value;
    }
    for (index, value) in inputs.d1_x.iter().enumerate() {
        let weight = params.lam_1 * beads_weighting(*value, params.cost_function, params.eps_1);
        penalty[0][index] += weight;
        penalty[0][index + 1] += weight;
        penalty[1][index + 1] -= weight;
    }
    for (index, value) in inputs.d2_x.iter().enumerate() {
        let weight = params.lam_2 * beads_weighting(*value, params.cost_function, params.eps_1);
        penalty[0][index] += weight;
        penalty[0][index + 1] += 4.0 * weight;
        penalty[0][index + 2] += weight;
        penalty[1][index + 1] -= 2.0 * weight;
        penalty[1][index + 2] -= 2.0 * weight;
        penalty[2][index + 2] += weight;
    }

    add_btb_tridiagonal_bands(system, coefficients.b);
    add_a_penalty_a_bands(system, coefficients.a, &penalty);
}

fn zero_symmetric_bands(n: usize, bandwidth: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; bandwidth + 1]
}

fn reset_symmetric_bands(bands: &mut [Vec<f64>]) {
    for band in bands {
        band.fill(0.0);
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
    let offdiag_sq = coefficients.offdiag * coefficients.offdiag;
    let diagonal_sq = coefficients.diag * coefficients.diag;
    let first_offdiag = 2.0 * coefficients.offdiag * coefficients.diag;
    let (diag_bands, offdiag_bands) = system.split_at_mut(1);
    let diagonal = &mut diag_bands[0];
    let (first_bands, second_bands) = offdiag_bands.split_at_mut(1);
    let first = &mut first_bands[0];
    let second = &mut second_bands[0];
    for (row, diagonal_value) in diagonal.iter_mut().enumerate() {
        *diagonal_value += diagonal_sq;
        if row > 0 {
            *diagonal_value += offdiag_sq;
            first[row] += first_offdiag;
        }
        if row + 1 < n {
            *diagonal_value += offdiag_sq;
        }
        if row >= 2 {
            second[row] += offdiag_sq;
        }
    }
}

fn add_a_penalty_a_bands(
    system: &mut [Vec<f64>],
    coefficients: TridiagonalCoefficients,
    penalty: &[Vec<f64>],
) {
    let n = system[0].len();
    let a_offdiag = coefficients.offdiag;
    let a_diag = coefficients.diag;
    for row in 0..n {
        for col in row.saturating_sub(BEADS_SYSTEM_BANDWIDTH)..=row {
            let mut value = 0.0;
            let left_start = row.saturating_sub(1);
            let left_end = (row + 1).min(n - 1);
            let right_start = col.saturating_sub(1);
            let right_end = (col + 1).min(n - 1);
            for left in left_start..=left_end {
                let a_left = if left == row { a_diag } else { a_offdiag };
                for right in right_start..=right_end {
                    let offset = left.abs_diff(right);
                    if offset <= BEADS_PENALTY_BANDWIDTH {
                        let lower = left.max(right);
                        let a_right = if right == col { a_diag } else { a_offdiag };
                        value += a_left * penalty[offset][lower] * a_right;
                    }
                }
            }
            system[row - col][row] += value;
        }
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
    apply_tridiagonal_into(offdiag, diag, x, &mut output);
    output
}

fn apply_tridiagonal_into(offdiag: f64, diag: f64, x: &[f64], output: &mut [f64]) {
    debug_assert_eq!(x.len(), output.len());
    for (index, target) in output.iter_mut().enumerate() {
        *target = diag * x[index];
        if index > 0 {
            *target += offdiag * x[index - 1];
        }
        if index + 1 < x.len() {
            *target += offdiag * x[index + 1];
        }
    }
}

fn apply_btb_tridiagonal(offdiag: f64, diag: f64, x: &[f64]) -> Vec<f64> {
    apply_tridiagonal(offdiag, diag, &apply_tridiagonal(offdiag, diag, x))
}

fn fill_abs(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    for (target, value) in output.iter_mut().zip(input) {
        *target = value.abs();
    }
}

fn fill_difference(observed: &[f64], signal: &[f64], output: &mut [f64]) {
    debug_assert_eq!(observed.len(), signal.len());
    debug_assert_eq!(observed.len(), output.len());
    for ((target, observed), signal) in output.iter_mut().zip(observed).zip(signal) {
        *target = observed - signal;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beads_system_bands_matches_dense_reference() {
        let n = 7;
        let coefficients = BeadsSystemCoefficients {
            a: TridiagonalCoefficients {
                offdiag: -0.21,
                diag: 1.64,
            },
            b: TridiagonalCoefficients {
                offdiag: -1.0,
                diag: 2.0,
            },
        };
        let diagonal = [1.1, 0.7, 1.8, 0.9, 1.3, 1.6, 0.8];
        let d1_x = [0.4, 0.9, 1.2, 0.6, 1.5, 0.8];
        let d2_x = [0.7, 1.1, 0.5, 1.6, 0.9];
        let params = BeadsPenaltyParams {
            lam_1: 0.8,
            lam_2: 1.4,
            cost_function: BeadsCostFunction::L1V2,
            eps_1: 1.0e-3,
        };

        let bands = beads_system_bands(
            n,
            coefficients,
            BeadsPenaltyInputs {
                d1_x: &d1_x,
                d2_x: &d2_x,
                diagonal: &diagonal,
            },
            params,
        );
        let expected = dense_beads_system(n, coefficients, &diagonal, &d1_x, &d2_x, params);

        for row in 0..n {
            for col in row.saturating_sub(BEADS_SYSTEM_BANDWIDTH)..=row {
                let actual = bands[row - col][row];
                let expected = expected[row][col];
                assert!(
                    (actual - expected).abs() < 1.0e-9,
                    "band ({row}, {col}) mismatch: actual={actual}, expected={expected}"
                );
            }
        }
    }

    fn dense_beads_system(
        n: usize,
        coefficients: BeadsSystemCoefficients,
        diagonal: &[f64],
        d1_x: &[f64],
        d2_x: &[f64],
        params: BeadsPenaltyParams,
    ) -> Vec<Vec<f64>> {
        let mut penalty = vec![vec![0.0; n]; n];
        for (index, &value) in diagonal.iter().enumerate() {
            penalty[index][index] += value;
        }
        for (index, &value) in d1_x.iter().enumerate() {
            let weight = params.lam_1 * beads_weighting(value, params.cost_function, params.eps_1);
            penalty[index][index] += weight;
            penalty[index + 1][index + 1] += weight;
            penalty[index + 1][index] -= weight;
            penalty[index][index + 1] -= weight;
        }
        for (index, &value) in d2_x.iter().enumerate() {
            let weight = params.lam_2 * beads_weighting(value, params.cost_function, params.eps_1);
            let coeffs = [1.0, -2.0, 1.0];
            for row in 0..3 {
                for col in 0..3 {
                    penalty[index + row][index + col] += weight * coeffs[row] * coeffs[col];
                }
            }
        }

        let mut expected = vec![vec![0.0; n]; n];
        for (row, expected_row) in expected.iter_mut().enumerate() {
            for (col, expected_cell) in expected_row.iter_mut().enumerate() {
                for mid in 0..n {
                    *expected_cell += tridiagonal_dense(coefficients.b, mid, row)
                        * tridiagonal_dense(coefficients.b, mid, col);
                }
                for (left, penalty_row) in penalty.iter().enumerate() {
                    for (right, penalty_value) in penalty_row.iter().enumerate() {
                        *expected_cell += tridiagonal_dense(coefficients.a, row, left)
                            * penalty_value
                            * tridiagonal_dense(coefficients.a, right, col);
                    }
                }
            }
        }
        expected
    }

    fn tridiagonal_dense(coefficients: TridiagonalCoefficients, row: usize, col: usize) -> f64 {
        if row == col {
            coefficients.diag
        } else if row.abs_diff(col) == 1 {
            coefficients.offdiag
        } else {
            0.0
        }
    }
}
