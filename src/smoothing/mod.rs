//! Smoothing baseline algorithms.
//!
//! These algorithms are implemented with conservative CPU `f64` routines and
//! are intended to converge toward `pybaselines.Baseline` behavior through
//! golden fixture tests.

use crate::fit::{Fit, FitReport};
use crate::linalg::dense::solve_dense;
use crate::morphology::{SnipParams, snip as morphology_snip, snip_into as morphology_snip_into};
use crate::workspace::{validate_output, validate_signal};
use crate::{BaselineError, Result};

/// Parameters for moving-window smoothing methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmoothingParams {
    /// Moving-window size.
    pub window_size: usize,
    /// Maximum number of iterative smoothing passes.
    pub max_iter: usize,
}

impl Default for SmoothingParams {
    fn default() -> Self {
        Self {
            window_size: 31,
            max_iter: 20,
        }
    }
}

impl SmoothingParams {
    fn validate(&self) -> Result<()> {
        if self.window_size == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "window_size",
                reason: "must be greater than zero",
            });
        }
        if self.max_iter == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "max_iter",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }
}

/// Estimates a baseline with a moving median filter.
///
/// # References
///
/// - `pybaselines.Baseline.noise_median` is used as a behavioral reference.
pub fn noise_median(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = noise_median_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a moving-median baseline into an existing buffer.
pub fn noise_median_into(
    y: &[f64],
    params: SmoothingParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_smoothing_input(y, params, baseline)?;
    let radius = params.window_size / 2;
    let padded = pad_extrapolated(y, radius);
    let mut median = vec![0.0; padded.len()];
    moving_median_nearest(&padded, radius, &mut median);
    let kernel = gaussian_kernel(params.window_size, params.window_size as f64 / 6.0);
    let smoothed = convolve_reflect(&median, &kernel);
    baseline.copy_from_slice(&smoothed[radius..radius + y.len()]);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a baseline with the SNIP algorithm.
///
/// # References
///
/// - C. G. Ryan et al., 1988.
/// - `pybaselines.Baseline.snip` is used as a behavioral reference.
pub fn snip(y: &[f64], params: SnipParams) -> Result<Fit> {
    morphology_snip(y, params)
}

/// Estimates a SNIP baseline into an existing buffer.
pub fn snip_into(y: &[f64], params: SnipParams, baseline: &mut [f64]) -> Result<FitReport> {
    morphology_snip_into(y, params, baseline)
}

/// Estimates a baseline with a simple windowed moving-average iteration.
///
/// # References
///
/// - H. Schulze et al., "A Small-Window Moving Average-Based Fully Automated
///   Baseline Estimation Method for Raman Spectra", *Applied Spectroscopy*,
///   2012.
/// - `pybaselines.Baseline.swima` is used as a behavioral reference.
pub fn swima(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = swima_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a SWiMA baseline into an existing output buffer.
pub fn swima_into(y: &[f64], params: SmoothingParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_smoothing_input(y, params, baseline)?;
    let max_half_window = params.window_size / 2;
    if max_half_window == 0 {
        baseline.copy_from_slice(y);
        return Ok(FitReport::new(0, true, 0.0));
    }

    let mut padded = pad_extrapolated(y, max_half_window);
    let smooth_half_window = (y.len() / 50).max(1);
    padded = uniform_filter_reflect(&padded, smooth_half_window);

    let first = swima_loop(&padded, max_half_window, max_half_window)?;
    let mut fitted = first.baseline;
    let mut iterations = first.half_window;
    let mut converged = first.converged == Some(true);
    if !converged {
        let residual: Vec<f64> = padded
            .iter()
            .zip(&fitted)
            .map(|(observed, current)| observed - current)
            .collect();
        let height = residual.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let gaussian = gaussian_background(padded.len(), height);
        let second_input: Vec<f64> = residual
            .iter()
            .zip(&gaussian)
            .map(|(value, background)| value + background)
            .collect();
        let second = swima_loop(&second_input, max_half_window, 3.min(max_half_window))?;
        for ((target, second_value), background) in
            fitted.iter_mut().zip(second.baseline).zip(gaussian)
        {
            *target += second_value - background;
        }
        iterations += second.half_window;
        converged = second.converged == Some(true);
    }

    baseline.copy_from_slice(&fitted[max_half_window..max_half_window + y.len()]);
    Ok(FitReport::new(iterations, converged, 0.0))
}

/// Estimates a baseline with iterative polynomial-style averaging.
///
/// # References
///
/// - `pybaselines.Baseline.ipsa` is used as a behavioral reference.
pub fn ipsa(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let radius = params.window_size / 2;
    let window_size = 2 * radius + 1;
    let pad_len = window_size;
    let original = pad_extrapolated(y, pad_len);
    let mut working = original.clone();
    let mut baseline = vec![0.0; original.len()];
    let mut previous = original[pad_len..pad_len + y.len()].to_vec();
    let kernel = savitzky_golay_kernel(window_size, 2)?;
    let mut tolerance = f64::INFINITY;

    for iter in 0..=params.max_iter {
        baseline = convolve_edge(&working, &kernel);
        tolerance = relative_change(&previous, &baseline[pad_len..pad_len + y.len()]);
        if tolerance < 1.0e-3 {
            return Ok(Fit {
                baseline: baseline[pad_len..pad_len + y.len()].to_vec(),
                report: FitReport::new(iter + 1, true, tolerance),
            });
        }
        for ((target, observed), smooth) in working.iter_mut().zip(&original).zip(&baseline) {
            *target = observed.min(*smooth);
        }
        previous.copy_from_slice(&baseline[pad_len..pad_len + y.len()]);
    }

    Ok(Fit {
        baseline: baseline[pad_len..pad_len + y.len()].to_vec(),
        report: FitReport::new(params.max_iter + 1, false, tolerance),
    })
}

/// Estimates a baseline with range-independent averaging.
///
/// # References
///
/// - `pybaselines.Baseline.ria` is used as a behavioral reference.
pub fn ria(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    iterative_smoother(y, params)
}

/// Estimates a baseline by iteratively filling peaks from neighboring values.
///
/// # References
///
/// - `pybaselines.Baseline.peak_filling` is used as a behavioral reference.
pub fn peak_filling(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let sections = (y.len() / 10).max(1).min(y.len());
    let (x_fit, mut baseline_fit) = section_minima(y, sections);
    let mut half_window = (params.window_size / 2).min(sections.saturating_sub(1) / 2);
    if half_window == 0 {
        half_window = 1.min(sections.saturating_sub(1));
    }
    for half_window in logarithmic_half_windows(half_window, params.max_iter) {
        directional_min_moving_average(&mut baseline_fit, sections, half_window);
        baseline_fit.reverse();
        directional_min_moving_average(&mut baseline_fit, sections, half_window);
        baseline_fit.reverse();
    }
    let baseline = interpolate_default_domain(y.len(), &x_fit, &baseline_fit);
    Ok(Fit {
        baseline,
        report: FitReport::new(params.max_iter, true, 0.0),
    })
}

struct SwimaLoopResult {
    baseline: Vec<f64>,
    converged: Option<bool>,
    half_window: usize,
}

fn swima_loop(
    y: &[f64],
    max_half_window: usize,
    min_half_window: usize,
) -> Result<SwimaLoopResult> {
    let data_start = max_half_window;
    let data_end = y.len() - max_half_window;
    let actual_y = &y[data_start..data_end];
    let mut baseline = y.to_vec();
    let mut area_current = -1.0;
    let mut area_old = -1.0;
    let mut converged = None;
    let mut reached_half_window = 0;
    let min_half_window_check = min_half_window as isize - 2;

    for half_window in 1..=max_half_window {
        let smoothed = uniform_filter_reflect(&baseline, half_window);
        let baseline_new: Vec<f64> = baseline
            .iter()
            .zip(smoothed)
            .map(|(current, smooth)| current.min(smooth))
            .collect();

        if (half_window as isize) > min_half_window_check {
            let area_new = trapezoid(
                &baseline[data_start..data_end]
                    .iter()
                    .zip(&baseline_new[data_start..data_end])
                    .map(|(current, next)| current - next)
                    .collect::<Vec<_>>(),
            );
            if area_new > area_current && area_current < area_old {
                converged = Some(true);
                reached_half_window = half_window - 1;
                break;
            }

            if half_window > min_half_window {
                let residual: Vec<f64> = actual_y
                    .iter()
                    .zip(&baseline_new[data_start..data_end])
                    .map(|(observed, next)| observed - next)
                    .collect();
                let diff = gradient(&residual);
                let polynomial_diff = fit_polynomial_values(&diff, 3)?;
                if trapezoid_abs(&polynomial_diff) > 0.15 * trapezoid_abs(&diff) {
                    converged = Some(false);
                    reached_half_window = half_window;
                    break;
                }
            }

            area_old = area_current;
            area_current = area_new;
        }

        baseline = baseline_new;
        reached_half_window = half_window;
    }

    Ok(SwimaLoopResult {
        baseline,
        converged,
        half_window: reached_half_window,
    })
}

fn iterative_smoother(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let mut baseline = y.to_vec();
    let mut smoothed = vec![0.0; y.len()];
    for _ in 0..params.max_iter {
        moving_average(&baseline, params.window_size / 2, &mut smoothed);
        for (target, smooth) in baseline.iter_mut().zip(&smoothed) {
            *target = *smooth;
        }
    }
    Ok(Fit {
        baseline,
        report: FitReport::new(params.max_iter, true, 0.0),
    })
}

fn validate_smoothing_input(y: &[f64], params: SmoothingParams, baseline: &[f64]) -> Result<()> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()
}

fn moving_average(y: &[f64], radius: usize, output: &mut [f64]) {
    for (i, target) in output.iter_mut().enumerate() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(y.len());
        *target = y[start..end].iter().sum::<f64>() / (end - start) as f64;
    }
}

fn uniform_filter_reflect(y: &[f64], radius: usize) -> Vec<f64> {
    if radius == 0 {
        return y.to_vec();
    }

    let window_size = 2 * radius + 1;
    let mut output = vec![0.0; y.len()];
    for (index, target) in output.iter_mut().enumerate() {
        let sum = (0..window_size)
            .map(|offset| {
                let source =
                    reflect_edge_index(index as isize + offset as isize - radius as isize, y.len());
                y[source]
            })
            .sum::<f64>();
        *target = sum / window_size as f64;
    }
    output
}

fn moving_median_nearest(y: &[f64], radius: usize, output: &mut [f64]) {
    let mut window = Vec::with_capacity(2 * radius + 1);
    for (i, target) in output.iter_mut().enumerate() {
        window.clear();
        for offset in 0..(2 * radius + 1) {
            let index = (i + offset).saturating_sub(radius).min(y.len() - 1);
            window.push(y[index]);
        }
        window.sort_by(f64::total_cmp);
        *target = window[window.len() / 2];
    }
}

fn gaussian_kernel(window_size: usize, sigma: f64) -> Vec<f64> {
    let window_size = window_size.max(1);
    let center = (window_size - 1) as f64 / 2.0;
    let mut kernel = (0..window_size)
        .map(|index| {
            let x = index as f64 - center;
            (-0.5 * (x / sigma.max(f64::EPSILON)).powi(2)).exp()
        })
        .collect::<Vec<_>>();
    let sum = kernel.iter().sum::<f64>().max(f64::EPSILON);
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn convolve_reflect(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let mut output = vec![0.0; data.len()];
    for (index, target) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (kernel_index, weight) in kernel.iter().enumerate() {
            let source = reflect_index(
                index as isize + kernel_index as isize - radius as isize,
                data.len(),
            );
            sum += data[source] * weight;
        }
        *target = sum;
    }
    output
}

fn convolve_edge(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let mut output = vec![0.0; data.len()];
    for (index, target) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (kernel_index, weight) in kernel.iter().enumerate() {
            let shifted = index as isize + kernel_index as isize - radius as isize;
            let source = shifted.clamp(0, data.len() as isize - 1) as usize;
            sum += data[source] * weight;
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
    let mut reflected = index.rem_euclid(period);
    if reflected >= len as isize {
        reflected = period - reflected;
    }
    reflected as usize
}

fn reflect_edge_index(mut index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let len = len as isize;
    while index < 0 || index >= len {
        if index < 0 {
            index = -index - 1;
        } else {
            index = 2 * len - index - 1;
        }
    }
    index as usize
}

fn gaussian_background(len: usize, height: f64) -> Vec<f64> {
    let center = len as f64 / 2.0;
    let sigma = (len as f64 / 6.0).max(f64::EPSILON);
    (0..len)
        .map(|index| {
            let centered = index as f64 - center;
            height * (-0.5 * centered * centered / (sigma * sigma)).exp()
        })
        .collect()
}

fn gradient(values: &[f64]) -> Vec<f64> {
    match values.len() {
        0 => Vec::new(),
        1 => vec![0.0],
        len => {
            let mut output = vec![0.0; len];
            output[0] = values[1] - values[0];
            output[len - 1] = values[len - 1] - values[len - 2];
            for index in 1..len - 1 {
                output[index] = 0.5 * (values[index + 1] - values[index - 1]);
            }
            output
        }
    }
}

fn fit_polynomial_values(values: &[f64], order: usize) -> Result<Vec<f64>> {
    if values.len() <= order {
        return Ok(values.to_vec());
    }

    let basis_len = order + 1;
    let mut normal = vec![vec![0.0; basis_len]; basis_len];
    let mut rhs = vec![0.0; basis_len];
    for (index, value) in values.iter().enumerate() {
        let x = scaled_x(index, values.len());
        let powers = powers(x, order);
        for row in 0..basis_len {
            rhs[row] += value * powers[row];
            for col in 0..basis_len {
                normal[row][col] += powers[row] * powers[col];
            }
        }
    }
    let coefficients = solve_dense(normal, rhs)?;
    Ok((0..values.len())
        .map(|index| {
            let x = scaled_x(index, values.len());
            powers(x, order)
                .iter()
                .zip(&coefficients)
                .map(|(basis, coeff)| basis * coeff)
                .sum()
        })
        .collect())
}

fn trapezoid(values: &[f64]) -> f64 {
    values
        .windows(2)
        .map(|pair| 0.5 * (pair[0] + pair[1]))
        .sum()
}

fn trapezoid_abs(values: &[f64]) -> f64 {
    values
        .windows(2)
        .map(|pair| 0.5 * (pair[0].abs() + pair[1].abs()))
        .sum()
}

fn savitzky_golay_kernel(window_size: usize, poly_order: usize) -> Result<Vec<f64>> {
    let window_size = window_size.max(1);
    let poly_order = poly_order.min(window_size - 1);
    let radius = window_size / 2;
    let basis_len = poly_order + 1;
    let mut normal = vec![vec![0.0; basis_len]; basis_len];
    for offset in 0..window_size {
        let x = offset as f64 - radius as f64;
        let powers = powers(x, poly_order);
        for row in 0..basis_len {
            for col in 0..basis_len {
                normal[row][col] += powers[row] * powers[col];
            }
        }
    }
    let mut rhs = vec![0.0; basis_len];
    rhs[0] = 1.0;
    let projection = solve_dense(normal, rhs)?;
    Ok((0..window_size)
        .map(|offset| {
            let x = offset as f64 - radius as f64;
            powers(x, poly_order)
                .iter()
                .zip(&projection)
                .map(|(basis, coeff)| basis * coeff)
                .sum()
        })
        .collect())
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

fn pad_extrapolated(y: &[f64], radius: usize) -> Vec<f64> {
    if radius == 0 {
        return y.to_vec();
    }
    let fit_len = radius.min(y.len()).max(1);
    let mut padded = Vec::with_capacity(y.len() + 2 * radius);
    let (left_slope, left_intercept) =
        edge_line((0..fit_len).map(|index| (index as f64, y[index])));
    for offset in (1..=radius).rev() {
        let x = -(offset as f64);
        padded.push(left_slope * x + left_intercept);
    }
    padded.extend_from_slice(y);
    let start = y.len().saturating_sub(fit_len);
    let (right_slope, right_intercept) =
        edge_line((start..y.len()).map(|index| (index as f64, y[index])));
    for offset in 0..radius {
        let x = y.len() as f64 + offset as f64;
        padded.push(right_slope * x + right_intercept);
    }
    padded
}

fn edge_line(points: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    let values = points.collect::<Vec<_>>();
    let n = values.len() as f64;
    let x_mean = values.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let y_mean = values.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (x, y) in values {
        let centered_x = x - x_mean;
        numerator += centered_x * (y - y_mean);
        denominator += centered_x * centered_x;
    }
    let slope = if denominator <= f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    };
    (slope, y_mean - slope * x_mean)
}

fn section_minima(y: &[f64], sections: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x_fit = Vec::with_capacity(sections + 2);
    let mut y_fit = Vec::with_capacity(sections + 2);
    for section in 0..sections {
        let start = section * y.len() / sections;
        let end = ((section + 1) * y.len() / sections).max(start + 1);
        let values = &y[start..end.min(y.len())];
        y_fit.push(values.iter().copied().fold(f64::INFINITY, f64::min));
        let x_sum = (start..end.min(y.len()))
            .map(|index| scaled_x(index, y.len()))
            .sum::<f64>();
        x_fit.push(x_sum / values.len() as f64);
    }
    if x_fit.first().is_some_and(|value| *value != -1.0) {
        x_fit.insert(0, -1.0);
        y_fit.insert(0, y[0]);
    }
    if x_fit.last().is_some_and(|value| *value != 1.0) {
        x_fit.push(1.0);
        y_fit.push(*y.last().expect("validated signal is non-empty"));
    }
    (x_fit, y_fit)
}

fn logarithmic_half_windows(start: usize, count: usize) -> Vec<usize> {
    if count == 1 {
        return vec![start.max(1)];
    }
    let log_start = (start.max(1) as f64).log10();
    let step = log_start / (count - 1) as f64;
    let mut values = (0..count)
        .map(|index| 10f64.powf(log_start - step * index as f64).ceil() as usize)
        .map(|value| value.max(1))
        .collect::<Vec<_>>();
    values[0] = start.max(1);
    values
}

fn directional_min_moving_average(y: &mut [f64], active_len: usize, half_window: usize) {
    let active_len = active_len.min(y.len());
    if active_len < 3 {
        return;
    }
    let half_window = half_window.min((active_len - 1) / 2);
    if half_window == 0 {
        return;
    }
    for index in 1..active_len - 1 {
        let radius = index.min(active_len - 1 - index).min(half_window);
        let start = index - radius;
        let end = index + radius + 1;
        let mean = y[start..end].iter().sum::<f64>() / (end - start) as f64;
        if mean < y[index] {
            y[index] = mean;
        }
    }
}

fn interpolate_default_domain(len: usize, x_fit: &[f64], y_fit: &[f64]) -> Vec<f64> {
    (0..len)
        .map(|index| interpolate_piecewise_linear(scaled_x(index, len), x_fit, y_fit))
        .collect()
}

fn scaled_x(index: usize, len: usize) -> f64 {
    if len <= 1 {
        0.0
    } else {
        2.0 * index as f64 / (len - 1) as f64 - 1.0
    }
}

fn interpolate_piecewise_linear(x: f64, x_fit: &[f64], y_fit: &[f64]) -> f64 {
    if x <= x_fit[0] {
        return y_fit[0];
    }
    for index in 1..x_fit.len() {
        if x <= x_fit[index] {
            let left_x = x_fit[index - 1];
            let right_x = x_fit[index];
            let fraction = if (right_x - left_x).abs() <= f64::EPSILON {
                0.0
            } else {
                (x - left_x) / (right_x - left_x)
            };
            return y_fit[index - 1] + fraction * (y_fit[index] - y_fit[index - 1]);
        }
    }
    *y_fit.last().expect("fit values are non-empty")
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
    use super::*;

    #[test]
    fn smoothing_methods_return_finite_baselines() {
        let y: Vec<f64> = (0..80)
            .map(|i| {
                let x = i as f64 / 79.0;
                1.0 + 0.1 * x + (-(x - 0.5).powi(2) / 0.003).exp()
            })
            .collect();
        let params = SmoothingParams::default();
        for fit in [
            noise_median(&y, params).unwrap(),
            swima(&y, params).unwrap(),
            ipsa(&y, params).unwrap(),
            ria(&y, params).unwrap(),
            peak_filling(&y, params).unwrap(),
        ] {
            assert!(fit.baseline.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn smoothing_methods_preserve_constant_signals() {
        let y = vec![2.5; 80];
        let params = SmoothingParams {
            window_size: 17,
            max_iter: 20,
        };
        for fit in [
            noise_median(&y, params).unwrap(),
            swima(&y, params).unwrap(),
            ipsa(&y, params).unwrap(),
            ria(&y, params).unwrap(),
            peak_filling(&y, params).unwrap(),
        ] {
            assert!(
                fit.baseline
                    .iter()
                    .all(|value| (*value - 2.5).abs() < 1e-10)
            );
        }
    }

    #[test]
    fn peak_filling_tracks_section_minimum_baseline() {
        let y = vec![
            1.0, 1.0, 1.0, 2.5, 5.0, 2.5, 1.1, 1.2, 1.3, 3.5, 7.0, 3.5, 1.4, 1.5, 1.6, 1.7, 1.8,
            1.9, 2.0, 2.1,
        ];
        let fit = peak_filling(
            &y,
            SmoothingParams {
                window_size: 5,
                max_iter: 8,
            },
        )
        .unwrap();

        assert!(fit.baseline.iter().all(|value| value.is_finite()));
        for (baseline, observed) in fit.baseline.iter().zip(y) {
            assert!(baseline <= &(observed + 1e-10));
        }
    }
}
