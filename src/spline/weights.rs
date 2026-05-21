//! Reweighting rules used by penalized spline baseline algorithms.

use crate::workspace::logistic;

const AIRPLS_MAX_EXPONENT: f64 = 700.0;

pub(super) fn airpls_weights(
    y: &[f64],
    baseline: &[f64],
    iteration: usize,
) -> (Vec<f64>, f64, bool) {
    let residuals = residuals(y, baseline);
    let negative: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|residual| *residual < 0.0)
        .collect();
    if negative.len() < 2 {
        return (vec![0.0; y.len()], 0.0, true);
    }

    let residual_l1_norm = negative.iter().sum::<f64>().abs();
    let scale = iteration.min(50) as f64 / residual_l1_norm;
    let weights = residuals
        .into_iter()
        .map(|residual| {
            if residual < 0.0 {
                (scale * residual.abs()).min(AIRPLS_MAX_EXPONENT).exp()
            } else {
                0.0
            }
        })
        .collect();

    (weights, residual_l1_norm, false)
}

pub(super) fn arpls_weights(y: &[f64], baseline: &[f64]) -> Option<Vec<f64>> {
    let residuals = residuals(y, baseline);
    let (mean, std) = negative_residual_stats(&residuals)?;
    let weights = residuals
        .into_iter()
        .map(|residual| logistic(-(2.0 / std) * (residual - (2.0 * std - mean))))
        .collect();

    Some(weights)
}

pub(super) fn aspls_weights(
    y: &[f64],
    baseline: &[f64],
    asymmetric_coef: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let residuals = residuals(y, baseline);
    let (_mean, std) = negative_residual_stats(&residuals)?;
    let scale = asymmetric_coef / std;
    let weights = residuals
        .iter()
        .map(|residual| logistic(-scale * (*residual - std)))
        .collect();

    Some((weights, residuals))
}

pub(super) fn brpls_weights(y: &[f64], baseline: &[f64], beta: f64) -> Option<Vec<f64>> {
    let residuals = residuals(y, baseline);
    let positive: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|residual| *residual > 0.0)
        .collect();
    let negative: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|residual| *residual < 0.0)
        .collect();
    if positive.len() < 2 || negative.len() < 2 {
        return None;
    }

    let mean = positive.iter().sum::<f64>() / positive.len() as f64;
    let sigma = (negative
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
        / negative.len() as f64)
        .sqrt()
        .max(f64::MIN_POSITIVE);
    let denominator = (1.0 - beta).max(f64::MIN_POSITIVE);
    let multiplier = ((beta * (0.5 * std::f64::consts::PI).sqrt()) / denominator) * (sigma / mean);
    let max_inner = f64::MAX.ln().sqrt();
    let sqrt_two = std::f64::consts::SQRT_2;

    let weights = residuals
        .into_iter()
        .map(|residual| {
            let inner = residual / (sigma * sqrt_two) - sigma / (mean * sqrt_two);
            let clipped_inner = inner.clamp(-max_inner, max_inner);
            let mut partial = (clipped_inner * clipped_inner).exp();
            if multiplier >= 0.5 {
                partial = partial.min(f64::MAX / (2.0 * multiplier));
            }
            1.0 / (1.0 + multiplier * (1.0 + libm::erf(inner)) * partial)
        })
        .collect();

    Some(weights)
}

pub(super) fn derpsalsa_weights(
    y: &[f64],
    baseline: &[f64],
    p: f64,
    k: f64,
    partial_weights: &[f64],
) -> Vec<f64> {
    y.iter()
        .zip(baseline)
        .zip(partial_weights)
        .map(|((observed, fitted), partial)| {
            let residual = observed - fitted;
            let asymmetric = if residual > 0.0 {
                p * (-0.5 * (residual / k).powi(2)).exp()
            } else {
                1.0 - p
            };
            asymmetric * partial
        })
        .collect()
}

pub(super) fn derivative_peak_screening_weights(
    y: &[f64],
    smooth_half_window: usize,
    num_smooths: usize,
) -> Vec<f64> {
    let smoothed = smooth_for_derivatives(y, smooth_half_window, num_smooths);
    let first = gradient(&smoothed);
    let second = gradient(&first);
    let first_rms = root_mean_square(&first).max(f64::MIN_POSITIVE);
    let second_rms = root_mean_square(&second).max(f64::MIN_POSITIVE);

    first
        .iter()
        .zip(&second)
        .map(|(first_deriv, second_deriv)| {
            (-0.5 * (first_deriv / first_rms).powi(2)).exp()
                * (-0.5 * (second_deriv / second_rms).powi(2)).exp()
        })
        .collect()
}

pub(super) fn iarpls_weights(y: &[f64], baseline: &[f64], iteration: usize) -> Option<Vec<f64>> {
    let residuals = residuals(y, baseline);
    let (_mean, std) = negative_residual_stats(&residuals)?;
    let scale = iteration.min(100) as f64;
    let scale = scale.exp() / std;
    let weights = residuals
        .into_iter()
        .map(|residual| {
            let inner = scale * (residual - 2.0 * std);
            0.5 * (1.0 - inner / (1.0 + inner * inner).sqrt())
        })
        .collect();

    Some(weights)
}

pub(super) fn lsrpls_weights(y: &[f64], baseline: &[f64], iteration: usize) -> Option<Vec<f64>> {
    let residuals = residuals(y, baseline);
    let (mean, std) = negative_residual_stats(&residuals)?;
    let scale = 10f64.powi(iteration.min(100) as i32) / std;
    let weights = residuals
        .into_iter()
        .map(|residual| {
            let inner = scale * (residual - (2.0 * std - mean));
            0.5 * (1.0 - inner / (1.0 + inner.abs()))
        })
        .collect();

    Some(weights)
}

pub(super) fn psalsa_weights(y: &[f64], baseline: &[f64], p: f64, k: f64) -> Vec<f64> {
    y.iter()
        .zip(baseline)
        .map(|(observed, fitted)| {
            let residual = observed - fitted;
            if residual > 0.0 {
                p * (-residual / k).exp()
            } else {
                1.0 - p
            }
        })
        .collect()
}

pub(super) fn mpls_anchor_weights(y: &[f64], radius: usize, p: f64) -> Vec<f64> {
    let rough_baseline = opening_reflect(y, radius);
    let mut padded = Vec::with_capacity(rough_baseline.len() + 2);
    if let Some(first) = rough_baseline.first() {
        padded.push(*first);
    }
    padded.extend_from_slice(&rough_baseline);
    if let Some(last) = rough_baseline.last() {
        padded.push(*last);
    }

    let diff: Vec<f64> = padded.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let indices: Vec<usize> = (0..y.len())
        .filter(|&index| {
            ((diff[index + 1] == 0.0) || (diff[index] == 0.0))
                && ((diff[index + 1] != 0.0) || (diff[index] != 0.0))
        })
        .collect();

    let mut weights = vec![p; y.len()];
    for pair_index in (1..indices.len().saturating_sub(1)).step_by(2) {
        let previous_segment = indices[pair_index];
        let next_segment = indices[pair_index + 1];
        if previous_segment > next_segment || next_segment >= y.len() {
            continue;
        }
        let mut min_index = previous_segment;
        let mut min_value = y[previous_segment];
        for (offset, value) in y[previous_segment..=next_segment].iter().enumerate() {
            if *value < min_value {
                min_value = *value;
                min_index = previous_segment + offset;
            }
        }
        weights[min_index] = 1.0 - p;
    }
    weights
}

pub(super) fn standard_deviation(values: &[f64]) -> f64 {
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

fn residuals(y: &[f64], baseline: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(baseline)
        .map(|(observed, fitted)| observed - fitted)
        .collect()
}

fn negative_residual_stats(residuals: &[f64]) -> Option<(f64, f64)> {
    let negative: Vec<f64> = residuals
        .iter()
        .copied()
        .filter(|residual| *residual < 0.0)
        .collect();
    if negative.len() < 2 {
        return None;
    }

    let mean = negative.iter().sum::<f64>() / negative.len() as f64;
    let variance = negative
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (negative.len() - 1) as f64;
    Some((mean, variance.sqrt().max(f64::MIN_POSITIVE)))
}

fn smooth_for_derivatives(y: &[f64], smooth_half_window: usize, num_smooths: usize) -> Vec<f64> {
    if smooth_half_window == 0 || num_smooths == 0 {
        return y.to_vec();
    }

    let kernel = mollifier_kernel(smooth_half_window);
    let mut current = extrapolate_pad(y, smooth_half_window);
    for _ in 0..num_smooths {
        current = convolve_reflect_same(&current, &kernel);
    }
    current[smooth_half_window..smooth_half_window + y.len()].to_vec()
}

fn mollifier_kernel(half_window: usize) -> Vec<f64> {
    if half_window == 0 {
        return vec![1.0];
    }
    let mut kernel = Vec::with_capacity(2 * half_window + 1);
    for index in 0..=2 * half_window {
        if index == 0 || index == 2 * half_window {
            kernel.push(0.0);
        } else {
            let x = (index as f64 - half_window as f64) / half_window as f64;
            kernel.push((-1.0 / (1.0 - x * x)).exp());
        }
    }
    let sum = kernel.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn extrapolate_pad(y: &[f64], pad: usize) -> Vec<f64> {
    if pad == 0 {
        return y.to_vec();
    }
    let left_slope = if y.len() > 1 { y[1] - y[0] } else { 0.0 };
    let right_slope = if y.len() > 1 {
        y[y.len() - 1] - y[y.len() - 2]
    } else {
        0.0
    };
    let mut output = Vec::with_capacity(y.len() + 2 * pad);
    for i in (1..=pad).rev() {
        output.push(y[0] - left_slope * i as f64);
    }
    output.extend_from_slice(y);
    let last = *y.last().unwrap_or(&0.0);
    for i in 1..=pad {
        output.push(last + right_slope * i as f64);
    }
    output
}

fn convolve_reflect_same(y: &[f64], kernel: &[f64]) -> Vec<f64> {
    let radius = kernel.len() / 2;
    let mut output = vec![0.0; y.len()];
    for (i, target) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (j, weight) in kernel.iter().enumerate() {
            let offset = j as isize - radius as isize;
            let index = reflect_index(i as isize + offset, y.len());
            sum += weight * y[index];
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
    let mut value = index.rem_euclid(period);
    if value >= len as isize {
        value = period - value;
    }
    value as usize
}

fn gradient(values: &[f64]) -> Vec<f64> {
    match values.len() {
        0 => Vec::new(),
        1 => vec![0.0],
        len => {
            let mut output = vec![0.0; len];
            output[0] = values[1] - values[0];
            output[len - 1] = values[len - 1] - values[len - 2];
            for i in 1..len - 1 {
                output[i] = 0.5 * (values[i + 1] - values[i - 1]);
            }
            output
        }
    }
}

fn root_mean_square(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum = values.iter().map(|value| value * value).sum::<f64>();
    (sum / values.len() as f64).sqrt()
}

fn opening_reflect(y: &[f64], radius: usize) -> Vec<f64> {
    let eroded = moving_min_reflect(y, radius);
    let mut opened = vec![0.0; y.len()];
    moving_max_reflect(&eroded, radius, &mut opened);
    opened
}

fn moving_min_reflect(y: &[f64], radius: usize) -> Vec<f64> {
    (0..y.len())
        .map(|i| {
            let start = i as isize - radius as isize;
            let end = i as isize + radius as isize;
            (start..=end)
                .map(|index| y[reflect_index(index, y.len())])
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

fn moving_max_reflect(y: &[f64], radius: usize, output: &mut [f64]) {
    for (i, target) in output.iter_mut().enumerate() {
        let start = i as isize - radius as isize;
        let end = i as isize + radius as isize;
        *target = (start..=end)
            .map(|index| y[reflect_index(index, y.len())])
            .fold(f64::NEG_INFINITY, f64::max);
    }
}
