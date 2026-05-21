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
