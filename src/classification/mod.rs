//! Classification-style baseline algorithms.
//!
//! First-pass implementations expose the algorithm family using robust
//! lower-envelope fitting primitives. Golden pybaselines fixtures should drive
//! later refinements where individual classifiers differ.

use crate::fit::{Fit, FitReport};
use crate::smoothing::{SmoothingParams, noise_median};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

/// Parameters for classification-style baseline methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassificationParams {
    /// Smoothing window used while identifying the lower envelope.
    pub window_size: usize,
}

impl Default for ClassificationParams {
    fn default() -> Self {
        Self { window_size: 31 }
    }
}

/// Parameters for Golotvin-style baseline classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GolotvinParams {
    /// Half-window for rolling maximum and minimum calculations.
    pub half_window: usize,
    /// Number of standard deviations included in the baseline threshold.
    pub num_std: f64,
    /// Number of sections for estimating the minimum local standard deviation.
    pub sections: usize,
    /// Half-window for smoothing the interpolated baseline.
    pub smooth_half_window: usize,
    /// Half-window for averaging interpolation anchor points.
    pub interp_half_window: usize,
    /// Minimum consecutive baseline-region length.
    pub min_length: usize,
}

impl Default for GolotvinParams {
    fn default() -> Self {
        Self {
            half_window: 8,
            num_std: 2.0,
            sections: 32,
            smooth_half_window: 8,
            interp_half_window: 5,
            min_length: 2,
        }
    }
}

impl GolotvinParams {
    fn validate(&self) -> Result<()> {
        if self.half_window == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "half_window",
                reason: "must be greater than zero",
            });
        }
        if !self.num_std.is_finite() || self.num_std <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "num_std",
                reason: "must be finite and positive",
            });
        }
        if self.sections == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "sections",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }
}

/// Estimates a baseline using Dietrich-style peak classification.
///
/// # References
///
/// - `pybaselines.Baseline.dietrich` is used as a behavioral reference.
pub fn dietrich(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    envelope_from_smoothed(y, params)
}

/// Estimates a baseline using Golotvin-style peak classification.
///
/// # References
///
/// - S. Golotvin and A. Williams, "Improved Baseline Recognition and Modeling
///   of FT NMR Spectra", *Journal of Magnetic Resonance*, 2000.
/// - `pybaselines.Baseline.golotvin` is used as a behavioral reference.
pub fn golotvin(y: &[f64], params: GolotvinParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;

    let min_sigma = minimum_section_std(y, params.sections);
    let max_values = rolling_extreme_reflect(y, params.half_window, f64::max, f64::NEG_INFINITY);
    let min_values = rolling_extreme_reflect(y, params.half_window, f64::min, f64::INFINITY);
    let mut mask: Vec<bool> = max_values
        .iter()
        .zip(&min_values)
        .map(|(max_value, min_value)| max_value - min_value < params.num_std * min_sigma)
        .collect();
    refine_mask(&mut mask, params.min_length);

    let rough_baseline = averaged_interp(y, &mask, params.interp_half_window);
    let baseline = moving_average_extrapolated(&rough_baseline, params.smooth_half_window);
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Estimates a baseline using standard-deviation distribution classification.
///
/// # References
///
/// - `pybaselines.Baseline.std_distribution` is used as a behavioral reference.
pub fn std_distribution(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    envelope_from_smoothed(y, params)
}

/// Estimates a baseline using FastChrom-style classification.
///
/// # References
///
/// - `pybaselines.Baseline.fastchrom` is used as a behavioral reference.
pub fn fastchrom(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    envelope_from_smoothed(y, params)
}

/// Estimates a baseline using continuous-wavelet-transform classification.
///
/// # References
///
/// - `pybaselines.Baseline.cwt_br` is used as a behavioral reference.
pub fn cwt_br(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    envelope_from_smoothed(y, params)
}

/// Estimates a baseline using fully automatic baseline correction.
///
/// # References
///
/// - `pybaselines.Baseline.fabc` is used as a behavioral reference.
pub fn fabc(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    envelope_from_smoothed(y, params)
}

/// Estimates a baseline using a lower convex-hull rubberband.
///
/// # References
///
/// - `pybaselines.Baseline.rubberband` is used as a behavioral reference.
pub fn rubberband(y: &[f64]) -> Result<Fit> {
    validate_signal(y)?;
    let hull = lower_hull(y);
    let baseline = interpolate_hull(y.len(), &hull);
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

fn envelope_from_smoothed(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    let smooth = noise_median(
        y,
        SmoothingParams {
            window_size: params.window_size,
            max_iter: 1,
        },
    )?;
    let fit = rubberband(&smooth.baseline)?;
    Ok(Fit {
        baseline: fit.baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

fn minimum_section_std(y: &[f64], sections: usize) -> f64 {
    let mut min_sigma = f64::INFINITY;
    for section in 0..sections {
        let left = section * y.len() / sections;
        let right = (section + 1) * y.len() / sections;
        if right > left + 1 {
            min_sigma = min_sigma.min(sample_standard_deviation(&y[left..right]));
        }
    }
    min_sigma
}

fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
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
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn rolling_extreme_reflect(
    y: &[f64],
    radius: usize,
    op: fn(f64, f64) -> f64,
    initial: f64,
) -> Vec<f64> {
    (0..y.len())
        .map(|index| {
            let start = index as isize - radius as isize;
            let end = index as isize + radius as isize;
            (start..=end)
                .map(|candidate| y[reflect_index(candidate, y.len())])
                .fold(initial, op)
        })
        .collect()
}

fn refine_mask(mask: &mut [bool], min_length: usize) {
    let min_length = min_length.max(1);
    let mut index = 0usize;
    while index < mask.len() {
        let value = mask[index];
        let start = index;
        while index < mask.len() && mask[index] == value {
            index += 1;
        }
        if value && index - start < min_length {
            mask[start..index].fill(false);
        }
    }

    if mask.len() < 3 {
        return;
    }
    let mut output = mask.to_vec();
    for index in 1..mask.len() - 1 {
        if !mask[index] && mask[index - 1] && mask[index + 1] {
            output[index] = true;
        }
    }
    mask.copy_from_slice(&output);
}

fn averaged_interp(y: &[f64], mask: &[bool], half_window: usize) -> Vec<f64> {
    let mut output = y.to_vec();
    if mask.iter().all(|keep| *keep) {
        return output;
    }

    for (start, end) in peak_segments(mask) {
        if end <= start + 1 {
            continue;
        }
        let left_mean = window_mean(y, start, half_window);
        let right_mean = window_mean(y, end, half_window);
        let width = (end - start) as f64;
        for (index, target) in output.iter_mut().enumerate().take(end).skip(start + 1) {
            let t = (index - start) as f64 / width;
            *target = left_mean.mul_add(1.0 - t, right_mean * t);
        }
    }
    output
}

fn peak_segments(mask: &[bool]) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut index = 0usize;
    while index < mask.len() {
        if mask[index] {
            index += 1;
            continue;
        }
        let start = index.saturating_sub(1);
        while index < mask.len() && !mask[index] {
            index += 1;
        }
        let end = if index < mask.len() {
            index
        } else {
            mask.len().saturating_sub(1)
        };
        segments.push((start, end));
    }
    segments
}

fn window_mean(y: &[f64], index: usize, radius: usize) -> f64 {
    let start = index.saturating_sub(radius);
    let end = (index + radius + 1).min(y.len());
    y[start..end].iter().sum::<f64>() / (end - start) as f64
}

fn moving_average_extrapolated(y: &[f64], radius: usize) -> Vec<f64> {
    if radius == 0 {
        return y.to_vec();
    }
    let padded = extrapolate_pad(y, radius);
    (0..y.len())
        .map(|index| {
            let start = index;
            let end = index + 2 * radius + 1;
            padded[start..end].iter().sum::<f64>() / (2 * radius + 1) as f64
        })
        .collect()
}

fn extrapolate_pad(y: &[f64], radius: usize) -> Vec<f64> {
    let fit_window = radius.min(y.len());
    let (left_intercept, left_slope) = linear_fit_edge(&y[..fit_window], 0);
    let right_start = y.len() - fit_window;
    let (right_intercept, right_slope) = linear_fit_edge(&y[right_start..], right_start);
    let mut output = Vec::with_capacity(y.len() + 2 * radius);
    for offset in (1..=radius).rev() {
        let x = -(offset as f64);
        output.push(left_intercept + left_slope * x);
    }
    output.extend_from_slice(y);
    for offset in 1..=radius {
        let x = (y.len() - 1 + offset) as f64;
        output.push(right_intercept + right_slope * x);
    }
    output
}

fn linear_fit_edge(y: &[f64], start_index: usize) -> (f64, f64) {
    if y.len() <= 1 {
        return (*y.first().unwrap_or(&0.0), 0.0);
    }
    let len = y.len() as f64;
    let mean_x = start_index as f64 + (len - 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / len;
    let (numerator, denominator) =
        y.iter()
            .enumerate()
            .fold((0.0, 0.0), |(num, den), (offset, value)| {
                let centered_x = (start_index + offset) as f64 - mean_x;
                let centered_y = value - mean_y;
                (
                    centered_x.mul_add(centered_y, num),
                    centered_x.mul_add(centered_x, den),
                )
            });
    let slope = numerator / denominator;
    (mean_y - slope * mean_x, slope)
}

fn reflect_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len as isize;
    let mut value = index.rem_euclid(period);
    if value >= len as isize {
        value = period - value - 1;
    }
    value as usize
}

fn lower_hull(y: &[f64]) -> Vec<(usize, f64)> {
    let mut hull: Vec<(usize, f64)> = Vec::new();
    for (i, value) in y.iter().copied().enumerate() {
        hull.push((i, value));
        while hull.len() >= 3 {
            let len = hull.len();
            let a = hull[len - 3];
            let b = hull[len - 2];
            let c = hull[len - 1];
            if cross(a, b, c) > 0.0 {
                break;
            }
            hull.remove(len - 2);
        }
    }
    hull
}

fn cross(a: (usize, f64), b: (usize, f64), c: (usize, f64)) -> f64 {
    let abx = (b.0 - a.0) as f64;
    let aby = b.1 - a.1;
    let acx = (c.0 - a.0) as f64;
    let acy = c.1 - a.1;
    abx * acy - aby * acx
}

fn interpolate_hull(n: usize, hull: &[(usize, f64)]) -> Vec<f64> {
    let mut baseline = vec![0.0; n];
    for pair in hull.windows(2) {
        let (start, y0) = pair[0];
        let (end, y1) = pair[1];
        let width = (end - start).max(1) as f64;
        for (offset, target) in baseline[start..=end].iter_mut().enumerate() {
            let t = offset as f64 / width;
            *target = y0.mul_add(1.0 - t, y1 * t);
        }
    }
    if let Some(&(index, value)) = hull.last() {
        baseline[index] = value;
    }
    baseline
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rubberband_keeps_constant_signal() {
        let y = vec![4.0; 16];
        let fit = rubberband(&y).unwrap();
        assert!(
            fit.baseline
                .iter()
                .all(|value| (*value - 4.0).abs() < 1e-12)
        );
    }
}
