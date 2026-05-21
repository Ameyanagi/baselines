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

/// Parameters for distribution-based baseline classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StdDistributionParams {
    /// Half-window for rolling standard-deviation calculations.
    pub half_window: usize,
    /// Half-window for averaging interpolation anchor points.
    pub interp_half_window: usize,
    /// Half-window for expanding detected peak regions.
    pub fill_half_window: usize,
    /// Multiple of the estimated noise standard deviation used for thresholding.
    pub num_std: f64,
    /// Half-window for smoothing the interpolated baseline.
    pub smooth_half_window: usize,
}

impl Default for StdDistributionParams {
    fn default() -> Self {
        Self {
            half_window: 8,
            interp_half_window: 5,
            fill_half_window: 3,
            num_std: 1.1,
            smooth_half_window: 8,
        }
    }
}

impl StdDistributionParams {
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
        Ok(())
    }
}

/// Parameters for FastChrom-style baseline classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastChromParams {
    /// Half-window for rolling standard-deviation calculations.
    pub half_window: usize,
    /// Optional rolling standard-deviation threshold. Uses the 15th percentile when `None`.
    pub threshold: Option<f64>,
    /// Minimum width for adding an extra baseline point during correction.
    pub min_fwhm: Option<usize>,
    /// Half-window for averaging interpolation anchor points.
    pub interp_half_window: usize,
    /// Half-window for smoothing the interpolated baseline.
    pub smooth_half_window: usize,
    /// Maximum number of interpolation correction passes.
    pub max_iter: usize,
    /// Minimum consecutive baseline-region length.
    pub min_length: usize,
}

impl Default for FastChromParams {
    fn default() -> Self {
        Self {
            half_window: 8,
            threshold: None,
            min_fwhm: None,
            interp_half_window: 5,
            smooth_half_window: 8,
            max_iter: 100,
            min_length: 2,
        }
    }
}

impl FastChromParams {
    fn validate(&self) -> Result<()> {
        if self.half_window == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "half_window",
                reason: "must be greater than zero",
            });
        }
        if self.threshold.is_some_and(|value| !value.is_finite()) {
            return Err(BaselineError::InvalidParameter {
                name: "threshold",
                reason: "must be finite when set",
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
/// - K. C. Wang et al., "Distribution-Based Classification Method for Baseline
///   Correction of Metabolomic 1D Proton Nuclear Magnetic Resonance Spectra",
///   *Analytical Chemistry*, 2013.
/// - `pybaselines.Baseline.std_distribution` is used as a behavioral reference.
pub fn std_distribution(y: &[f64], params: StdDistributionParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;

    let rolling_std = padded_rolling_std(y, params.half_window, 1);
    let mut median = median(&rolling_std);
    let mut median_2 = median_below(&rolling_std, 2.0 * median);
    while median_2 / median < 0.999 {
        median = median_2;
        median_2 = median_below(&rolling_std, 2.0 * median);
    }
    let noise_std = median_2;
    let peak_regions: Vec<bool> = rolling_std
        .iter()
        .map(|value| *value > params.num_std * noise_std)
        .collect();
    let dilated_peaks = dilate_mask(&peak_regions, params.fill_half_window);
    let mask: Vec<bool> = dilated_peaks.iter().map(|is_peak| !is_peak).collect();

    let rough_baseline = averaged_interp(y, &mask, params.interp_half_window);
    let baseline = moving_average_extrapolated(&rough_baseline, params.smooth_half_window);
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Estimates a baseline using FastChrom-style classification.
///
/// # References
///
/// - L. Johnsen et al., "An automated method for baseline correction, peak
///   finding and peak grouping in chromatographic data", *Analyst*, 2013.
/// - `pybaselines.Baseline.fastchrom` is used as a behavioral reference.
pub fn fastchrom(y: &[f64], params: FastChromParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;

    let rolling_std = padded_rolling_std(y, params.half_window, 1);
    let threshold = params
        .threshold
        .unwrap_or_else(|| percentile(&rolling_std, 15.0));
    let mut mask: Vec<bool> = rolling_std.iter().map(|value| *value < threshold).collect();
    refine_mask(&mut mask, params.min_length);

    let min_fwhm = params.min_fwhm.unwrap_or(2 * params.half_window);
    let mut rough_baseline = averaged_interp(y, &mask, params.interp_half_window);
    let mask_sum = mask.iter().filter(|value| **value).count();
    if mask_sum != 0 && mask_sum != mask.len() {
        let initial_peak_segments = peak_segments(&mask);
        for _ in 0..params.max_iter {
            let mut modified_baseline = false;
            for (start, end) in initial_peak_segments.iter().copied() {
                let section_mask: Vec<bool> = rough_baseline[start..=end]
                    .iter()
                    .zip(&y[start..=end])
                    .map(|(baseline, observed)| baseline < observed)
                    .collect();
                let has_wide_above_data_segment = peak_segments(&section_mask)
                    .iter()
                    .any(|(seg_start, seg_end)| seg_end - seg_start > min_fwhm);
                if has_wide_above_data_segment {
                    modified_baseline = true;
                    let local_min = y[start..=end]
                        .iter()
                        .zip(&rough_baseline[start..=end])
                        .enumerate()
                        .min_by(
                            |(_, (left_y, left_baseline)), (_, (right_y, right_baseline))| {
                                (*left_y - *left_baseline).total_cmp(&(*right_y - *right_baseline))
                            },
                        )
                        .map(|(index, _)| index)
                        .unwrap_or(0);
                    mask[start + local_min] = true;
                }
            }
            if modified_baseline {
                rough_baseline = averaged_interp(y, &mask, params.interp_half_window);
            } else {
                break;
            }
        }
    }

    let baseline = moving_average_extrapolated(&rough_baseline, params.smooth_half_window);
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
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

fn padded_rolling_std(y: &[f64], radius: usize, ddof: usize) -> Vec<f64> {
    let window_size = 2 * radius + 1;
    (0..y.len())
        .map(|index| {
            let candidates = index as isize - radius as isize..=index as isize + radius as isize;
            let sum = candidates
                .clone()
                .map(|candidate| y[reflect_pad_index(candidate, y.len())])
                .sum::<f64>();
            let mean = sum / window_size as f64;
            let variance = candidates
                .map(|candidate| {
                    let value = y[reflect_pad_index(candidate, y.len())];
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f64>()
                / (window_size - ddof) as f64;
            variance.sqrt()
        })
        .collect()
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn percentile(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = percentile / 100.0 * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower].mul_add(1.0 - fraction, sorted[upper] * fraction)
    }
}

fn median_below(values: &[f64], threshold: f64) -> f64 {
    let filtered: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| *value < threshold)
        .collect();
    median(&filtered)
}

fn dilate_mask(mask: &[bool], radius: usize) -> Vec<bool> {
    let mut output = vec![false; mask.len()];
    for (index, is_set) in mask.iter().copied().enumerate() {
        if is_set {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(mask.len());
            output[start..end].fill(true);
        }
    }
    output
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

fn reflect_pad_index(index: isize, len: usize) -> usize {
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
