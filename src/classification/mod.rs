//! Classification-style baseline algorithms.
//!
//! First-pass implementations expose the algorithm family using robust
//! lower-envelope fitting primitives. Golden pybaselines fixtures should drive
//! later refinements where individual classifiers differ.

use crate::fit::{Fit, FitReport};
use crate::linalg::pentadiagonal::{PentadiagonalWorkspace, solve_second_order};
use crate::polynomial::{evaluate_polynomial_coefficients, fit_weighted_polynomial_coefficients};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

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

/// Parameters for Dietrich-style baseline classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DietrichParams {
    /// Half-window for moving-average smoothing before derivative thresholding.
    pub smooth_half_window: usize,
    /// Number of standard deviations included in derivative-power thresholding.
    pub num_std: f64,
    /// Half-window for averaging interpolation anchor points.
    pub interp_half_window: usize,
    /// Polynomial order for iterative fitting; ignored when `max_iter` is zero.
    pub poly_order: usize,
    /// Maximum polynomial refinement iterations. Set to zero to return linear interpolation.
    pub max_iter: usize,
    /// Relative coefficient-change tolerance for polynomial refinement.
    pub tol: f64,
    /// Minimum consecutive baseline-region length.
    pub min_length: usize,
}

impl Default for DietrichParams {
    fn default() -> Self {
        Self {
            smooth_half_window: 1,
            num_std: 3.0,
            interp_half_window: 5,
            poly_order: 5,
            max_iter: 50,
            tol: 1.0e-3,
            min_length: 2,
        }
    }
}

impl DietrichParams {
    fn validate(&self, len: usize) -> Result<()> {
        if !self.num_std.is_finite() || self.num_std <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "num_std",
                reason: "must be finite and positive",
            });
        }
        if self.max_iter > 0 && self.poly_order + 1 > len {
            return Err(BaselineError::TooShort {
                algorithm: "dietrich",
                len,
                min: self.poly_order + 1,
            });
        }
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "tol",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for fully automatic baseline correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FabcParams {
    /// Whittaker smoothing parameter. Larger values produce smoother baselines.
    pub lambda: f64,
    /// Scale for the Haar wavelet derivative estimate.
    pub scale: usize,
    /// Number of standard deviations included in wavelet-power thresholding.
    pub num_std: f64,
    /// Differential order for the Whittaker penalty. Currently only `2` is supported.
    pub diff_order: usize,
    /// Minimum consecutive baseline-region length.
    pub min_length: usize,
}

impl Default for FabcParams {
    fn default() -> Self {
        Self {
            lambda: 1.0e6,
            scale: 8,
            num_std: 3.0,
            diff_order: 2,
            min_length: 2,
        }
    }
}

impl FabcParams {
    fn validate(&self) -> Result<()> {
        if !self.lambda.is_finite() || self.lambda <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "lambda",
                reason: "must be finite and positive",
            });
        }
        if self.scale == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "scale",
                reason: "must be greater than zero",
            });
        }
        if !self.num_std.is_finite() || self.num_std <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "num_std",
                reason: "must be finite and positive",
            });
        }
        if self.diff_order != 2 {
            return Err(BaselineError::InvalidParameter {
                name: "diff_order",
                reason: "only second-order Whittaker penalties are currently supported",
            });
        }
        Ok(())
    }
}

/// Parameters for continuous-wavelet-transform baseline recognition.
#[derive(Debug, Clone, PartialEq)]
pub struct CwtBrParams {
    /// Polynomial order for fitting the identified baseline points.
    pub poly_order: usize,
    /// CWT scales to evaluate. `None` uses the pybaselines default scale range.
    pub scales: Option<Vec<usize>>,
    /// Number of residual standard deviations used during iterative masking.
    pub num_std: f64,
    /// Minimum consecutive baseline-region length.
    pub min_length: usize,
    /// Maximum polynomial refinement iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
    /// Preserve both positive and negative peak regions during polynomial fitting.
    pub symmetric: bool,
}

impl Default for CwtBrParams {
    fn default() -> Self {
        Self {
            poly_order: 5,
            scales: None,
            num_std: 1.0,
            min_length: 2,
            max_iter: 50,
            tol: 1.0e-3,
            symmetric: false,
        }
    }
}

impl CwtBrParams {
    fn validate(&self, len: usize) -> Result<()> {
        if self.poly_order + 1 > len {
            return Err(BaselineError::TooShort {
                algorithm: "cwt_br",
                len,
                min: self.poly_order + 1,
            });
        }
        if self
            .scales
            .as_ref()
            .is_some_and(|scales| scales.is_empty() || scales.contains(&0))
        {
            return Err(BaselineError::InvalidParameter {
                name: "scales",
                reason: "must be non-empty and contain positive scales when set",
            });
        }
        if !self.num_std.is_finite() || self.num_std <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "num_std",
                reason: "must be finite and positive",
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
        Ok(())
    }
}

/// Estimates a baseline using Dietrich-style peak classification.
///
/// # References
///
/// - W. Dietrich et al., "Fast and Precise Automatic Baseline Correction of
///   One- and Two-Dimensional NMR Spectra", *Journal of Magnetic Resonance*,
///   1991.
/// - `pybaselines.Baseline.dietrich` is used as a behavioral reference.
pub fn dietrich(y: &[f64], params: DietrichParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate(y.len())?;

    let smooth_y = moving_average_extrapolated(y, params.smooth_half_window);
    let power: Vec<f64> = gradient(&smooth_y)
        .into_iter()
        .map(|value| value * value)
        .collect();
    let mut mask = iter_threshold(&power, params.num_std);
    refine_mask(&mut mask, params.min_length);

    let mut rough_baseline = averaged_interp(y, &mask, params.interp_half_window);
    if params.max_iter == 0 {
        return Ok(Fit {
            baseline: rough_baseline,
            report: FitReport::new(1, true, 0.0),
        });
    }

    let weights = vec![1.0; y.len()];
    let mut coeffs =
        fit_weighted_polynomial_coefficients(&rough_baseline, &weights, params.poly_order)?;
    let mut baseline = vec![0.0; y.len()];
    evaluate_polynomial_coefficients(&coeffs, &mut baseline);
    let mut tolerance = f64::INFINITY;

    for iter in 1..params.max_iter {
        for ((rough, fitted), keep) in rough_baseline.iter_mut().zip(&baseline).zip(&mask) {
            if *keep {
                *rough = *fitted;
            }
        }
        let new_coeffs =
            fit_weighted_polynomial_coefficients(&rough_baseline, &weights, params.poly_order)?;
        evaluate_polynomial_coefficients(&new_coeffs, &mut baseline);
        tolerance = relative_difference(&coeffs, &new_coeffs);
        if tolerance < params.tol {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iter + 1, true, tolerance),
            });
        }
        coeffs = new_coeffs;
    }

    Ok(Fit {
        baseline,
        report: FitReport::new(params.max_iter, false, tolerance),
    })
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
/// - C. Bertinetto et al., "Automatic Baseline Recognition for the Correction
///   of Large Sets of Spectra Using Continuous Wavelet Transform and Iterative
///   Fitting", *Applied Spectroscopy*, 2014.
/// - `pybaselines.Baseline.cwt_br` is used as a behavioral reference.
pub fn cwt_br(y: &[f64], params: CwtBrParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate(y.len())?;

    let (min_y, max_y) = min_max(y);
    if (max_y - min_y).abs() <= f64::EPSILON {
        return Ok(Fit {
            baseline: y.to_vec(),
            report: FitReport::new(1, true, 0.0),
        });
    }

    let scaled_y: Vec<f64> = y
        .iter()
        .map(|value| 2.0 * (value - min_y) / (max_y - min_y) - 1.0)
        .collect();
    let scales = cwt_scales(y.len(), &params);
    let max_scale = *scales.iter().max().expect("validated scales are non-empty");
    let half_window = 2 * max_scale;
    let padded_y = extrapolate_pad(&scaled_y, half_window);
    let (_best_scale, wavelet_cwt) = best_ricker_cwt(&padded_y, y.len(), half_window, &scales);
    let threshold = 0.6 * sample_standard_deviation(&wavelet_cwt);
    let mut mask: Vec<bool> = wavelet_cwt
        .iter()
        .map(|value| value.abs() < threshold)
        .collect();
    refine_mask(&mut mask, params.min_length);

    let check_radius = y.len() / 200;
    let mut baseline_old = scaled_y.clone();
    let mut baseline = vec![0.0; y.len()];
    let mut tolerance = f64::INFINITY;
    let mut iterations = 0usize;

    for iter in 0..=params.max_iter {
        fit_masked_polynomial(&scaled_y, &mask, params.poly_order, &mut baseline)?;
        let residual: Vec<f64> = scaled_y
            .iter()
            .zip(&baseline)
            .map(|(observed, fitted)| observed - fitted)
            .collect();
        let residual_std = population_standard_deviation(&residual);
        for (keep, residual) in mask.iter_mut().zip(&residual) {
            if *residual > params.num_std * residual_std {
                *keep = false;
            }
        }

        fit_masked_polynomial(&scaled_y, &mask, params.poly_order, &mut baseline)?;
        tolerance = relative_difference(&baseline_old, &baseline);
        iterations = iter + 1;
        if tolerance < params.tol {
            break;
        }
        baseline_old.copy_from_slice(&baseline);
        if !params.symmetric {
            let below_fit: Vec<bool> = scaled_y
                .iter()
                .zip(&baseline)
                .map(|(observed, fitted)| observed < fitted)
                .collect();
            for (keep, eroded) in mask.iter_mut().zip(erode_mask(&below_fit, check_radius)) {
                *keep |= eroded;
            }
        }
    }

    for value in &mut baseline {
        *value = min_y + 0.5 * (*value + 1.0) * (max_y - min_y);
    }

    Ok(Fit {
        baseline,
        report: FitReport::new(iterations, tolerance < params.tol, tolerance),
    })
}

/// Estimates a baseline using fully automatic baseline correction.
///
/// # References
///
/// - J. C. Cobas et al., "A new general-purpose fully automatic
///   baseline-correction procedure for 1D and 2D NMR data", *Journal of
///   Magnetic Resonance*, 2006.
/// - `pybaselines.Baseline.fabc` is used as a behavioral reference.
pub fn fabc(y: &[f64], params: FabcParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;

    let power = haar_cwt_power(y, params.scale);
    let mut mask = iter_threshold(&power, params.num_std);
    refine_mask(&mut mask, params.min_length);

    let weights: Vec<f64> = mask
        .iter()
        .map(|is_baseline| if *is_baseline { 1.0 } else { 0.0 })
        .collect();
    let mut baseline = vec![0.0; y.len()];
    let mut workspace = PentadiagonalWorkspace::new(y.len());
    solve_second_order(y, &weights, params.lambda, &mut baseline, &mut workspace)?;

    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
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

fn min_max(y: &[f64]) -> (f64, f64) {
    y.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        })
}

fn cwt_scales(len: usize, params: &CwtBrParams) -> Vec<usize> {
    if let Some(scales) = &params.scales {
        return scales.clone();
    }
    let min_scale = 2.max(len / 500);
    let max_scale = len / 4;
    if max_scale > min_scale {
        (min_scale..max_scale).collect()
    } else {
        vec![min_scale]
    }
}

fn best_ricker_cwt(
    padded_y: &[f64],
    len: usize,
    half_window: usize,
    scales: &[usize],
) -> (usize, Vec<f64>) {
    let mut shannon_old = f64::NEG_INFINITY;
    let mut shannon_current = f64::NEG_INFINITY;
    let mut best_scale = scales[0];
    let mut best_cwt = Vec::new();

    for scale in scales {
        let cwt = ricker_cwt(padded_y, len, half_window, *scale);
        let entropy = shannon_entropy(&cwt);
        best_scale = *scale;
        best_cwt = cwt;
        if shannon_current < shannon_old && entropy > shannon_current {
            break;
        }
        shannon_old = shannon_current;
        shannon_current = entropy;
    }

    (best_scale, best_cwt)
}

fn ricker_cwt(padded_y: &[f64], len: usize, half_window: usize, scale: usize) -> Vec<f64> {
    let wavelet_len = (10 * scale).min(padded_y.len()).max(1);
    let mut wavelet = ricker_wavelet(wavelet_len, scale as f64);
    wavelet.reverse();
    let cwt = convolve_same(padded_y, &wavelet);
    cwt[half_window..half_window + len].to_vec()
}

fn ricker_wavelet(points: usize, scale: f64) -> Vec<f64> {
    let amplitude = 2.0 / ((3.0 * scale).sqrt() * std::f64::consts::PI.powf(0.25));
    let scale_squared = scale * scale;
    let center = (points as f64 - 1.0) / 2.0;
    (0..points)
        .map(|index| {
            let x = index as f64 - center;
            let x_squared = x * x;
            amplitude
                * (1.0 - x_squared / scale_squared)
                * (-x_squared / (2.0 * scale_squared)).exp()
        })
        .collect()
}

fn shannon_entropy(values: &[f64]) -> f64 {
    let total = values.iter().map(|value| value.abs()).sum::<f64>();
    if total <= f64::EPSILON {
        return 0.0;
    }
    -values
        .iter()
        .map(|value| {
            let probability = value.abs() / total;
            probability * (probability + f64::MIN_POSITIVE).ln()
        })
        .sum::<f64>()
}

fn fit_masked_polynomial(
    y: &[f64],
    mask: &[bool],
    order: usize,
    baseline: &mut [f64],
) -> Result<()> {
    if mask.iter().filter(|keep| **keep).count() < order + 1 {
        return Err(BaselineError::TooShort {
            algorithm: "cwt_br",
            len: mask.iter().filter(|keep| **keep).count(),
            min: order + 1,
        });
    }
    let weights: Vec<f64> = mask
        .iter()
        .map(|keep| if *keep { 1.0 } else { 0.0 })
        .collect();
    let coeffs = fit_weighted_polynomial_coefficients(y, &weights, order)?;
    evaluate_polynomial_coefficients(&coeffs, baseline);
    Ok(())
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

fn population_standard_deviation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = mean(values);
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

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn gradient(y: &[f64]) -> Vec<f64> {
    match y.len() {
        0 => Vec::new(),
        1 => vec![0.0],
        len => {
            let mut output = vec![0.0; len];
            output[0] = y[1] - y[0];
            output[len - 1] = y[len - 1] - y[len - 2];
            for index in 1..len - 1 {
                output[index] = 0.5 * (y[index + 1] - y[index - 1]);
            }
            output
        }
    }
}

fn iter_threshold(power: &[f64], num_std: f64) -> Vec<bool> {
    let threshold = mean(power) + num_std * sample_standard_deviation(power);
    let mut mask: Vec<bool> = power.iter().map(|value| *value < threshold).collect();
    loop {
        let masked_power: Vec<f64> = power
            .iter()
            .zip(&mask)
            .filter_map(|(value, keep)| keep.then_some(*value))
            .collect();
        if masked_power.len() < 2 {
            return mask;
        }
        let threshold = mean(&masked_power) + num_std * sample_standard_deviation(&masked_power);
        let new_mask: Vec<bool> = power.iter().map(|value| *value < threshold).collect();
        if new_mask == mask {
            return mask;
        }
        mask = new_mask;
    }
}

fn haar_cwt_power(y: &[f64], scale: usize) -> Vec<f64> {
    let half_window = 2 * scale;
    let padded = extrapolate_pad(y, half_window);
    let wavelet_len = (10 * scale).min(padded.len());
    let mut wavelet = haar_wavelet(wavelet_len, scale);
    wavelet.reverse();
    let cwt = convolve_same(&padded, &wavelet);
    cwt[half_window..half_window + y.len()]
        .iter()
        .map(|value| value * value)
        .collect()
}

fn haar_wavelet(mut num_points: usize, scale: usize) -> Vec<f64> {
    let odd_scale = !scale.is_multiple_of(2);
    let odd_window = !num_points.is_multiple_of(2);
    if (odd_scale && !odd_window) || (!odd_scale && odd_window) {
        num_points += 1;
    }
    let center = (num_points - 1) as f64 / 2.0;
    let half_scale = scale as f64 / 2.0;
    let norm = (scale as f64).sqrt();
    (0..num_points)
        .map(|index| {
            let x = index as f64 - center;
            let value = if odd_scale {
                if x > -half_scale && x < 0.0 {
                    1.0
                } else if x < half_scale && x > 0.0 {
                    -1.0
                } else {
                    0.0
                }
            } else if x >= -half_scale && x < 0.0 {
                1.0
            } else if x < half_scale && x >= 0.0 {
                -1.0
            } else {
                0.0
            };
            value / norm
        })
        .collect()
}

fn convolve_same(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    let start = (kernel.len() - 1) / 2;
    (0..data.len())
        .map(|output_index| {
            let full_index = output_index + start;
            let data_start = full_index.saturating_sub(kernel.len() - 1);
            let data_end = full_index.min(data.len() - 1);
            (data_start..=data_end)
                .map(|data_index| {
                    let kernel_index = full_index - data_index;
                    data[data_index] * kernel[kernel_index]
                })
                .sum()
        })
        .collect()
}

fn relative_difference(previous: &[f64], current: &[f64]) -> f64 {
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

fn erode_mask(mask: &[bool], radius: usize) -> Vec<bool> {
    if radius == 0 {
        return mask.to_vec();
    }
    let mut output = vec![false; mask.len()];
    for (index, target) in output.iter_mut().enumerate() {
        if index < radius || index + radius >= mask.len() {
            continue;
        }
        *target = mask[index - radius..=index + radius]
            .iter()
            .all(|value| *value);
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
