//! Morphological and smoothing baseline algorithms.
//!
//! # References
//!
//! - M. Kneen and H. Annegarn, "Algorithm for fitting XRF, SEM and PIXE
//!   X-ray spectra backgrounds", *Nuclear Instruments and Methods in Physics
//!   Research Section B*, 1996.
//! - Z. Li et al., "Morphological weighted penalized least squares for
//!   background correction", *Analyst*, 2013.
//! - L. Dai et al., "An Automated Baseline Correction Method Based on
//!   Iterative Morphological Operations", *Applied Spectroscopy*, 2018.
//! - C. G. Ryan et al., "SNIP, a statistics-sensitive background treatment
//!   for the quantitative analysis of PIXE spectra", 1988.
//! - `pybaselines` is used as a behavioral reference.

use crate::fit::{Fit, FitReport};
use crate::linalg::pentadiagonal::{
    GeneralPentadiagonalSystem, GeneralPentadiagonalWorkspace, PentadiagonalWorkspace,
    solve_general_pentadiagonal, solve_second_order,
};
use crate::linalg::pspline::PenalizedSpline;
use crate::workspace::{validate_output, validate_signal};
use crate::{BaselineError, Result};

const MPLS_LAMBDA: f64 = 1.0e6;
const MPLS_P: f64 = 0.0;
const IMOR_MAX_ITER: usize = 200;
const IMOR_TOL: f64 = 1.0e-3;
const MORMOL_MAX_ITER: usize = 250;
const MORMOL_TOL: f64 = 1.0e-3;
const JBCD_ALPHA: f64 = 0.1;
const JBCD_BETA: f64 = 10.0;
const JBCD_GAMMA: f64 = 1.0;
const JBCD_BETA_MULT: f64 = 1.1;
const JBCD_GAMMA_MULT: f64 = 0.909;
const JBCD_MAX_ITER: usize = 20;
const JBCD_SIGNAL_TOL: f64 = 1.0e-2;
const JBCD_BASELINE_TOL: f64 = 1.0e-3;
const MPSPLINE_LAMBDA: f64 = 1.0e4;
const MPSPLINE_SMOOTH_LAMBDA: f64 = 1.0e-2;
const MPSPLINE_P: f64 = 0.0;
const MPSPLINE_NUM_KNOTS: usize = 100;
const MPSPLINE_DEGREE: usize = 3;
const MPSPLINE_DIFF_ORDER: usize = 2;

/// Parameters for window-based morphology baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MorphologyParams {
    /// Full moving-window size. Even values are rounded up internally.
    pub window_size: usize,
}

impl Default for MorphologyParams {
    fn default() -> Self {
        Self { window_size: 31 }
    }
}

impl MorphologyParams {
    fn validate(&self) -> Result<()> {
        if self.window_size == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "window_size",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }

    fn radius(&self) -> usize {
        self.window_size / 2
    }
}

/// Parameters for SNIP baseline estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnipParams {
    /// Number of clipping iterations.
    pub max_half_window: usize,
}

impl Default for SnipParams {
    fn default() -> Self {
        Self {
            max_half_window: 40,
        }
    }
}

impl SnipParams {
    fn validate(&self) -> Result<()> {
        if self.max_half_window == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "max_half_window",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }
}

/// Estimates a baseline using a rolling-ball style opening followed by smoothing.
///
/// # References
///
/// - M. Kneen and H. Annegarn, 1996.
/// - `pybaselines.Baseline.rolling_ball` is used as a behavioral reference.
pub fn rolling_ball(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = rolling_ball_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a rolling-ball baseline into an existing output buffer.
pub fn rolling_ball_into(
    y: &[f64],
    params: MorphologyParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    let opened = opening_reflect(y, params.radius());
    moving_average_extrapolated(&opened, params.radius(), baseline);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a top-hat baseline using morphological opening.
///
/// # References
///
/// - `pybaselines.Baseline.tophat` is used as a behavioral reference.
pub fn tophat(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = tophat_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a top-hat baseline into an existing output buffer.
pub fn tophat_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    let opened = opening_reflect(y, params.radius());
    baseline.copy_from_slice(&opened);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a moving-window minimum-value baseline.
///
/// # References
///
/// - `pybaselines.Baseline.mwmv` is used as a behavioral reference.
pub fn mwmv(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = mwmv_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates an MWMV baseline into an existing output buffer.
pub fn mwmv_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    let mins = moving_min_reflect(y, params.radius());
    moving_average_extrapolated(&mins, params.radius(), baseline);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a morphology baseline from an opening and its averaged envelope.
///
/// # References
///
/// - `pybaselines.Baseline.mor` is used as a behavioral reference.
pub fn mor(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = mor_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a morphology penalized least-squares baseline.
///
/// # References
///
/// - Z. Li et al., "Morphological weighted penalized least squares for
///   background correction", *Analyst*, 2013.
/// - `pybaselines.Baseline.mpls` is used as a behavioral reference.
pub fn mpls(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = mpls_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates an improved morphology baseline.
///
/// # References
///
/// - L. Dai et al., "An Automated Baseline Correction Method Based on
///   Iterative Morphological Operations", *Applied Spectroscopy*, 2018.
/// - `pybaselines.Baseline.imor` is used as a behavioral reference.
pub fn imor(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = imor_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a morphology and mollification baseline.
///
/// # References
///
/// - M. Koch et al., "Iterative morphological and mollifier-based baseline
///   correction for Raman spectra", *Journal of Raman Spectroscopy*, 2017.
/// - `pybaselines.Baseline.mormol` is used as a behavioral reference.
pub fn mormol(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = mormol_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates an averaged morphology and mollification baseline.
///
/// # References
///
/// - `pybaselines.Baseline.amormol` is used as a behavioral reference.
pub fn amormol(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mor_fit = mor(y, params)?;
    let roll_fit = rolling_ball(y, params)?;
    let baseline = mor_fit
        .baseline
        .iter()
        .zip(&roll_fit.baseline)
        .map(|(left, right)| 0.5 * (left + right))
        .collect();
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Estimates a morphology-guided penalized spline baseline.
///
/// # References
///
/// - J. Gonzalez-Vidal et al., "Automatic morphology-based cubic p-spline
///   fitting methodology for smoothing and baseline-removal of Raman spectra",
///   *Journal of Raman Spectroscopy*, 2017.
/// - `pybaselines.Baseline.mpspline` is used as a behavioral reference.
pub fn mpspline(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = mpspline_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a joint baseline correction and denoising baseline.
///
/// # References
///
/// - H. Liu et al., "Joint Baseline-Correction and Denoising for Raman
///   Spectra", *Applied Spectroscopy*, 2015.
/// - `pybaselines.Baseline.jbcd` is used as a behavioral reference.
pub fn jbcd(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = jbcd_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a morphology baseline into an existing output buffer.
pub fn mor_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    let opened = opening_reflect(y, params.radius());
    let averaged = average_opening_from_opened_reflect(&opened, params.radius());
    for ((target, open), average) in baseline.iter_mut().zip(opened).zip(averaged) {
        *target = open.min(average);
    }
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates an MPLS baseline into an existing output buffer.
pub fn mpls_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "mpls",
            len: y.len(),
            min: 3,
        });
    }

    let rough_baseline = opening_reflect(y, params.radius());
    let weights = mpls_anchor_weights(y, &rough_baseline, MPLS_P);
    if !weights.iter().any(|weight| *weight > 0.0) {
        baseline.copy_from_slice(y);
        return Ok(FitReport::new(1, true, 0.0));
    }
    let mut workspace = PentadiagonalWorkspace::new(y.len());
    solve_second_order(y, &weights, MPLS_LAMBDA, baseline, &mut workspace)?;
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates an IMor baseline into an existing output buffer.
pub fn imor_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    baseline.copy_from_slice(y);
    let mut next = vec![0.0; y.len()];
    let mut averaged = vec![0.0; y.len()];
    let mut tolerance = f64::INFINITY;

    for iter in 0..=IMOR_MAX_ITER {
        average_opening_reflect(baseline, params.radius(), &mut averaged);
        for ((target, observed), opened) in next.iter_mut().zip(y).zip(&averaged) {
            *target = observed.min(*opened);
        }
        tolerance = relative_change(baseline, &next);
        if tolerance < IMOR_TOL {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        baseline.copy_from_slice(&next);
    }

    Ok(FitReport::new(IMOR_MAX_ITER + 1, false, tolerance))
}

/// Estimates a MorMol baseline into an existing output buffer.
pub fn mormol_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    let radius = params.radius();
    let full_window = 2 * radius + 1;
    let padded_y = pad_extrapolated(y, full_window);
    let bounds = full_window..full_window + y.len();
    let kernel = mollifier_kernel(full_window);
    let smooth_kernel = mollifier_kernel(1);
    let mut padded_baseline = vec![0.0; padded_y.len()];
    let mut y_minus_baseline = vec![0.0; padded_y.len()];
    let mut next = vec![0.0; padded_y.len()];
    let mut tolerance = f64::INFINITY;

    for iter in 0..=MORMOL_MAX_ITER {
        for ((target, observed), current) in y_minus_baseline
            .iter_mut()
            .zip(&padded_y)
            .zip(&padded_baseline)
        {
            *target = observed - current;
        }
        let y_smooth = convolve_reflect_same(&y_minus_baseline, &smooth_kernel);
        let eroded = moving_min_reflect(&y_smooth, radius);
        let correction = convolve_reflect_same(&eroded, &kernel);
        for ((target, current), update) in next.iter_mut().zip(&padded_baseline).zip(correction) {
            *target = current + update;
        }
        tolerance = relative_change(&padded_baseline[bounds.clone()], &next[bounds.clone()]);
        padded_baseline.copy_from_slice(&next);
        if tolerance < MORMOL_TOL {
            baseline.copy_from_slice(&padded_baseline[bounds]);
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    baseline.copy_from_slice(&padded_baseline[bounds]);
    Ok(FitReport::new(MORMOL_MAX_ITER + 1, false, tolerance))
}

/// Estimates a JBCD baseline into an existing output buffer.
pub fn jbcd_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "jbcd",
            len: y.len(),
            min: 3,
        });
    }

    let opening = opening_reflect(y, params.radius());
    let averaged = average_opening_from_opened_reflect(&opening, params.radius());
    let robust_opening: Vec<f64> = opening
        .into_iter()
        .zip(averaged)
        .map(|(opened, average)| opened.min(average))
        .collect();

    let n = y.len();
    let mut band_workspace = GeneralPentadiagonalWorkspace::new(n);
    let lower2 = vec![0.0; n - 2];
    let upper2 = vec![0.0; n - 2];
    let mut lower1 = vec![0.0; n - 1];
    let mut diag = vec![0.0; n];
    let mut upper1 = vec![0.0; n - 1];
    let mut rhs = vec![0.0; n];
    let mut signal = y.to_vec();
    let mut previous_signal = y.to_vec();
    let mut previous_baseline = robust_opening.clone();
    let mut beta = JBCD_BETA;
    let mut gamma = JBCD_GAMMA;
    let mut signal_tolerance = f64::INFINITY;
    let mut baseline_tolerance = f64::INFINITY;

    for iter in 0..=JBCD_MAX_ITER {
        fill_first_order_system(gamma, 1.0, &mut lower1, &mut diag, &mut upper1);
        for ((target, observed), previous) in rhs.iter_mut().zip(y).zip(&previous_baseline) {
            *target = observed - previous;
        }
        solve_general_pentadiagonal(
            GeneralPentadiagonalSystem {
                lower2: &lower2,
                lower1: &lower1,
                diag: &diag,
                upper1: &upper1,
                upper2: &upper2,
            },
            &rhs,
            &mut signal,
            &mut band_workspace,
        )?;

        fill_first_order_system(
            2.0 * beta,
            1.0 + 2.0 * JBCD_ALPHA,
            &mut lower1,
            &mut diag,
            &mut upper1,
        );
        for (((target, observed), signal_value), opened) in
            rhs.iter_mut().zip(y).zip(&signal).zip(&robust_opening)
        {
            *target = observed - signal_value + 2.0 * JBCD_ALPHA * opened;
        }
        solve_general_pentadiagonal(
            GeneralPentadiagonalSystem {
                lower2: &lower2,
                lower1: &lower1,
                diag: &diag,
                upper1: &upper1,
                upper2: &upper2,
            },
            &rhs,
            baseline,
            &mut band_workspace,
        )?;

        signal_tolerance = relative_change(&previous_signal, &signal);
        baseline_tolerance = relative_change(&previous_baseline, baseline);
        if signal_tolerance < JBCD_SIGNAL_TOL && baseline_tolerance < JBCD_BASELINE_TOL {
            return Ok(FitReport::new(
                iter + 1,
                true,
                signal_tolerance.max(baseline_tolerance),
            ));
        }

        previous_signal.copy_from_slice(&signal);
        previous_baseline.copy_from_slice(baseline);
        gamma *= JBCD_GAMMA_MULT;
        beta *= JBCD_BETA_MULT;
    }

    Ok(FitReport::new(
        JBCD_MAX_ITER + 1,
        false,
        signal_tolerance.max(baseline_tolerance),
    ))
}

/// Estimates an MPSpline baseline into an existing output buffer.
pub fn mpspline_into(
    y: &[f64],
    params: MorphologyParams,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    if y.len() < MPSPLINE_DEGREE + 2 {
        return Err(BaselineError::TooShort {
            algorithm: "mpspline",
            len: y.len(),
            min: MPSPLINE_DEGREE + 2,
        });
    }

    let num_knots = MPSPLINE_NUM_KNOTS.min(y.len()).max(2);
    let pspline = PenalizedSpline::new(y.len(), num_knots, MPSPLINE_DEGREE, MPSPLINE_DIFF_ORDER);

    let closed = closing_reflect(y, 1);
    let smooth_weights: Vec<f64> = y
        .iter()
        .zip(&closed)
        .map(|(observed, closed)| if observed == closed { 1.0 } else { 0.0 })
        .collect();
    let spline_fit = pspline.solve(y, &smooth_weights, MPSPLINE_SMOOTH_LAMBDA)?;

    let radius = params.radius();
    let full_window = 2 * radius + 1;
    let padded = pad_extrapolated(&spline_fit, full_window);
    let opened = opening_reflect(&padded, radius);
    let averaged = average_opening_from_opened_reflect(&opened, radius);
    let optimal_opening: Vec<f64> = opened[full_window..full_window + y.len()]
        .iter()
        .zip(&averaged[full_window..full_window + y.len()])
        .map(|(opened, average)| opened.min(*average))
        .collect();

    let weights: Vec<f64> = spline_fit
        .iter()
        .zip(&optimal_opening)
        .map(|(fit, opening)| {
            if (*fit - *opening).abs() <= 1.0e-12 {
                1.0 - MPSPLINE_P
            } else {
                MPSPLINE_P
            }
        })
        .collect();
    if !weights.iter().any(|weight| *weight > 0.0) {
        baseline.copy_from_slice(&spline_fit);
        return Ok(FitReport::new(1, true, 0.0));
    }

    let fitted = pspline.solve(&spline_fit, &weights, MPSPLINE_LAMBDA)?;
    baseline.copy_from_slice(&fitted);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a baseline with the statistics-sensitive nonlinear iterative peak-clipping algorithm.
///
/// # References
///
/// - C. G. Ryan et al., 1988.
/// - `pybaselines.Baseline.snip` is used as a behavioral reference.
pub fn snip(y: &[f64], params: SnipParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = snip_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Estimates a SNIP baseline into an existing output buffer.
pub fn snip_into(y: &[f64], params: SnipParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()?;
    baseline.copy_from_slice(y);
    let max_half_window = params.max_half_window.min(y.len().saturating_sub(1) / 2);

    for half_window in (1..=max_half_window).rev() {
        for i in half_window..y.len() - half_window {
            let average = 0.5 * (baseline[i - half_window] + baseline[i + half_window]);
            if average < baseline[i] {
                baseline[i] = average;
            }
        }
    }

    Ok(FitReport::new(max_half_window, true, 0.0))
}

fn validate_morphology_input(y: &[f64], params: MorphologyParams, baseline: &[f64]) -> Result<()> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()
}

fn moving_average_extrapolated(y: &[f64], radius: usize, output: &mut [f64]) {
    if radius == 0 {
        output.copy_from_slice(y);
        return;
    }
    let padded = pad_extrapolated(y, radius);
    let window = 2 * radius + 1;
    for (i, target) in output.iter_mut().enumerate() {
        let sum = padded[i..i + window].iter().sum::<f64>();
        *target = sum / window as f64;
    }
}

fn pad_extrapolated(y: &[f64], radius: usize) -> Vec<f64> {
    let fit_len = radius.min(y.len()).max(1);
    let mut padded = Vec::with_capacity(y.len() + 2 * radius);
    let left = linear_extrapolation(
        (0..fit_len).map(|index| ((radius + index) as f64, y[index])),
        0..radius,
    );
    padded.extend(left);
    padded.extend_from_slice(y);
    let right_start = radius + y.len();
    let fit_start = y.len().saturating_sub(fit_len);
    let right = linear_extrapolation(
        (fit_start..y.len()).map(|index| ((radius + index) as f64, y[index])),
        right_start..right_start + radius,
    );
    padded.extend(right);
    padded
}

fn linear_extrapolation(
    points: impl Iterator<Item = (f64, f64)>,
    output_range: std::ops::Range<usize>,
) -> Vec<f64> {
    let points: Vec<(f64, f64)> = points.collect();
    if points.len() == 1 {
        return vec![points[0].1; output_range.len()];
    }

    let count = points.len() as f64;
    let sum_x = points.iter().map(|(x, _)| x).sum::<f64>();
    let sum_y = points.iter().map(|(_, y)| y).sum::<f64>();
    let sum_xx = points.iter().map(|(x, _)| x * x).sum::<f64>();
    let sum_xy = points.iter().map(|(x, y)| x * y).sum::<f64>();
    let denominator = count * sum_xx - sum_x * sum_x;
    let slope = if denominator.abs() <= f64::EPSILON {
        0.0
    } else {
        (count * sum_xy - sum_x * sum_y) / denominator
    };
    let intercept = (sum_y - slope * sum_x) / count;

    output_range
        .map(|index| slope.mul_add(index as f64, intercept))
        .collect()
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

fn fill_first_order_system(
    penalty: f64,
    diagonal_offset: f64,
    lower1: &mut [f64],
    diag: &mut [f64],
    upper1: &mut [f64],
) {
    let n = diag.len();
    for (index, value) in diag.iter_mut().enumerate() {
        let penalty_diag = if index == 0 || index + 1 == n {
            penalty
        } else {
            2.0 * penalty
        };
        *value = diagonal_offset + penalty_diag;
    }
    lower1.fill(-penalty);
    upper1.fill(-penalty);
}

fn average_opening_reflect(y: &[f64], radius: usize, output: &mut [f64]) {
    let opened = opening_reflect(y, radius);
    let averaged = average_opening_from_opened_reflect(&opened, radius);
    output.copy_from_slice(&averaged);
}

fn average_opening_from_opened_reflect(opened: &[f64], radius: usize) -> Vec<f64> {
    let mut dilated = vec![0.0; opened.len()];
    moving_max_reflect(opened, radius, &mut dilated);
    let eroded = moving_min_reflect(opened, radius);
    dilated
        .into_iter()
        .zip(eroded)
        .map(|(dilation, erosion)| 0.5 * (dilation + erosion))
        .collect()
}

fn opening_reflect(y: &[f64], radius: usize) -> Vec<f64> {
    let eroded = moving_min_reflect(y, radius);
    let mut opened = vec![0.0; y.len()];
    moving_max_reflect(&eroded, radius, &mut opened);
    opened
}

fn closing_reflect(y: &[f64], radius: usize) -> Vec<f64> {
    let mut dilated = vec![0.0; y.len()];
    moving_max_reflect(y, radius, &mut dilated);
    moving_min_reflect(&dilated, radius)
}

fn moving_min_reflect(y: &[f64], radius: usize) -> Vec<f64> {
    let mut output = vec![0.0; y.len()];
    for (i, value) in output.iter_mut().enumerate() {
        let start = i as isize - radius as isize;
        let end = i as isize + radius as isize;
        *value = (start..=end)
            .map(|index| y[reflect_index(index, y.len())])
            .fold(f64::INFINITY, f64::min);
    }
    output
}

fn moving_max_reflect(y: &[f64], radius: usize, output: &mut [f64]) {
    for (i, value) in output.iter_mut().enumerate() {
        let start = i as isize - radius as isize;
        let end = i as isize + radius as isize;
        *value = (start..=end)
            .map(|index| y[reflect_index(index, y.len())])
            .fold(f64::NEG_INFINITY, f64::max);
    }
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    if len == 1 {
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

fn relative_change(previous: &[f64], current: &[f64]) -> f64 {
    let numerator = previous
        .iter()
        .zip(current)
        .map(|(old, new)| {
            let difference = new - old;
            difference * difference
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

fn mpls_anchor_weights(y: &[f64], rough_baseline: &[f64], p: f64) -> Vec<f64> {
    let mut diff = Vec::with_capacity(rough_baseline.len() + 1);
    diff.push(0.0);
    diff.extend(rough_baseline.windows(2).map(|pair| pair[1] - pair[0]));
    diff.push(0.0);

    let indices: Vec<usize> = (0..rough_baseline.len())
        .filter(|&index| {
            let left_flat = diff[index] == 0.0;
            let right_flat = diff[index + 1] == 0.0;
            let left_changes = diff[index] != 0.0;
            let right_changes = diff[index + 1] != 0.0;
            (right_flat || left_flat) && (right_changes || left_changes)
        })
        .collect();

    let mut weights = vec![p; y.len()];
    for (&previous_segment, &next_segment) in indices
        .iter()
        .skip(1)
        .step_by(2)
        .zip(indices.iter().skip(2).step_by(2))
    {
        let region = &y[previous_segment..=next_segment];
        if let Some((offset, _)) = region
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
        {
            weights[previous_segment + offset] = 1.0 - p;
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morphology_preserves_constant_signal() {
        let y = vec![2.0; 51];
        for fit in [
            rolling_ball(&y, MorphologyParams::default()).unwrap(),
            tophat(&y, MorphologyParams::default()).unwrap(),
            mwmv(&y, MorphologyParams::default()).unwrap(),
            mor(&y, MorphologyParams::default()).unwrap(),
            mpls(&y, MorphologyParams::default()).unwrap(),
            snip(&y, SnipParams::default()).unwrap(),
        ] {
            for value in fit.baseline {
                assert!((value - 2.0).abs() < 1.0e-12);
            }
        }
    }
}
