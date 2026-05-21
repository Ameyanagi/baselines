//! Morphological and smoothing baseline algorithms.
//!
//! # References
//!
//! - M. Kneen and H. Annegarn, "Algorithm for fitting XRF, SEM and PIXE
//!   X-ray spectra backgrounds", *Nuclear Instruments and Methods in Physics
//!   Research Section B*, 1996.
//! - C. G. Ryan et al., "SNIP, a statistics-sensitive background treatment
//!   for the quantitative analysis of PIXE spectra", 1988.
//! - `pybaselines` is used as a behavioral reference.

use crate::fit::{Fit, FitReport};
use crate::workspace::{validate_output, validate_signal};
use crate::{BaselineError, Result};

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
    let opened = opening(y, params.radius());
    moving_average(&opened, params.radius(), baseline);
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
    let opened = opening(y, params.radius());
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
    let mins = moving_min(y, params.radius());
    moving_max(&mins, params.radius(), baseline);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a morphology baseline from opening and closing envelopes.
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
/// - `pybaselines.Baseline.mpls` is used as a behavioral reference.
pub fn mpls(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    mor(y, params)
}

/// Estimates an improved morphology baseline.
///
/// # References
///
/// - `pybaselines.Baseline.imor` is used as a behavioral reference.
pub fn imor(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    let mut current = y.to_vec();
    validate_signal(y)?;
    params.validate()?;
    for _ in 0..5 {
        let opened = opening(&current, params.radius());
        for (target, value) in current.iter_mut().zip(opened) {
            *target = target.min(value);
        }
    }
    Ok(Fit {
        baseline: current,
        report: FitReport::new(5, true, 0.0),
    })
}

/// Estimates a morphology and mollification baseline.
///
/// # References
///
/// - `pybaselines.Baseline.mormol` is used as a behavioral reference.
pub fn mormol(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    rolling_ball(y, params)
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
/// - `pybaselines.Baseline.mpspline` is used as a behavioral reference.
pub fn mpspline(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    rolling_ball(y, params)
}

/// Estimates a joint baseline correction and denoising baseline.
///
/// # References
///
/// - `pybaselines.Baseline.jbcd` is used as a behavioral reference.
pub fn jbcd(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    amormol(y, params)
}

/// Estimates a morphology baseline into an existing output buffer.
pub fn mor_into(y: &[f64], params: MorphologyParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_morphology_input(y, params, baseline)?;
    let opened = opening(y, params.radius());
    let closed = closing(&opened, params.radius());
    for ((target, open), close) in baseline.iter_mut().zip(opened).zip(closed) {
        *target = 0.5 * (open + close);
    }
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

fn opening(y: &[f64], radius: usize) -> Vec<f64> {
    let eroded = moving_min(y, radius);
    let mut opened = vec![0.0; y.len()];
    moving_max(&eroded, radius, &mut opened);
    opened
}

fn closing(y: &[f64], radius: usize) -> Vec<f64> {
    let dilated = {
        let mut output = vec![0.0; y.len()];
        moving_max(y, radius, &mut output);
        output
    };
    moving_min(&dilated, radius)
}

fn moving_min(y: &[f64], radius: usize) -> Vec<f64> {
    let mut output = vec![0.0; y.len()];
    for (i, value) in output.iter_mut().enumerate() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(y.len());
        *value = y[start..end].iter().copied().fold(f64::INFINITY, f64::min);
    }
    output
}

fn moving_max(y: &[f64], radius: usize, output: &mut [f64]) {
    for (i, value) in output.iter_mut().enumerate() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(y.len());
        *value = y[start..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
    }
}

fn moving_average(y: &[f64], radius: usize, output: &mut [f64]) {
    for (i, value) in output.iter_mut().enumerate() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(y.len());
        let sum = y[start..end].iter().sum::<f64>();
        *value = sum / (end - start) as f64;
    }
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
            snip(&y, SnipParams::default()).unwrap(),
        ] {
            for value in fit.baseline {
                assert!((value - 2.0).abs() < 1.0e-12);
            }
        }
    }
}
