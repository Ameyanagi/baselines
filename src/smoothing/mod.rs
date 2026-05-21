//! Smoothing baseline algorithms.
//!
//! These algorithms are implemented with conservative CPU `f64` routines and
//! are intended to converge toward `pybaselines.Baseline` behavior through
//! golden fixture tests.

use crate::fit::{Fit, FitReport};
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
    moving_median(y, params.window_size / 2, baseline);
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
/// - `pybaselines.Baseline.swima` is used as a behavioral reference.
pub fn swima(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    iterative_smoother(y, params, BaselineLimiter::Minimum)
}

/// Estimates a baseline with iterative polynomial-style averaging.
///
/// # References
///
/// - `pybaselines.Baseline.ipsa` is used as a behavioral reference.
pub fn ipsa(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    iterative_smoother(y, params, BaselineLimiter::Observed)
}

/// Estimates a baseline with range-independent averaging.
///
/// # References
///
/// - `pybaselines.Baseline.ria` is used as a behavioral reference.
pub fn ria(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    iterative_smoother(y, params, BaselineLimiter::Smoothed)
}

/// Estimates a baseline by iteratively filling peaks from neighboring values.
///
/// # References
///
/// - `pybaselines.Baseline.peak_filling` is used as a behavioral reference.
pub fn peak_filling(y: &[f64], params: SmoothingParams) -> Result<Fit> {
    let mut baseline = y.to_vec();
    let mut next = vec![0.0; y.len()];
    validate_signal(y)?;
    params.validate()?;
    for _ in 0..params.max_iter {
        next.copy_from_slice(&baseline);
        for i in 1..y.len().saturating_sub(1) {
            next[i] = baseline[i].min(0.5 * (baseline[i - 1] + baseline[i + 1]));
        }
        baseline.copy_from_slice(&next);
    }
    Ok(Fit {
        baseline,
        report: FitReport::new(params.max_iter, true, 0.0),
    })
}

enum BaselineLimiter {
    Minimum,
    Observed,
    Smoothed,
}

fn iterative_smoother(y: &[f64], params: SmoothingParams, limiter: BaselineLimiter) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let mut baseline = y.to_vec();
    let mut smoothed = vec![0.0; y.len()];
    for _ in 0..params.max_iter {
        moving_average(&baseline, params.window_size / 2, &mut smoothed);
        for ((target, observed), smooth) in baseline.iter_mut().zip(y).zip(&smoothed) {
            *target = match limiter {
                BaselineLimiter::Minimum => target.min(*smooth),
                BaselineLimiter::Observed => observed.min(*smooth),
                BaselineLimiter::Smoothed => *smooth,
            };
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

fn moving_median(y: &[f64], radius: usize, output: &mut [f64]) {
    let mut window = Vec::with_capacity(2 * radius + 1);
    for (i, target) in output.iter_mut().enumerate() {
        window.clear();
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(y.len());
        window.extend_from_slice(&y[start..end]);
        window.sort_by(f64::total_cmp);
        *target = window[window.len() / 2];
    }
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
}
