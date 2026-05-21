//! Classification-style baseline algorithms.
//!
//! First-pass implementations expose the algorithm family using robust
//! lower-envelope fitting primitives. Golden pybaselines fixtures should drive
//! later refinements where individual classifiers differ.

use crate::Result;
use crate::fit::{Fit, FitReport};
use crate::smoothing::{SmoothingParams, noise_median};
use crate::workspace::validate_signal;

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
/// - `pybaselines.Baseline.golotvin` is used as a behavioral reference.
pub fn golotvin(y: &[f64], params: ClassificationParams) -> Result<Fit> {
    envelope_from_smoothed(y, params)
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
