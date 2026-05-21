//! Mixture-model spline baseline fitting.

use crate::fit::{Fit, FitReport};
use crate::linalg::pspline::PenalizedSpline;
use crate::{BaselineError, Result};

use super::{PSPLINE_DEGREE, PSPLINE_NUM_KNOTS, relative_change, validate_spline_signal};

const MIXTURE_DIFF_ORDER: usize = 3;
const MIN_PDF: f64 = f64::MIN_POSITIVE;

/// Parameters for mixture-model spline fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixtureModelParams {
    /// P-spline smoothing parameter.
    pub lambda: f64,
    /// Initial asymmetric least-squares weight for points above the baseline.
    pub p: f64,
    /// Maximum number of expectation-maximization iterations.
    pub max_iter: usize,
    /// Relative posterior-weight change tolerance.
    pub tol: f64,
    /// Whether to include both positive and negative non-noise residual models.
    pub symmetric: bool,
}

impl Default for MixtureModelParams {
    fn default() -> Self {
        Self {
            lambda: 1.0e5,
            p: 1.0e-2,
            max_iter: 50,
            tol: 1.0e-3,
            symmetric: false,
        }
    }
}

impl MixtureModelParams {
    fn validate(&self) -> Result<()> {
        if !self.lambda.is_finite() || self.lambda <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "lambda",
                reason: "must be finite and positive",
            });
        }
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
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

/// Fits a mixture-model spline baseline.
///
/// # References
///
/// - J. de Rooi et al., "Mixture models for baseline estimation",
///   *Chemometrics and Intelligent Laboratory Systems*, 2012.
/// - B. Ghojogh et al., "Fitting A Mixture Distribution to Data: Tutorial",
///   arXiv, 2019.
/// - `pybaselines.Baseline.mixture_model` is used as a behavioral reference.
pub fn mixture_model(y: &[f64], params: MixtureModelParams) -> Result<Fit> {
    params.validate()?;
    validate_spline_signal("mixture_model", y)?;

    let (scaled_y, domain) = scale_to_unit_range(y);
    let pspline = PenalizedSpline::new(
        y.len(),
        PSPLINE_NUM_KNOTS.min(y.len()).max(2),
        PSPLINE_DEGREE,
        MIXTURE_DIFF_ORDER,
    );
    let mut weights = vec![1.0; y.len()];
    let mut baseline = Vec::new();

    for _ in 0..2 {
        baseline = pspline.solve(&scaled_y, &weights, params.lambda)?;
        for ((weight, observed), fitted) in weights.iter_mut().zip(&scaled_y).zip(&baseline) {
            *weight = if observed > fitted {
                params.p
            } else {
                1.0 - params.p
            };
        }
    }

    let mut residual = residuals(&scaled_y, &baseline);
    let mut sigma = (0.2 * standard_deviation(&residual)).max(f64::MIN_POSITIVE);
    let mut fraction_noise = 0.5;
    let mut fraction_positive = if params.symmetric { 0.25 } else { 0.5 };
    let mut tolerance = f64::INFINITY;

    for iter in 0..=params.max_iter {
        let max_positive = residual
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .abs()
            .max(1.0e-6);
        let min_negative = residual
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .abs()
            .max(1.0e-6);
        let positive_scale = fraction_positive / max_positive;
        let negative_scale = (1.0 - fraction_noise - fraction_positive) / min_negative;

        let mut posterior_noise = vec![0.0; y.len()];
        let mut positive_pdf = vec![0.0; y.len()];
        let mut negative_pdf = vec![0.0; y.len()];
        let mut total_pdf = vec![0.0; y.len()];
        for index in 0..y.len() {
            let value = residual[index];
            if value >= 0.0 {
                positive_pdf[index] = positive_scale;
            } else if params.symmetric {
                negative_pdf[index] = negative_scale;
            }
            let noise_pdf = fraction_noise * gaussian(value, sigma);
            total_pdf[index] = noise_pdf + positive_pdf[index] + negative_pdf[index];
            posterior_noise[index] = noise_pdf / total_pdf[index].max(MIN_PDF);
        }

        tolerance = relative_change(&weights, &posterior_noise);
        if tolerance < params.tol {
            return Ok(Fit {
                baseline: scale_from_unit_range(&baseline, domain),
                report: FitReport::new(iter + 1, true, tolerance),
            });
        }

        let noise_sum = posterior_noise.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        sigma = (posterior_noise
            .iter()
            .zip(&residual)
            .map(|(posterior, residual)| posterior * residual * residual)
            .sum::<f64>()
            / noise_sum)
            .sqrt()
            .max(f64::MIN_POSITIVE);
        if params.symmetric {
            let positive_sum: f64 = positive_pdf
                .iter()
                .zip(&total_pdf)
                .map(|(pdf, total)| pdf / total.max(MIN_PDF))
                .sum();
            let negative_sum: f64 = negative_pdf
                .iter()
                .zip(&total_pdf)
                .map(|(pdf, total)| pdf / total.max(MIN_PDF))
                .sum();
            let total_sum = noise_sum + positive_sum + negative_sum;
            fraction_noise = noise_sum / total_sum;
            fraction_positive = positive_sum / total_sum;
        } else {
            fraction_noise = noise_sum / y.len() as f64;
            fraction_positive = 1.0 - fraction_noise;
        }

        weights = posterior_noise;
        baseline = pspline.solve(&scaled_y, &weights, params.lambda)?;
        residual = residuals(&scaled_y, &baseline);
    }

    Ok(Fit {
        baseline: scale_from_unit_range(&baseline, domain),
        report: FitReport::new(params.max_iter + 1, false, tolerance),
    })
}

fn scale_to_unit_range(y: &[f64]) -> (Vec<f64>, (f64, f64)) {
    let min = y.iter().copied().fold(f64::INFINITY, f64::min);
    let max = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = max - min;
    if span <= f64::EPSILON {
        (vec![0.0; y.len()], (min, max))
    } else {
        (
            y.iter()
                .map(|value| 2.0 * (value - min) / span - 1.0)
                .collect(),
            (min, max),
        )
    }
}

fn scale_from_unit_range(values: &[f64], domain: (f64, f64)) -> Vec<f64> {
    let (min, max) = domain;
    let span = max - min;
    if span <= f64::EPSILON {
        vec![min; values.len()]
    } else {
        values
            .iter()
            .map(|value| 0.5 * (value + 1.0) * span + min)
            .collect()
    }
}

fn residuals(y: &[f64], baseline: &[f64]) -> Vec<f64> {
    y.iter().zip(baseline).map(|(y, z)| y - z).collect()
}

fn gaussian(value: f64, sigma: f64) -> f64 {
    let height = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    height * (-0.5 * (value / sigma).powi(2)).exp()
}

fn standard_deviation(values: &[f64]) -> f64 {
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
