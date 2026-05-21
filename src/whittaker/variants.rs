//! Additional Whittaker-family algorithm entry points.

use crate::fit::Fit;
use crate::linalg::pentadiagonal::solve_second_order_with_first_order;
use crate::polynomial::fit_weighted_polynomial;
use crate::whittaker::engine::{Reweighter, WhittakerParams, fit_alloc, relative_change};
use crate::whittaker::{ArPlsParams, arpls};
use crate::workspace::{validate_output, validate_signal};
use crate::{BaselineError, FitReport, Result};

/// Parameters for improved asymmetric least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IaslsParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Smoothness penalty for the first derivative of the residual.
    pub lambda_1: f64,
}

impl Default for IaslsParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams::default(),
            p: 0.01,
            lambda_1: 1.0e-4,
        }
    }
}

impl IaslsParams {
    /// Validates IAsLS parameters.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        if !self.lambda_1.is_finite() || self.lambda_1 <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "lambda_1",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for doubly reweighted penalized least squares.
pub type DrPlsParams = ArPlsParams;
/// Parameters for improved asymmetrically reweighted penalized least squares.
pub type IarPlsParams = ArPlsParams;
/// Parameters for adaptive smoothness penalized least squares.
pub type AsPlsParams = ArPlsParams;
/// Parameters for Bayesian reweighted penalized least squares.
pub type BrPlsParams = ArPlsParams;
/// Parameters for locally symmetric reweighted penalized least squares.
pub type LsrPlsParams = ArPlsParams;

/// Parameters for peaked signal asymmetric least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PsalsaParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Exponential decay scale. If `None`, uses `std(y) / 10`.
    pub k: Option<f64>,
}

impl Default for PsalsaParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                max_iter: 50,
                tol: 1.0e-3,
            },
            p: 0.5,
            k: None,
        }
    }
}

impl PsalsaParams {
    /// Validates psalsa parameters that do not depend on the input signal.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        if let Some(k) = self.k
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(BaselineError::InvalidParameter {
                name: "k",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Parameters for derivative peak-screening asymmetric least squares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerPsalsaParams {
    /// Shared Whittaker parameters.
    pub whittaker: WhittakerParams,
    /// Asymmetry parameter in `(0, 1)`.
    pub p: f64,
    /// Peak-height scale. If `None`, uses `std(y) / 10`.
    pub k: Option<f64>,
    /// Optional half-window for derivative smoothing. If `None`, uses
    /// `len(y) / 200`.
    pub smooth_half_window: Option<usize>,
    /// Number of mollifier smoothing passes before computing derivatives.
    pub num_smooths: usize,
}

impl Default for DerPsalsaParams {
    fn default() -> Self {
        Self {
            whittaker: WhittakerParams::default(),
            p: 0.01,
            k: None,
            smooth_half_window: None,
            num_smooths: 16,
        }
    }
}

impl DerPsalsaParams {
    /// Validates derpsalsa parameters that do not depend on the input signal.
    pub fn validate(&self) -> Result<()> {
        self.whittaker.validate()?;
        if !self.p.is_finite() || self.p <= 0.0 || self.p >= 1.0 {
            return Err(BaselineError::InvalidParameter {
                name: "p",
                reason: "must be finite and between 0 and 1",
            });
        }
        if let Some(k) = self.k
            && (!k.is_finite() || k <= 0.0)
        {
            return Err(BaselineError::InvalidParameter {
                name: "k",
                reason: "must be finite and positive",
            });
        }
        Ok(())
    }
}

/// Fits an IAsLS baseline.
///
/// # References
///
/// - S. He et al., "Baseline correction for Raman spectra using an improved
///   asymmetric least squares method", *Analytical Methods*, 2014.
/// - `pybaselines.Baseline.iasls` is used as a behavioral reference.
pub fn iasls(y: &[f64], params: IaslsParams) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = iasls_into(y, params, &mut baseline)?;
    Ok(Fit { baseline, report })
}

/// Fits an IAsLS baseline into an existing output buffer.
pub fn iasls_into(y: &[f64], params: IaslsParams, baseline: &mut [f64]) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "iasls",
            len: y.len(),
            min: 3,
        });
    }
    params.validate()?;

    let n = y.len();
    let mut workspace = crate::whittaker::WhittakerWorkspace::new(n);
    workspace.iter.weights.fill(1.0);
    fit_weighted_polynomial(y, &workspace.iter.weights, 2, &mut workspace.iter.residual)?;
    for ((weight, observed), fitted) in workspace
        .iter
        .weights
        .iter_mut()
        .zip(y)
        .zip(&workspace.iter.residual)
    {
        *weight = asls_weight(*observed, *fitted, params.p);
    }
    let mut first_order_rhs = vec![0.0; n];
    first_order_penalty_rhs(y, params.lambda_1, &mut first_order_rhs);

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.whittaker.max_iter {
        workspace
            .iter
            .previous_weights
            .copy_from_slice(&workspace.iter.weights);

        for (((diagonal, rhs), weight), (observed, first_order_rhs)) in workspace
            .iter
            .residual
            .iter_mut()
            .zip(workspace.iter.rhs.iter_mut())
            .zip(&workspace.iter.weights)
            .zip(y.iter().zip(&first_order_rhs))
        {
            let weight_squared = weight * weight;
            *diagonal = weight_squared;
            *rhs = weight_squared * observed + first_order_rhs;
        }

        solve_second_order_with_first_order(
            &workspace.iter.residual,
            &workspace.iter.rhs,
            params.whittaker.lambda,
            params.lambda_1,
            baseline,
            &mut workspace.solver,
        )?;

        for ((weight, observed), fitted) in workspace
            .iter
            .weights
            .iter_mut()
            .zip(y)
            .zip(baseline.iter())
        {
            *weight = asls_weight(*observed, *fitted, params.p);
        }
        tolerance = relative_change(&workspace.iter.previous_weights, &workspace.iter.weights);
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.whittaker.max_iter, false, tolerance))
}

/// Fits a drPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.drpls` is used as a behavioral reference.
pub fn drpls(y: &[f64], params: DrPlsParams) -> Result<Fit> {
    arpls(y, params)
}

/// Fits an IarPLS baseline.
///
/// # References
///
/// - J. Ye et al., "Baseline correction method based on improved
///   asymmetrically reweighted penalized least squares for Raman spectrum",
///   *Applied Optics*, 2020.
/// - `pybaselines.Baseline.iarpls` is used as a behavioral reference.
pub fn iarpls(y: &[f64], params: IarPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    fit_alloc(y, params.whittaker, IarPlsWeights)
}

/// Fits an asPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.aspls` is used as a behavioral reference.
pub fn aspls(y: &[f64], params: AsPlsParams) -> Result<Fit> {
    arpls(y, params)
}

/// Fits a psalsa baseline.
///
/// # References
///
/// - `pybaselines.Baseline.psalsa` is used as a behavioral reference.
pub fn psalsa(y: &[f64], params: PsalsaParams) -> Result<Fit> {
    params.validate()?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }
    fit_alloc(y, params.whittaker, PsalsaWeights { p: params.p, k })
}

/// Fits a derpsalsa baseline.
///
/// # References
///
/// - V. Korepanov, "Asymmetric least-squares baseline algorithm with peak
///   screening for automatic processing of the Raman spectra", *Journal of
///   Raman Spectroscopy*, 2020.
/// - `pybaselines.Baseline.derpsalsa` is used as a behavioral reference.
pub fn derpsalsa(y: &[f64], params: DerPsalsaParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }
    let partial_weights = derivative_peak_screening_weights(
        y,
        params.smooth_half_window.unwrap_or(y.len() / 200),
        params.num_smooths,
    );
    fit_alloc(
        y,
        params.whittaker,
        DerPsalsaWeights {
            p: params.p,
            k,
            partial_weights,
        },
    )
}

/// Fits a brPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.brpls` is used as a behavioral reference.
pub fn brpls(y: &[f64], params: BrPlsParams) -> Result<Fit> {
    arpls(y, params)
}

/// Fits an lsrPLS baseline.
///
/// # References
///
/// - Z. Heng et al., "Baseline correction for Raman spectra based on locally
///   symmetric reweighted penalized least squares", *Chinese Journal of
///   Lasers*, 2018.
/// - `pybaselines.Baseline.lsrpls` is used as a behavioral reference.
pub fn lsrpls(y: &[f64], params: LsrPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    fit_alloc(y, params.whittaker, LsrPlsWeights)
}

#[derive(Debug, Clone, Copy)]
struct PsalsaWeights {
    p: f64,
    k: f64,
}

#[derive(Debug, Clone)]
struct DerPsalsaWeights {
    p: f64,
    k: f64,
    partial_weights: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct IarPlsWeights;

#[derive(Debug, Clone, Copy)]
struct LsrPlsWeights;

impl Reweighter for PsalsaWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], _iter: usize) -> f64 {
        let previous = weights.to_vec();
        for ((weight, observed), fitted) in weights.iter_mut().zip(y).zip(baseline) {
            let residual = observed - fitted;
            *weight = if residual > 0.0 {
                self.p * (-residual / self.k).exp()
            } else {
                1.0 - self.p
            };
        }
        relative_change(&previous, weights)
    }
}

impl Reweighter for DerPsalsaWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], _iter: usize) -> f64 {
        let previous = weights.to_vec();
        for (((weight, observed), fitted), partial) in weights
            .iter_mut()
            .zip(y)
            .zip(baseline)
            .zip(&self.partial_weights)
        {
            let residual = observed - fitted;
            let asymmetric = if residual > 0.0 {
                self.p * (-0.5 * (residual / self.k).powi(2)).exp()
            } else {
                1.0 - self.p
            };
            *weight = asymmetric * partial;
        }
        relative_change(&previous, weights)
    }
}

impl Reweighter for IarPlsWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], iter: usize) -> f64 {
        let previous = weights.to_vec();
        let Some((_mean, std)) = negative_residual_stats(y, baseline) else {
            return 0.0;
        };
        let factor = (iter + 1).min(100) as f64;
        let scale = factor.exp() / std.max(f64::EPSILON);

        for ((weight, observed), fitted) in weights.iter_mut().zip(y).zip(baseline) {
            let residual = observed - fitted;
            let inner = scale * (residual - 2.0 * std);
            *weight = 0.5 * (1.0 - inner / (1.0 + inner * inner).sqrt());
        }

        relative_change(&previous, weights)
    }
}

impl Reweighter for LsrPlsWeights {
    fn initialize(&self, _y: &[f64], weights: &mut [f64]) {
        weights.fill(1.0);
    }

    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], iter: usize) -> f64 {
        let previous = weights.to_vec();
        let Some((mean, std)) = negative_residual_stats(y, baseline) else {
            return 0.0;
        };
        let scale = 10f64.powi((iter + 1).min(100) as i32) / std.max(f64::EPSILON);

        for ((weight, observed), fitted) in weights.iter_mut().zip(y).zip(baseline) {
            let residual = observed - fitted;
            let inner = scale * (residual - (2.0 * std - mean));
            *weight = 0.5 * (1.0 - inner / (1.0 + inner.abs()));
        }

        relative_change(&previous, weights)
    }
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

fn negative_residual_stats(y: &[f64], baseline: &[f64]) -> Option<(f64, f64)> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for (observed, fitted) in y.iter().zip(baseline) {
        let residual = observed - fitted;
        if residual < 0.0 {
            count += 1;
            sum += residual;
        }
    }
    if count < 2 {
        return None;
    }

    let mean = sum / count as f64;
    let mut sum_squares = 0.0;
    for (observed, fitted) in y.iter().zip(baseline) {
        let residual = observed - fitted;
        if residual < 0.0 {
            let centered = residual - mean;
            sum_squares += centered * centered;
        }
    }
    let std = (sum_squares / (count - 1) as f64).sqrt();
    Some((mean, std))
}

fn asls_weight(observed: f64, fitted: f64, p: f64) -> f64 {
    if observed > fitted { p } else { 1.0 - p }
}

fn first_order_penalty_rhs(y: &[f64], lambda_1: f64, output: &mut [f64]) {
    output[0] = lambda_1 * (y[0] - y[1]);
    for i in 1..y.len() - 1 {
        output[i] = lambda_1 * (2.0 * y[i] - y[i - 1] - y[i + 1]);
    }
    let last = y.len() - 1;
    output[last] = lambda_1 * (y[last] - y[last - 1]);
}

fn derivative_peak_screening_weights(
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
