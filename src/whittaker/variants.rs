//! Additional Whittaker-family algorithm entry points.

use crate::fit::Fit;
use crate::whittaker::engine::{Reweighter, WhittakerParams, fit_alloc, relative_change};
use crate::whittaker::{ArPlsParams, AslsParams, arpls, asls};
use crate::{BaselineError, Result};

/// Parameters for improved asymmetric least squares.
pub type IaslsParams = AslsParams;
/// Parameters for doubly reweighted penalized least squares.
pub type DrPlsParams = ArPlsParams;
/// Parameters for improved asymmetrically reweighted penalized least squares.
pub type IarPlsParams = ArPlsParams;
/// Parameters for adaptive smoothness penalized least squares.
pub type AsPlsParams = ArPlsParams;
/// Parameters for derivative peaked signal asymmetric least squares.
pub type DerPsalsaParams = AslsParams;
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

/// Fits an IAsLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.iasls` is used as a behavioral reference.
pub fn iasls(y: &[f64], params: IaslsParams) -> Result<Fit> {
    asls(y, params)
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
/// - `pybaselines.Baseline.iarpls` is used as a behavioral reference.
pub fn iarpls(y: &[f64], params: IarPlsParams) -> Result<Fit> {
    arpls(y, params)
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
/// - `pybaselines.Baseline.derpsalsa` is used as a behavioral reference.
pub fn derpsalsa(y: &[f64], params: DerPsalsaParams) -> Result<Fit> {
    asls(y, params)
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
/// - `pybaselines.Baseline.lsrpls` is used as a behavioral reference.
pub fn lsrpls(y: &[f64], params: LsrPlsParams) -> Result<Fit> {
    arpls(y, params)
}

#[derive(Debug, Clone, Copy)]
struct PsalsaWeights {
    p: f64,
    k: f64,
}

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
