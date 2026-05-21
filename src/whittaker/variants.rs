//! Additional Whittaker-family algorithm entry points.

use crate::Result;
use crate::fit::Fit;
use crate::whittaker::{ArPlsParams, AslsParams, arpls, asls};

/// Parameters for improved asymmetric least squares.
pub type IaslsParams = AslsParams;
/// Parameters for doubly reweighted penalized least squares.
pub type DrPlsParams = ArPlsParams;
/// Parameters for improved asymmetrically reweighted penalized least squares.
pub type IarPlsParams = ArPlsParams;
/// Parameters for adaptive smoothness penalized least squares.
pub type AsPlsParams = ArPlsParams;
/// Parameters for peaked signal asymmetric least squares.
pub type PsalsaParams = AslsParams;
/// Parameters for derivative peaked signal asymmetric least squares.
pub type DerPsalsaParams = AslsParams;
/// Parameters for Bayesian reweighted penalized least squares.
pub type BrPlsParams = ArPlsParams;
/// Parameters for locally symmetric reweighted penalized least squares.
pub type LsrPlsParams = ArPlsParams;

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
    asls(y, params)
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
