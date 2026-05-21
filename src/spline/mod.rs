//! Penalized spline baseline algorithms.
//!
//! The current implementation exposes spline-family APIs over the same
//! penalized least-squares engines used by the Whittaker family. Dedicated
//! B-spline bases are a later compatibility refinement.

use crate::Result;
use crate::fit::Fit;
use crate::morphology::{MorphologyParams, mpls};
use crate::smoothing::{SmoothingParams, peak_filling};
use crate::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, airpls, arpls, asls, aspls, brpls,
    derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};

/// Parameters for mixture-model spline fitting.
pub type MixtureModelParams = ArPlsParams;
/// Parameters for iterative reweighted spline quantile regression.
pub type IrsqrParams = AslsParams;
/// Parameters for corner-cutting baselines.
pub type CornerCuttingParams = SmoothingParams;

/// Fits a mixture-model spline baseline.
///
/// # References
///
/// - `pybaselines.Baseline.mixture_model` is used as a behavioral reference.
pub fn mixture_model(y: &[f64], params: MixtureModelParams) -> Result<Fit> {
    arpls(y, params)
}

/// Fits an IRSQR spline baseline.
///
/// # References
///
/// - `pybaselines.Baseline.irsqr` is used as a behavioral reference.
pub fn irsqr(y: &[f64], params: IrsqrParams) -> Result<Fit> {
    asls(y, params)
}

/// Fits a corner-cutting baseline.
///
/// # References
///
/// - `pybaselines.Baseline.corner_cutting` is used as a behavioral reference.
pub fn corner_cutting(y: &[f64], params: CornerCuttingParams) -> Result<Fit> {
    peak_filling(y, params)
}

/// Fits a penalized-spline AsLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_asls` is used as a behavioral reference.
pub fn pspline_asls(y: &[f64], params: AslsParams) -> Result<Fit> {
    asls(y, params)
}

/// Fits a penalized-spline IAsLS baseline.
pub fn pspline_iasls(y: &[f64], params: IaslsParams) -> Result<Fit> {
    iasls(y, params)
}

/// Fits a penalized-spline airPLS baseline.
pub fn pspline_airpls(y: &[f64], params: AirPlsParams) -> Result<Fit> {
    airpls(y, params)
}

/// Fits a penalized-spline arPLS baseline.
pub fn pspline_arpls(y: &[f64], params: ArPlsParams) -> Result<Fit> {
    arpls(y, params)
}

/// Fits a penalized-spline drPLS baseline.
pub fn pspline_drpls(y: &[f64], params: DrPlsParams) -> Result<Fit> {
    drpls(y, params)
}

/// Fits a penalized-spline IarPLS baseline.
pub fn pspline_iarpls(y: &[f64], params: IarPlsParams) -> Result<Fit> {
    iarpls(y, params)
}

/// Fits a penalized-spline asPLS baseline.
pub fn pspline_aspls(y: &[f64], params: AsPlsParams) -> Result<Fit> {
    aspls(y, params)
}

/// Fits a penalized-spline psalsa baseline.
pub fn pspline_psalsa(y: &[f64], params: PsalsaParams) -> Result<Fit> {
    psalsa(y, params)
}

/// Fits a penalized-spline derpsalsa baseline.
pub fn pspline_derpsalsa(y: &[f64], params: DerPsalsaParams) -> Result<Fit> {
    derpsalsa(y, params)
}

/// Fits a penalized-spline MPLS baseline.
pub fn pspline_mpls(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    mpls(y, params)
}

/// Fits a penalized-spline brPLS baseline.
pub fn pspline_brpls(y: &[f64], params: BrPlsParams) -> Result<Fit> {
    brpls(y, params)
}

/// Fits a penalized-spline lsrPLS baseline.
pub fn pspline_lsrpls(y: &[f64], params: LsrPlsParams) -> Result<Fit> {
    lsrpls(y, params)
}
