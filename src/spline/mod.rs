//! Penalized spline baseline algorithms.
//!
//! Dedicated P-spline implementations are being added family by family. The
//! remaining compatibility APIs reuse the closest one-dimensional baseline
//! engines until their dedicated spline forms are implemented.

use crate::fit::{Fit, FitReport};
use crate::linalg::pspline::PenalizedSpline;
use crate::morphology::{MorphologyParams, mpls};
use crate::smoothing::{SmoothingParams, peak_filling};
use crate::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, airpls, arpls, asls, aspls, brpls,
    derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

const PSPLINE_NUM_KNOTS: usize = 100;
const PSPLINE_DEGREE: usize = 3;
const PSPLINE_DIFF_ORDER: usize = 2;

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
/// - P. H. C. Eilers, "A Perfect Smoother", *Analytical Chemistry*, 2003.
/// - P. H. C. Eilers and H. F. M. Boelens, "Baseline Correction with
///   Asymmetric Least Squares Smoothing", 2005.
/// - P. H. C. Eilers, "Parametric Time Warping", *Analytical Chemistry*, 2004.
/// - P. H. C. Eilers, I. D. Currie, and M. Durban, "Fast and compact smoothing
///   on large multidimensional grids", *Computational Statistics & Data
///   Analysis*, 2006.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_asls` is used as a behavioral reference.
pub fn pspline_asls(y: &[f64], params: AslsParams) -> Result<Fit> {
    params.validate()?;
    validate_spline_signal("pspline_asls", y)?;

    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let new_weights: Vec<f64> = y
            .iter()
            .zip(&baseline)
            .map(|(observed, fitted)| {
                if observed > fitted {
                    params.p
                } else {
                    1.0 - params.p
                }
            })
            .collect();
        tolerance = relative_change(&weights, &new_weights);
        if tolerance < params.whittaker.tol {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iter + 1, true, tolerance),
            });
        }
        weights = new_weights;
    }

    Ok(Fit {
        baseline,
        report: FitReport::new(params.whittaker.max_iter, false, tolerance),
    })
}

/// Fits a penalized-spline IAsLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_iasls` is used as a behavioral reference.
pub fn pspline_iasls(y: &[f64], params: IaslsParams) -> Result<Fit> {
    iasls(y, params)
}

/// Fits a penalized-spline airPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_airpls` is used as a behavioral reference.
pub fn pspline_airpls(y: &[f64], params: AirPlsParams) -> Result<Fit> {
    airpls(y, params)
}

/// Fits a penalized-spline arPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_arpls` is used as a behavioral reference.
pub fn pspline_arpls(y: &[f64], params: ArPlsParams) -> Result<Fit> {
    arpls(y, params)
}

/// Fits a penalized-spline drPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_drpls` is used as a behavioral reference.
pub fn pspline_drpls(y: &[f64], params: DrPlsParams) -> Result<Fit> {
    drpls(y, params)
}

/// Fits a penalized-spline IarPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_iarpls` is used as a behavioral reference.
pub fn pspline_iarpls(y: &[f64], params: IarPlsParams) -> Result<Fit> {
    iarpls(y, params)
}

/// Fits a penalized-spline asPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_aspls` is used as a behavioral reference.
pub fn pspline_aspls(y: &[f64], params: AsPlsParams) -> Result<Fit> {
    aspls(y, params)
}

/// Fits a penalized-spline psalsa baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_psalsa` is used as a behavioral reference.
pub fn pspline_psalsa(y: &[f64], params: PsalsaParams) -> Result<Fit> {
    psalsa(y, params)
}

/// Fits a penalized-spline derpsalsa baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_derpsalsa` is used as a behavioral reference.
pub fn pspline_derpsalsa(y: &[f64], params: DerPsalsaParams) -> Result<Fit> {
    derpsalsa(y, params)
}

/// Fits a penalized-spline MPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_mpls` is used as a behavioral reference.
pub fn pspline_mpls(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    mpls(y, params)
}

/// Fits a penalized-spline brPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_brpls` is used as a behavioral reference.
pub fn pspline_brpls(y: &[f64], params: BrPlsParams) -> Result<Fit> {
    brpls(y, params)
}

/// Fits a penalized-spline lsrPLS baseline.
///
/// # References
///
/// - `pybaselines.Baseline.pspline_lsrpls` is used as a behavioral reference.
pub fn pspline_lsrpls(y: &[f64], params: LsrPlsParams) -> Result<Fit> {
    lsrpls(y, params)
}

fn validate_spline_signal(algorithm: &'static str, y: &[f64]) -> Result<()> {
    validate_signal(y)?;
    let min = PSPLINE_DEGREE + 2;
    if y.len() < min {
        return Err(BaselineError::TooShort {
            algorithm,
            len: y.len(),
            min,
        });
    }
    Ok(())
}

fn default_pspline(n: usize) -> PenalizedSpline {
    PenalizedSpline::new(
        n,
        PSPLINE_NUM_KNOTS.min(n).max(2),
        PSPLINE_DEGREE,
        PSPLINE_DIFF_ORDER,
    )
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
