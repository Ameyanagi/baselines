//! Penalized spline baseline algorithms.
//!
//! Dedicated P-spline implementations are being added family by family. The
//! remaining compatibility APIs reuse the closest one-dimensional baseline
//! engines until their dedicated spline forms are implemented.

mod weights;

use crate::fit::{Fit, FitReport};
use crate::linalg::pspline::PenalizedSpline;
use crate::morphology::MorphologyParams;
use crate::smoothing::{SmoothingParams, peak_filling};
use crate::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, arpls, asls, aspls, drpls, iasls,
};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};
use weights::{
    airpls_weights, arpls_weights, brpls_weights, derivative_peak_screening_weights,
    derpsalsa_weights, iarpls_weights, lsrpls_weights, mpls_anchor_weights, psalsa_weights,
    standard_deviation,
};

const PSPLINE_NUM_KNOTS: usize = 100;
const PSPLINE_DEGREE: usize = 3;
const PSPLINE_DIFF_ORDER: usize = 2;
const PSPLINE_MPLS_LAMBDA: f64 = 1.0e3;
const PSPLINE_MPLS_P: f64 = 0.0;

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

    for iter in 0..=params.whittaker.max_iter {
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
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
/// - Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using
///   adaptive iteratively reweighted penalized least squares", *Analyst*, 2010.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_airpls` is used as a behavioral reference.
pub fn pspline_airpls(y: &[f64], params: AirPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    validate_spline_signal("pspline_airpls", y)?;

    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let y_l1_norm = y
        .iter()
        .map(|value| value.abs())
        .sum::<f64>()
        .max(f64::EPSILON);
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..=params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let (new_weights, residual_l1_norm, exit_early) = airpls_weights(y, &baseline, iter + 1);
        if exit_early {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iter + 1, false, tolerance),
            });
        }

        tolerance = residual_l1_norm / y_l1_norm;
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
    })
}

/// Fits a penalized-spline arPLS baseline.
///
/// # References
///
/// - J. Baek et al., "Baseline correction using asymmetrically reweighted
///   penalized least squares smoothing", *Analyst*, 2015.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_arpls` is used as a behavioral reference.
pub fn pspline_arpls(y: &[f64], params: ArPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    validate_spline_signal("pspline_arpls", y)?;

    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..=params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let Some(new_weights) = arpls_weights(y, &baseline) else {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iter + 1, false, tolerance),
            });
        };
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
    })
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
/// - J. Ye et al., "Baseline correction method based on improved
///   asymmetrically reweighted penalized least squares for Raman spectrum",
///   *Applied Optics*, 2020.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_iarpls` is used as a behavioral reference.
pub fn pspline_iarpls(y: &[f64], params: IarPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    validate_spline_signal("pspline_iarpls", y)?;

    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..=params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let Some(new_weights) = iarpls_weights(y, &baseline, iter + 1) else {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iter + 1, false, tolerance),
            });
        };
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
    })
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
/// - S. Oller-Moreno et al., "Adaptive Asymmetric Least Squares baseline
///   estimation for analytical instruments", IEEE SSD, 2014.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_psalsa` is used as a behavioral reference.
pub fn pspline_psalsa(y: &[f64], params: PsalsaParams) -> Result<Fit> {
    params.validate()?;
    validate_spline_signal("pspline_psalsa", y)?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }

    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..=params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let new_weights = psalsa_weights(y, &baseline, params.p, k);
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
    })
}

/// Fits a penalized-spline derpsalsa baseline.
///
/// # References
///
/// - V. Korepanov, "Asymmetric least-squares baseline algorithm with peak
///   screening for automatic processing of the Raman spectra", *Journal of
///   Raman Spectroscopy*, 2020.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_derpsalsa` is used as a behavioral reference.
pub fn pspline_derpsalsa(y: &[f64], params: DerPsalsaParams) -> Result<Fit> {
    params.validate()?;
    validate_spline_signal("pspline_derpsalsa", y)?;
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
    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..=params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let new_weights = derpsalsa_weights(y, &baseline, params.p, k, &partial_weights);
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
    })
}

/// Fits a penalized-spline MPLS baseline.
///
/// # References
///
/// - Z. Li et al., "Morphological weighted penalized least squares for
///   background correction", *Analyst*, 2013.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_mpls` is used as a behavioral reference.
pub fn pspline_mpls(y: &[f64], params: MorphologyParams) -> Result<Fit> {
    validate_spline_signal("pspline_mpls", y)?;
    if params.window_size == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "window_size",
            reason: "must be greater than zero",
        });
    }

    let radius = params.window_size / 2;
    let weights = mpls_anchor_weights(y, radius, PSPLINE_MPLS_P);
    let baseline = default_pspline(y.len()).solve(y, &weights, PSPLINE_MPLS_LAMBDA)?;
    Ok(Fit {
        baseline,
        report: FitReport::new(1, true, 0.0),
    })
}

/// Fits a penalized-spline brPLS baseline.
///
/// # References
///
/// - Q. Wang et al., "Spectral baseline estimation using penalized least
///   squares with weights derived from the Bayesian method", *Nuclear Science
///   and Techniques*, 2022.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_brpls` is used as a behavioral reference.
pub fn pspline_brpls(y: &[f64], params: BrPlsParams) -> Result<Fit> {
    params.validate()?;
    validate_spline_signal("pspline_brpls", y)?;

    let pspline = default_pspline(y.len());
    let mut weights = vec![1.0; y.len()];
    let mut baseline = y.to_vec();
    let mut latest_weights = weights.clone();
    let mut beta = 0.5;
    let mut tolerance = f64::INFINITY;
    let mut outer_tolerance = f64::INFINITY;
    let mut iterations = 0usize;

    for outer in 0..=params.max_iter_2 {
        for inner in 0..=params.whittaker.max_iter {
            let new_baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
            iterations += 1;
            let Some(new_weights) = brpls_weights(y, &new_baseline, beta) else {
                return Ok(Fit {
                    baseline,
                    report: FitReport::new(iterations, false, tolerance),
                });
            };

            tolerance = relative_change(&baseline, &new_baseline);
            latest_weights = new_weights;
            if tolerance < params.whittaker.tol {
                if outer == 0 && inner == 0 {
                    baseline = new_baseline;
                }
                break;
            }

            weights.clone_from(&latest_weights);
            baseline = new_baseline;
        }

        weights.clone_from(&latest_weights);
        let weight_mean = weights.iter().sum::<f64>() / weights.len() as f64;
        outer_tolerance = (beta + weight_mean - 1.0).abs();
        if outer_tolerance < params.tol_2 {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iterations, true, outer_tolerance),
            });
        }
        beta = 1.0 - weight_mean;
    }

    Ok(Fit {
        baseline,
        report: FitReport::new(iterations, false, outer_tolerance.max(tolerance)),
    })
}

/// Fits a penalized-spline lsrPLS baseline.
///
/// # References
///
/// - Z. Heng et al., "Baseline correction for Raman spectra based on locally
///   symmetric reweighted penalized least squares", *Chinese Journal of
///   Lasers*, 2018.
/// - P. H. C. Eilers, B. D. Marx, and M. Durban, "Twenty years of P-splines",
///   *SORT*, 2015.
/// - `pybaselines.Baseline.pspline_lsrpls` is used as a behavioral reference.
pub fn pspline_lsrpls(y: &[f64], params: LsrPlsParams) -> Result<Fit> {
    params.whittaker.validate()?;
    validate_spline_signal("pspline_lsrpls", y)?;

    let mut weights = vec![1.0; y.len()];
    let pspline = default_pspline(y.len());
    let mut tolerance = f64::INFINITY;
    let mut baseline = Vec::new();

    for iter in 0..=params.whittaker.max_iter {
        baseline = pspline.solve(y, &weights, params.whittaker.lambda)?;
        let Some(new_weights) = lsrpls_weights(y, &baseline, iter + 1) else {
            return Ok(Fit {
                baseline,
                report: FitReport::new(iter + 1, false, tolerance),
            });
        };
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
        report: FitReport::new(params.whittaker.max_iter + 1, false, tolerance),
    })
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
