//! Small, batteries-included entry points for common baseline corrections.
//!
//! The family modules and method-chain API remain available when an algorithm
//! needs custom parameters, reusable workspaces, or convergence history. This
//! module is intended for the common case where the documented defaults are a
//! good starting point.

use std::fmt;
use std::str::FromStr;

use crate::data::MatrixView;
use crate::fit::{Fit1D, Fit2D};
use crate::morphology::{self, MorphologyParams};
use crate::polynomial::{self, PolyParams};
use crate::two_d::whittaker as whittaker_2d;
use crate::two_d::{morphology as morphology_2d, polynomial as polynomial_2d};
use crate::whittaker::{self, AirPlsParams, ArPlsParams, AslsParams};
use crate::{BaselineError, Result};

/// Common one-dimensional algorithms exposed by the simple API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// Asymmetric least-squares smoothing.
    #[default]
    Asls,
    /// Asymmetrically reweighted penalized least-squares smoothing.
    Arpls,
    /// Adaptive iteratively reweighted penalized least-squares smoothing.
    Airpls,
    /// Rolling-ball morphology.
    RollingBall,
    /// Direct least-squares polynomial fitting.
    Polynomial,
}

impl Method {
    /// Methods accepted by [`baseline_with`] and [`correct_with`].
    pub const ALL: [Self; 5] = [
        Self::Asls,
        Self::Arpls,
        Self::Airpls,
        Self::RollingBall,
        Self::Polynomial,
    ];

    /// Returns the stable lowercase method name used by language bindings.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asls => "asls",
            Self::Arpls => "arpls",
            Self::Airpls => "airpls",
            Self::RollingBall => "rolling_ball",
            Self::Polynomial => "polynomial",
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Method {
    type Err = BaselineError;

    fn from_str(value: &str) -> Result<Self> {
        if value.eq_ignore_ascii_case("asls") {
            Ok(Self::Asls)
        } else if value.eq_ignore_ascii_case("arpls") {
            Ok(Self::Arpls)
        } else if value.eq_ignore_ascii_case("airpls") {
            Ok(Self::Airpls)
        } else if value.eq_ignore_ascii_case("rolling_ball")
            || value.eq_ignore_ascii_case("rolling-ball")
        {
            Ok(Self::RollingBall)
        } else if value.eq_ignore_ascii_case("polynomial") || value.eq_ignore_ascii_case("poly") {
            Ok(Self::Polynomial)
        } else {
            Err(BaselineError::InvalidParameter {
                name: "method",
                reason: "expected asls, arpls, airpls, rolling_ball, or polynomial",
            })
        }
    }
}

/// Common two-dimensional algorithms exposed by the simple API.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method2D {
    /// Two-dimensional asymmetric least-squares smoothing.
    #[default]
    Asls,
    /// Two-dimensional asymmetrically reweighted penalized least-squares smoothing.
    Arpls,
    /// Two-dimensional rolling-ball morphology.
    RollingBall,
    /// Direct two-dimensional polynomial fitting.
    Polynomial,
}

/// Configurable parameters for the simple one-dimensional API.
///
/// Fields left as [`None`] use the selected algorithm's documented defaults.
/// Options that do not apply to the selected method return an error rather than
/// being silently ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BaselineOptions {
    /// Baseline algorithm.
    pub method: Method,
    /// Whittaker smoothness penalty for `asls`, `arpls`, and `airpls`.
    pub lambda: Option<f64>,
    /// Asymmetry parameter for `asls`.
    pub p: Option<f64>,
    /// Maximum iterations for `asls`, `arpls`, and `airpls`.
    pub max_iter: Option<usize>,
    /// Convergence tolerance for `asls`, `arpls`, and `airpls`.
    pub tol: Option<f64>,
    /// Moving-window size for `rolling_ball`.
    pub window_size: Option<usize>,
    /// Polynomial degree for `polynomial`.
    pub order: Option<usize>,
}

impl BaselineOptions {
    fn reject_unsupported(&self, supported: &[&str]) -> Result<()> {
        for (name, is_set) in [
            ("lambda", self.lambda.is_some()),
            ("p", self.p.is_some()),
            ("max_iter", self.max_iter.is_some()),
            ("tol", self.tol.is_some()),
            ("window_size", self.window_size.is_some()),
            ("order", self.order.is_some()),
        ] {
            if is_set && !supported.contains(&name) {
                return Err(BaselineError::InvalidParameter {
                    name,
                    reason: "is not supported by the selected method",
                });
            }
        }
        Ok(())
    }

    fn whittaker(&self) -> crate::whittaker::WhittakerParams {
        let mut params = crate::whittaker::WhittakerParams::default();
        if let Some(lambda) = self.lambda {
            params.lambda = lambda;
        }
        if let Some(max_iter) = self.max_iter {
            params.max_iter = max_iter;
        }
        if let Some(tol) = self.tol {
            params.tol = tol;
        }
        params
    }
}

/// Configurable parameters for the simple two-dimensional API.
///
/// Fields left as [`None`] use the selected algorithm's documented defaults.
/// Options that do not apply to the selected method return an error rather than
/// being silently ignored.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BaselineOptions2D {
    /// Baseline algorithm.
    pub method: Method2D,
    /// Shared Whittaker smoothness penalty for `asls` and `arpls`.
    pub lambda: Option<f64>,
    /// Optional row-axis Whittaker smoothness penalty.
    pub lambda_rows: Option<f64>,
    /// Optional column-axis Whittaker smoothness penalty.
    pub lambda_cols: Option<f64>,
    /// Asymmetry parameter for `asls`.
    pub p: Option<f64>,
    /// Maximum reweighting iterations for `asls` and `arpls`.
    pub max_iter: Option<usize>,
    /// Reweighting convergence tolerance for `asls` and `arpls`.
    pub tol: Option<f64>,
    /// Maximum conjugate-gradient iterations per weighted solve.
    pub cg_max_iter: Option<usize>,
    /// Conjugate-gradient residual tolerance.
    pub cg_tol: Option<f64>,
    /// Moving-window row count for `rolling_ball`.
    pub window_rows: Option<usize>,
    /// Moving-window column count for `rolling_ball`.
    pub window_cols: Option<usize>,
    /// Polynomial degree for `polynomial`.
    pub order: Option<usize>,
}

impl BaselineOptions2D {
    fn reject_unsupported(&self, supported: &[&str]) -> Result<()> {
        for (name, is_set) in [
            ("lambda", self.lambda.is_some()),
            ("lambda_rows", self.lambda_rows.is_some()),
            ("lambda_cols", self.lambda_cols.is_some()),
            ("p", self.p.is_some()),
            ("max_iter", self.max_iter.is_some()),
            ("tol", self.tol.is_some()),
            ("cg_max_iter", self.cg_max_iter.is_some()),
            ("cg_tol", self.cg_tol.is_some()),
            ("window_rows", self.window_rows.is_some()),
            ("window_cols", self.window_cols.is_some()),
            ("order", self.order.is_some()),
        ] {
            if is_set && !supported.contains(&name) {
                return Err(BaselineError::InvalidParameter {
                    name,
                    reason: "is not supported by the selected method",
                });
            }
        }
        Ok(())
    }

    fn whittaker(&self) -> whittaker_2d::Whittaker2DParams {
        let mut params = whittaker_2d::Whittaker2DParams::default();
        if let Some(lambda) = self.lambda {
            params.lambda = lambda;
        }
        params.lambda_rows = self.lambda_rows;
        params.lambda_cols = self.lambda_cols;
        if let Some(max_iter) = self.max_iter {
            params.max_iter = max_iter;
        }
        if let Some(tol) = self.tol {
            params.tol = tol;
        }
        if let Some(cg_max_iter) = self.cg_max_iter {
            params.cg_max_iter = cg_max_iter;
        }
        if let Some(cg_tol) = self.cg_tol {
            params.cg_tol = cg_tol;
        }
        params
    }
}

impl Method2D {
    /// Methods accepted by [`baseline_2d_with`] and [`correct_2d_with`].
    pub const ALL: [Self; 4] = [Self::Asls, Self::Arpls, Self::RollingBall, Self::Polynomial];

    /// Returns the stable lowercase method name used by language bindings.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asls => "asls",
            Self::Arpls => "arpls",
            Self::RollingBall => "rolling_ball",
            Self::Polynomial => "polynomial",
        }
    }
}

impl fmt::Display for Method2D {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Method2D {
    type Err = BaselineError;

    fn from_str(value: &str) -> Result<Self> {
        if value.eq_ignore_ascii_case("asls") {
            Ok(Self::Asls)
        } else if value.eq_ignore_ascii_case("arpls") {
            Ok(Self::Arpls)
        } else if value.eq_ignore_ascii_case("rolling_ball")
            || value.eq_ignore_ascii_case("rolling-ball")
        {
            Ok(Self::RollingBall)
        } else if value.eq_ignore_ascii_case("polynomial") || value.eq_ignore_ascii_case("poly") {
            Ok(Self::Polynomial)
        } else {
            Err(BaselineError::InvalidParameter {
                name: "method",
                reason: "expected asls, arpls, rolling_ball, or polynomial",
            })
        }
    }
}

/// Estimates a one-dimensional baseline with the default [`Method::Asls`].
pub fn baseline(y: &[f64]) -> Result<Vec<f64>> {
    baseline_with(y, Method::default())
}

/// Estimates a one-dimensional baseline with a selected method and its defaults.
pub fn baseline_with(y: &[f64], method: Method) -> Result<Vec<f64>> {
    Ok(fit_with_options(
        y,
        BaselineOptions {
            method,
            ..BaselineOptions::default()
        },
    )?
    .baseline)
}

/// Estimates a one-dimensional baseline with configurable parameters.
pub fn baseline_with_options(y: &[f64], options: BaselineOptions) -> Result<Vec<f64>> {
    Ok(fit_with_options(y, options)?.baseline)
}

/// Returns `y - baseline` using the default [`Method::Asls`].
pub fn correct(y: &[f64]) -> Result<Vec<f64>> {
    correct_with(y, Method::default())
}

/// Returns `y - baseline` using a selected method and its defaults.
pub fn correct_with(y: &[f64], method: Method) -> Result<Vec<f64>> {
    fit_with_options(
        y,
        BaselineOptions {
            method,
            ..BaselineOptions::default()
        },
    )?
    .corrected(y)
}

/// Corrects a one-dimensional signal with configurable parameters.
pub fn correct_with_options(y: &[f64], options: BaselineOptions) -> Result<Vec<f64>> {
    fit_with_options(y, options)?.corrected(y)
}

/// Estimates a row-major two-dimensional baseline with the default [`Method2D::Asls`].
pub fn baseline_2d(data: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
    baseline_2d_with(data, rows, cols, Method2D::default())
}

/// Estimates a row-major two-dimensional baseline with a selected method and its defaults.
pub fn baseline_2d_with(
    data: &[f64],
    rows: usize,
    cols: usize,
    method: Method2D,
) -> Result<Vec<f64>> {
    Ok(fit_2d_with_options(
        data,
        rows,
        cols,
        BaselineOptions2D {
            method,
            ..BaselineOptions2D::default()
        },
    )?
    .baseline)
}

/// Estimates a row-major two-dimensional baseline with configurable parameters.
pub fn baseline_2d_with_options(
    data: &[f64],
    rows: usize,
    cols: usize,
    options: BaselineOptions2D,
) -> Result<Vec<f64>> {
    Ok(fit_2d_with_options(data, rows, cols, options)?.baseline)
}

/// Returns `data - baseline` using the default [`Method2D::Asls`].
pub fn correct_2d(data: &[f64], rows: usize, cols: usize) -> Result<Vec<f64>> {
    correct_2d_with(data, rows, cols, Method2D::default())
}

/// Returns `data - baseline` using a selected method and its defaults.
pub fn correct_2d_with(
    data: &[f64],
    rows: usize,
    cols: usize,
    method: Method2D,
) -> Result<Vec<f64>> {
    fit_2d_with_options(
        data,
        rows,
        cols,
        BaselineOptions2D {
            method,
            ..BaselineOptions2D::default()
        },
    )?
    .corrected(data)
}

/// Corrects row-major two-dimensional data with configurable parameters.
pub fn correct_2d_with_options(
    data: &[f64],
    rows: usize,
    cols: usize,
    options: BaselineOptions2D,
) -> Result<Vec<f64>> {
    fit_2d_with_options(data, rows, cols, options)?.corrected(data)
}

/// Fits a one-dimensional baseline and returns convergence metadata.
pub fn fit_with_options(y: &[f64], options: BaselineOptions) -> Result<Fit1D> {
    match options.method {
        Method::Asls => {
            options.reject_unsupported(&["lambda", "p", "max_iter", "tol"])?;
            let mut params = AslsParams {
                whittaker: options.whittaker(),
                ..AslsParams::default()
            };
            if let Some(p) = options.p {
                params.p = p;
            }
            whittaker::asls(y, params)
        }
        Method::Arpls => {
            options.reject_unsupported(&["lambda", "max_iter", "tol"])?;
            whittaker::arpls(
                y,
                ArPlsParams {
                    whittaker: options.whittaker(),
                },
            )
        }
        Method::Airpls => {
            options.reject_unsupported(&["lambda", "max_iter", "tol"])?;
            whittaker::airpls(
                y,
                AirPlsParams {
                    whittaker: options.whittaker(),
                },
            )
        }
        Method::RollingBall => {
            options.reject_unsupported(&["window_size"])?;
            morphology::rolling_ball(
                y,
                MorphologyParams {
                    window_size: options
                        .window_size
                        .unwrap_or(MorphologyParams::default().window_size),
                },
            )
        }
        Method::Polynomial => {
            options.reject_unsupported(&["order"])?;
            polynomial::poly(
                y,
                PolyParams {
                    order: options.order.unwrap_or(PolyParams::default().order),
                },
            )
        }
    }
}

/// Fits a row-major two-dimensional baseline and returns convergence metadata.
pub fn fit_2d_with_options(
    data: &[f64],
    rows: usize,
    cols: usize,
    options: BaselineOptions2D,
) -> Result<Fit2D> {
    let input = MatrixView::row_major(data, rows, cols)?;
    match options.method {
        Method2D::Asls => {
            options.reject_unsupported(&[
                "lambda",
                "lambda_rows",
                "lambda_cols",
                "p",
                "max_iter",
                "tol",
                "cg_max_iter",
                "cg_tol",
            ])?;
            let mut params = whittaker_2d::Asls2DParams {
                whittaker: options.whittaker(),
                ..whittaker_2d::Asls2DParams::default()
            };
            if let Some(p) = options.p {
                params.p = p;
            }
            whittaker_2d::asls(input, params)
        }
        Method2D::Arpls => {
            options.reject_unsupported(&[
                "lambda",
                "lambda_rows",
                "lambda_cols",
                "max_iter",
                "tol",
                "cg_max_iter",
                "cg_tol",
            ])?;
            whittaker_2d::arpls(
                input,
                whittaker_2d::ArPls2DParams {
                    whittaker: options.whittaker(),
                },
            )
        }
        Method2D::RollingBall => {
            options.reject_unsupported(&["window_rows", "window_cols"])?;
            let defaults = morphology_2d::Morphology2DParams::default();
            morphology_2d::rolling_ball(
                input,
                morphology_2d::Morphology2DParams {
                    window_rows: options.window_rows.unwrap_or(defaults.window_rows),
                    window_cols: options.window_cols.unwrap_or(defaults.window_cols),
                },
            )
        }
        Method2D::Polynomial => {
            options.reject_unsupported(&["order"])?;
            polynomial_2d::poly(
                input,
                polynomial_2d::Poly2DParams {
                    order: options
                        .order
                        .unwrap_or(polynomial_2d::Poly2DParams::default().order),
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_one_dimensional_methods_return_baselines_and_corrections() {
        let y: Vec<f64> = (0..64)
            .map(|index| 2.0 + 0.01 * index as f64 + if index == 30 { 5.0 } else { 0.0 })
            .collect();

        for method in Method::ALL {
            let fitted = baseline_with(&y, method).expect("simple method should fit");
            let corrected = correct_with(&y, method).expect("simple method should correct");
            assert_eq!(fitted.len(), y.len());
            assert_eq!(corrected.len(), y.len());
            assert!(fitted.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn simple_two_dimensional_methods_preserve_shape() {
        let data = vec![2.0; 6 * 7];
        for method in Method2D::ALL {
            let fitted = baseline_2d_with(&data, 6, 7, method)
                .expect("simple two-dimensional method should fit");
            assert_eq!(fitted.len(), data.len());
            assert!(fitted.iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn method_names_are_stable_and_parseable() {
        for method in Method::ALL {
            assert_eq!(method.as_str().parse::<Method>(), Ok(method));
        }
        for method in Method2D::ALL {
            assert_eq!(method.as_str().parse::<Method2D>(), Ok(method));
        }
    }

    #[test]
    fn configurable_options_return_metadata_and_reject_mismatches() {
        let y: Vec<f64> = (0..64).map(|index| 1.0 + index as f64 * 0.01).collect();
        let fit = fit_with_options(
            &y,
            BaselineOptions {
                lambda: Some(1.0e4),
                p: Some(0.05),
                max_iter: Some(8),
                ..BaselineOptions::default()
            },
        )
        .unwrap();
        assert_eq!(fit.baseline.len(), y.len());
        assert!(fit.report.iterations <= 8);

        let error = fit_with_options(
            &y,
            BaselineOptions {
                method: Method::Polynomial,
                lambda: Some(1.0e4),
                ..BaselineOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            BaselineError::InvalidParameter { name: "lambda", .. }
        ));
    }

    #[test]
    fn asls_accepts_the_documented_minimum_length() {
        let fit = fit_with_options(&[3.0, 2.0, 3.0], BaselineOptions::default()).unwrap();
        assert_eq!(fit.baseline.len(), 3);
        assert!(fit.baseline.iter().all(|value| value.is_finite()));
    }
}
