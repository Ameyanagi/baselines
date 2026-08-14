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
    Ok(fit_with(y, method)?.baseline)
}

/// Returns `y - baseline` using the default [`Method::Asls`].
pub fn correct(y: &[f64]) -> Result<Vec<f64>> {
    correct_with(y, Method::default())
}

/// Returns `y - baseline` using a selected method and its defaults.
pub fn correct_with(y: &[f64], method: Method) -> Result<Vec<f64>> {
    fit_with(y, method)?.corrected(y)
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
    Ok(fit_2d_with(data, rows, cols, method)?.baseline)
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
    fit_2d_with(data, rows, cols, method)?.corrected(data)
}

fn fit_with(y: &[f64], method: Method) -> Result<Fit1D> {
    match method {
        Method::Asls => whittaker::asls(y, AslsParams::default()),
        Method::Arpls => whittaker::arpls(y, ArPlsParams::default()),
        Method::Airpls => whittaker::airpls(y, AirPlsParams::default()),
        Method::RollingBall => morphology::rolling_ball(y, MorphologyParams::default()),
        Method::Polynomial => polynomial::poly(y, PolyParams::default()),
    }
}

fn fit_2d_with(data: &[f64], rows: usize, cols: usize, method: Method2D) -> Result<Fit2D> {
    let input = MatrixView::row_major(data, rows, cols)?;
    match method {
        Method2D::Asls => whittaker_2d::asls(input, whittaker_2d::Asls2DParams::default()),
        Method2D::Arpls => whittaker_2d::arpls(input, whittaker_2d::ArPls2DParams::default()),
        Method2D::RollingBall => {
            morphology_2d::rolling_ball(input, morphology_2d::Morphology2DParams::default())
        }
        Method2D::Polynomial => polynomial_2d::poly(input, polynomial_2d::Poly2DParams::default()),
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
}
