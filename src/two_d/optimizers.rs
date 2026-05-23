//! Two-dimensional optimizer and meta-algorithm baseline routines.
//!
//! # References
//!
//! - A. Cao et al., "A robust method for automated background subtraction of
//!   tissue fluorescence", *Journal of Raman Spectroscopy*, 2007.
//! - L. Chen et al., "Collaborative Penalized Least Squares for Background
//!   Correction of Multiple Raman Spectra", *Journal of Analytical Methods in
//!   Chemistry*, 2018.
//! - `pybaselines.Baseline2D` optimizer/meta methods are used as behavioral
//!   references.

use crate::data::{MatrixView, MatrixViewMut};
use crate::fit::{Fit2D, FitReport};
use crate::two_d::polynomial::{ModPoly2DParams, modpoly_into};
use crate::two_d::whittaker::{
    Asls2DParams, Whittaker2DWorkspace, asls as asls_2d, solve_fixed_weighted_system,
};
use crate::whittaker::{
    AslsParams as Asls1DParams, WhittakerParams as Whittaker1DParams,
    WhittakerWorkspace as Whittaker1DWorkspace, asls_into as asls_1d_into,
};
use crate::{BaselineError, Result};

/// Parameters for two-dimensional adaptive min-max fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveMinmax2DParams {
    /// Polynomial order used by the internal modified polynomial fit.
    pub order: usize,
    /// Maximum number of modified polynomial iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
}

impl Default for AdaptiveMinmax2DParams {
    fn default() -> Self {
        Self {
            order: 2,
            max_iter: 20,
            tol: 1.0e-3,
        }
    }
}

/// Parameters for fitting baselines independently along both axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IndividualAxes2DParams {
    /// One-dimensional AsLS parameters used for each row and column pass.
    pub asls: Asls1DParams,
}

impl Default for IndividualAxes2DParams {
    fn default() -> Self {
        Self {
            asls: Asls1DParams {
                whittaker: Whittaker1DParams {
                    lambda: 1.0e4,
                    ..Whittaker1DParams::default()
                },
                p: 0.01,
            },
        }
    }
}

/// Parameters for collaborative penalized least squares over related surfaces.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CollabPls2DParams {
    /// Shared two-dimensional AsLS parameters used to infer collaborative
    /// weights from the average surface.
    pub asls: Asls2DParams,
}

/// Fits a 2D adaptive min-max baseline.
///
/// The current Rust-native implementation uses the configured modified
/// polynomial baseline as the adaptive candidate surface.
///
/// # References
///
/// - `pybaselines.Baseline2D.adaptive_minmax` is used as a behavioral reference.
pub fn adaptive_minmax(input: MatrixView<'_>, params: AdaptiveMinmax2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = adaptive_minmax_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits a 2D adaptive min-max baseline into an existing output buffer.
pub fn adaptive_minmax_into(
    input: MatrixView<'_>,
    params: AdaptiveMinmax2DParams,
    output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_iter_params(params.max_iter, params.tol)?;
    modpoly_into(
        input,
        ModPoly2DParams {
            order: params.order,
            max_iter: params.max_iter,
            tol: params.tol,
        },
        output,
    )
}

/// Fits a baseline by applying 1D AsLS along rows and then columns.
///
/// # References
///
/// - `pybaselines.Baseline2D.individual_axes` is used as a behavioral reference.
pub fn individual_axes(input: MatrixView<'_>, params: IndividualAxes2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = individual_axes_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Fits an individual-axes baseline into an existing output buffer.
pub fn individual_axes_into(
    input: MatrixView<'_>,
    params: IndividualAxes2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_individual_axes_input(input, &output, params)?;
    let rows = input.rows();
    let cols = input.cols();
    let mut workspace = Whittaker1DWorkspace::new(cols.max(rows));
    output.as_mut_slice().fill(0.0);

    let mut column_input = vec![0.0; rows];
    let mut column_output = vec![0.0; rows];
    for col in 0..cols {
        for (row, value) in column_input.iter_mut().enumerate() {
            let index = row * cols + col;
            *value = input.as_slice()[index] - output.as_slice()[index];
        }
        asls_1d_into(
            &column_input,
            params.asls,
            &mut column_output,
            &mut workspace,
        )?;
        for (row, value) in column_output.iter().enumerate() {
            output.as_mut_slice()[row * cols + col] += *value;
        }
    }

    let mut row_input = vec![0.0; cols];
    let mut row_output = vec![0.0; cols];
    for row in 0..rows {
        let start = row * cols;
        for (col, value) in row_input.iter_mut().enumerate() {
            let index = start + col;
            *value = input.as_slice()[index] - output.as_slice()[index];
        }
        asls_1d_into(&row_input, params.asls, &mut row_output, &mut workspace)?;
        for (col, value) in row_output.iter().enumerate() {
            output.as_mut_slice()[start + col] += *value;
        }
    }

    Ok(FitReport::new(
        params.asls.whittaker.max_iter * 2,
        true,
        0.0,
    ))
}

/// Fits collaborative PLS baselines for related row-major surfaces.
///
/// # References
///
/// - `pybaselines.Baseline2D.collab_pls` is used as a behavioral reference.
pub fn collab_pls(surfaces: &[MatrixView<'_>], params: CollabPls2DParams) -> Result<Vec<Fit2D>> {
    validate_surfaces(surfaces)?;

    let rows = surfaces[0].rows();
    let cols = surfaces[0].cols();
    let len = surfaces[0].len();
    let mut average = vec![0.0; len];
    for surface in surfaces {
        for (target, value) in average.iter_mut().zip(surface.as_slice()) {
            *target += value;
        }
    }
    let scale = 1.0 / surfaces.len() as f64;
    for value in &mut average {
        *value *= scale;
    }

    let average_view = MatrixView::row_major(&average, rows, cols)?;
    let shared_fit = asls_2d(average_view, params.asls)?;
    let weights = asls_weights(&average, &shared_fit.baseline, params.asls.p);

    let mut workspace = Whittaker2DWorkspace::new(len);
    surfaces
        .iter()
        .map(|surface| {
            let mut baseline = vec![0.0; len];
            let output = MatrixViewMut::row_major(&mut baseline, rows, cols)?;
            let report = solve_fixed_weighted_system(
                *surface,
                params.asls.whittaker,
                &weights,
                output,
                &mut workspace,
            )?;
            Fit2D::new(baseline, rows, cols, report)
        })
        .collect()
}

fn validate_iter_params(max_iter: usize, tol: f64) -> Result<()> {
    if max_iter == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "max_iter",
            reason: "must be greater than zero",
        });
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "tol",
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

fn validate_individual_axes_input(
    input: MatrixView<'_>,
    output: &MatrixViewMut<'_>,
    params: IndividualAxes2DParams,
) -> Result<()> {
    params.asls.validate()?;
    if input.shape() != output.shape() {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: input.len(),
            actual: output.len(),
        });
    }
    if input.rows() < 3 || input.cols() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "individual_axes",
            len: input.len(),
            min: 9,
        });
    }
    Ok(())
}

fn validate_surfaces(surfaces: &[MatrixView<'_>]) -> Result<()> {
    if surfaces.is_empty() {
        return Err(BaselineError::EmptyInput);
    }
    let shape = surfaces[0].shape();
    for surface in surfaces {
        if surface.shape() != shape {
            return Err(BaselineError::LengthMismatch {
                name: "surface",
                expected: shape.len(),
                actual: surface.len(),
            });
        }
    }
    Ok(())
}

fn asls_weights(data: &[f64], baseline: &[f64], p: f64) -> Vec<f64> {
    data.iter()
        .zip(baseline)
        .map(
            |(observed, fitted)| {
                if observed > fitted { p } else { 1.0 - p }
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{IndividualAxes2DParams, individual_axes};
    use crate::MatrixView;

    #[test]
    fn individual_axes_preserves_constant_surface() {
        let data = vec![2.0; 30];
        let input = MatrixView::row_major(&data, 5, 6).unwrap();
        let fit = individual_axes(input, IndividualAxes2DParams::default()).unwrap();
        assert!(fit.baseline.iter().all(|value| (*value - 2.0).abs() < 1e-6));
    }
}
