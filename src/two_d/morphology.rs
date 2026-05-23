//! Two-dimensional morphology and smoothing baseline algorithms.
//!
//! # References
//!
//! - M. Kneen and H. Annegarn, "Algorithm for fitting XRF, SEM and PIXE
//!   X-ray spectra backgrounds", *Nuclear Instruments and Methods in Physics
//!   Research Section B*, 1996.
//! - L. Dai et al., "An Automated Baseline Correction Method Based on
//!   Iterative Morphological Operations", *Applied Spectroscopy*, 2018.
//! - `pybaselines.Baseline2D` morphology methods are used as behavioral
//!   references.

use crate::data::{MatrixView, MatrixViewMut};
use crate::fit::{Fit2D, FitReport};
use crate::{BaselineError, Result};

const IMOR_MAX_ITER: usize = 200;
const IMOR_TOL: f64 = 1.0e-3;

/// Parameters for two-dimensional window-based morphology baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Morphology2DParams {
    /// Full moving-window row count. Even values are accepted and use
    /// `window_rows / 2` samples on each side.
    pub window_rows: usize,
    /// Full moving-window column count. Even values are accepted and use
    /// `window_cols / 2` samples on each side.
    pub window_cols: usize,
}

impl Default for Morphology2DParams {
    fn default() -> Self {
        Self {
            window_rows: 7,
            window_cols: 7,
        }
    }
}

impl Morphology2DParams {
    fn validate(self) -> Result<()> {
        if self.window_rows == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "window_rows",
                reason: "must be greater than zero",
            });
        }
        if self.window_cols == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "window_cols",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }

    fn radii(self) -> (usize, usize) {
        (self.window_rows / 2, self.window_cols / 2)
    }
}

/// Estimates a 2D rolling-ball style baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.rolling_ball` is used as a behavioral reference.
pub fn rolling_ball(input: MatrixView<'_>, params: Morphology2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = rolling_ball_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Estimates a 2D rolling-ball baseline into an existing output buffer.
pub fn rolling_ball_into(
    input: MatrixView<'_>,
    params: Morphology2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    let (row_radius, col_radius) = params.radii();
    let opened = opening_reflect(
        input.as_slice(),
        input.rows(),
        input.cols(),
        row_radius,
        col_radius,
    );
    moving_mean_reflect(
        &opened,
        input.rows(),
        input.cols(),
        row_radius,
        col_radius,
        output.as_mut_slice(),
    );
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a 2D top-hat baseline using morphological opening.
///
/// # References
///
/// - `pybaselines.Baseline2D.tophat` is used as a behavioral reference.
pub fn tophat(input: MatrixView<'_>, params: Morphology2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = tophat_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Estimates a 2D top-hat baseline into an existing output buffer.
pub fn tophat_into(
    input: MatrixView<'_>,
    params: Morphology2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    let (row_radius, col_radius) = params.radii();
    let opened = opening_reflect(
        input.as_slice(),
        input.rows(),
        input.cols(),
        row_radius,
        col_radius,
    );
    output.as_mut_slice().copy_from_slice(&opened);
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates a 2D morphology baseline from an opening and averaged envelope.
///
/// # References
///
/// - `pybaselines.Baseline2D.mor` is used as a behavioral reference.
pub fn mor(input: MatrixView<'_>, params: Morphology2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = mor_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Estimates a 2D morphology baseline into an existing output buffer.
pub fn mor_into(
    input: MatrixView<'_>,
    params: Morphology2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    let (row_radius, col_radius) = params.radii();
    let opened = opening_reflect(
        input.as_slice(),
        input.rows(),
        input.cols(),
        row_radius,
        col_radius,
    );
    let averaged = average_opening_from_opened_reflect(
        &opened,
        input.rows(),
        input.cols(),
        row_radius,
        col_radius,
    );
    for ((target, opened), averaged) in output.as_mut_slice().iter_mut().zip(opened).zip(averaged) {
        *target = opened.min(averaged);
    }
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates an improved 2D morphology baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.imor` is used as a behavioral reference.
pub fn imor(input: MatrixView<'_>, params: Morphology2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = imor_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Estimates an improved 2D morphology baseline into an existing output buffer.
pub fn imor_into(
    input: MatrixView<'_>,
    params: Morphology2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    let (row_radius, col_radius) = params.radii();
    output.as_mut_slice().copy_from_slice(input.as_slice());
    let mut next = vec![0.0; input.len()];
    let mut averaged = vec![0.0; input.len()];
    let mut tolerance = f64::INFINITY;

    for iter in 0..=IMOR_MAX_ITER {
        average_opening_reflect(
            output.as_slice(),
            input.rows(),
            input.cols(),
            row_radius,
            col_radius,
            &mut averaged,
        );
        for ((target, observed), opened) in next.iter_mut().zip(input.as_slice()).zip(&averaged) {
            *target = observed.min(*opened);
        }
        tolerance = relative_change(output.as_slice(), &next);
        if tolerance < IMOR_TOL {
            output.as_mut_slice().copy_from_slice(&next);
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        output.as_mut_slice().copy_from_slice(&next);
    }

    Ok(FitReport::new(IMOR_MAX_ITER + 1, false, tolerance))
}

/// Estimates a 2D moving-median noise baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.noise_median` is used as a behavioral reference.
pub fn noise_median(input: MatrixView<'_>, params: Morphology2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = noise_median_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Estimates a 2D moving-median noise baseline into an existing output buffer.
pub fn noise_median_into(
    input: MatrixView<'_>,
    params: Morphology2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_input_output(input, &output, params)?;
    let (row_radius, col_radius) = params.radii();
    moving_median_reflect(
        input.as_slice(),
        input.rows(),
        input.cols(),
        row_radius,
        col_radius,
        output.as_mut_slice(),
    );
    Ok(FitReport::new(1, true, 0.0))
}

fn validate_input_output(
    input: MatrixView<'_>,
    output: &MatrixViewMut<'_>,
    params: Morphology2DParams,
) -> Result<()> {
    params.validate()?;
    let input_shape = input.shape();
    let output_shape = output.shape();
    if input_shape != output_shape {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: input_shape.len(),
            actual: output_shape.len(),
        });
    }
    Ok(())
}

fn opening_reflect(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
) -> Vec<f64> {
    let eroded = moving_min_reflect(data, rows, cols, row_radius, col_radius);
    let mut opened = vec![0.0; data.len()];
    moving_max_reflect(&eroded, rows, cols, row_radius, col_radius, &mut opened);
    opened
}

fn average_opening_reflect(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
    output: &mut [f64],
) {
    let opened = opening_reflect(data, rows, cols, row_radius, col_radius);
    let averaged = average_opening_from_opened_reflect(&opened, rows, cols, row_radius, col_radius);
    for ((target, opened), averaged) in output.iter_mut().zip(opened).zip(averaged) {
        *target = opened.min(averaged);
    }
}

fn average_opening_from_opened_reflect(
    opened: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
) -> Vec<f64> {
    let mut closed = vec![0.0; opened.len()];
    let dilated = moving_max_reflect_alloc(opened, rows, cols, row_radius, col_radius);
    moving_min_reflect_into(&dilated, rows, cols, row_radius, col_radius, &mut closed);
    opened
        .iter()
        .zip(closed)
        .map(|(opened, closed)| 0.5 * (opened + closed))
        .collect()
}

fn moving_min_reflect(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; data.len()];
    moving_min_reflect_into(data, rows, cols, row_radius, col_radius, &mut output);
    output
}

fn moving_min_reflect_into(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
    output: &mut [f64],
) {
    for row in 0..rows {
        for col in 0..cols {
            let mut best = f64::INFINITY;
            for window_row in window_indices(row, rows, row_radius) {
                let start = window_row * cols;
                for window_col in window_indices(col, cols, col_radius) {
                    best = best.min(data[start + window_col]);
                }
            }
            output[row * cols + col] = best;
        }
    }
}

fn moving_max_reflect_alloc(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
) -> Vec<f64> {
    let mut output = vec![0.0; data.len()];
    moving_max_reflect(data, rows, cols, row_radius, col_radius, &mut output);
    output
}

fn moving_max_reflect(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
    output: &mut [f64],
) {
    for row in 0..rows {
        for col in 0..cols {
            let mut best = f64::NEG_INFINITY;
            for window_row in window_indices(row, rows, row_radius) {
                let start = window_row * cols;
                for window_col in window_indices(col, cols, col_radius) {
                    best = best.max(data[start + window_col]);
                }
            }
            output[row * cols + col] = best;
        }
    }
}

fn moving_mean_reflect(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
    output: &mut [f64],
) {
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            let mut count = 0;
            for window_row in window_indices(row, rows, row_radius) {
                let start = window_row * cols;
                for window_col in window_indices(col, cols, col_radius) {
                    sum += data[start + window_col];
                    count += 1;
                }
            }
            output[row * cols + col] = sum / count as f64;
        }
    }
}

fn moving_median_reflect(
    data: &[f64],
    rows: usize,
    cols: usize,
    row_radius: usize,
    col_radius: usize,
    output: &mut [f64],
) {
    let mut window = Vec::with_capacity((2 * row_radius + 1) * (2 * col_radius + 1));
    for row in 0..rows {
        for col in 0..cols {
            window.clear();
            for window_row in window_indices(row, rows, row_radius) {
                let start = window_row * cols;
                for window_col in window_indices(col, cols, col_radius) {
                    window.push(data[start + window_col]);
                }
            }
            window.sort_by(f64::total_cmp);
            output[row * cols + col] = median_sorted(&window);
        }
    }
}

fn median_sorted(values: &[f64]) -> f64 {
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

fn window_indices(center: usize, len: usize, radius: usize) -> impl Iterator<Item = usize> {
    let window_len = 2 * radius + 1;
    (0..window_len).map(move |offset| {
        let raw = center as isize + offset as isize - radius as isize;
        reflect_index(raw, len)
    })
}

fn reflect_index(index: isize, len: usize) -> usize {
    debug_assert!(len > 0);
    if len == 1 {
        return 0;
    }
    let period = 2 * len as isize - 2;
    let wrapped = index.rem_euclid(period);
    if wrapped < len as isize {
        wrapped as usize
    } else {
        (period - wrapped) as usize
    }
}

fn relative_change(previous: &[f64], next: &[f64]) -> f64 {
    let numerator = previous
        .iter()
        .zip(next)
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>();
    let denominator = previous.iter().map(|value| value.abs()).sum::<f64>();
    numerator / denominator.max(f64::EPSILON)
}

#[cfg(test)]
mod tests {
    use super::{Morphology2DParams, rolling_ball};
    use crate::MatrixView;

    #[test]
    fn morphology_preserves_constant_surface() {
        let data = vec![2.0; 20];
        let view = MatrixView::row_major(&data, 4, 5).unwrap();
        let fit = rolling_ball(view, Morphology2DParams::default()).unwrap();
        assert!(
            fit.baseline
                .iter()
                .all(|value| (*value - 2.0).abs() < 1e-12)
        );
    }
}
