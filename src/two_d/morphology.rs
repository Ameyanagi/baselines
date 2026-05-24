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

const IMOR_DEFAULT_MAX_ITER: usize = 200;
const IMOR_DEFAULT_TOL: f64 = 1.0e-3;

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

/// Parameters for [`imor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Imor2DParams {
    /// Shared morphology window parameters.
    pub morphology: Morphology2DParams,
    /// Maximum number of IMor update iterations.
    pub max_iter: usize,
    /// Relative baseline-change tolerance.
    pub tol: f64,
}

impl Default for Imor2DParams {
    fn default() -> Self {
        Self {
            morphology: Morphology2DParams::default(),
            max_iter: IMOR_DEFAULT_MAX_ITER,
            tol: IMOR_DEFAULT_TOL,
        }
    }
}

impl Imor2DParams {
    fn validate(self) -> Result<()> {
        self.morphology.validate()?;
        if !self.tol.is_finite() || self.tol <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "tol",
                reason: "must be finite and positive",
            });
        }
        Ok(())
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
    let shape = MorphologyShape::new(input.rows(), input.cols());
    let window = MorphologyWindow::new(row_radius, col_radius);
    let mut scratch = vec![0.0; input.len()];
    let mut eroded = vec![0.0; input.len()];
    let mut opened = vec![0.0; input.len()];
    opening_reflect_into(
        input.as_slice(),
        shape,
        window,
        &mut scratch,
        &mut eroded,
        &mut opened,
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
    let shape = MorphologyShape::new(input.rows(), input.cols());
    let window = MorphologyWindow::new(row_radius, col_radius);
    let mut scratch = vec![0.0; input.len()];
    let mut eroded = vec![0.0; input.len()];
    opening_reflect_into(
        input.as_slice(),
        shape,
        window,
        &mut scratch,
        &mut eroded,
        output.as_mut_slice(),
    );
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
    let shape = MorphologyShape::new(input.rows(), input.cols());
    let window = MorphologyWindow::new(row_radius, col_radius);
    let mut scratch = vec![0.0; input.len()];
    let mut opened = vec![0.0; input.len()];
    let mut eroded = vec![0.0; input.len()];
    let mut dilated = vec![0.0; input.len()];
    opening_reflect_into(
        input.as_slice(),
        shape,
        window,
        &mut scratch,
        &mut eroded,
        &mut opened,
    );
    average_opening_from_opened_reflect_into(
        &opened,
        shape,
        window,
        &mut scratch,
        &mut eroded,
        &mut dilated,
        output.as_mut_slice(),
    );
    for (target, opened) in output.as_mut_slice().iter_mut().zip(&opened) {
        *target = opened.min(*target);
    }
    Ok(FitReport::new(1, true, 0.0))
}

/// Estimates an improved 2D morphology baseline.
///
/// # References
///
/// - `pybaselines.Baseline2D.imor` is used as a behavioral reference.
pub fn imor(input: MatrixView<'_>, params: Imor2DParams) -> Result<Fit2D> {
    let mut baseline = vec![0.0; input.len()];
    let output = MatrixViewMut::row_major(&mut baseline, input.rows(), input.cols())?;
    let report = imor_into(input, params, output)?;
    Fit2D::new(baseline, input.rows(), input.cols(), report)
}

/// Estimates an improved 2D morphology baseline into an existing output buffer.
pub fn imor_into(
    input: MatrixView<'_>,
    params: Imor2DParams,
    mut output: MatrixViewMut<'_>,
) -> Result<FitReport> {
    validate_imor_input_output(input, &output, params)?;
    let (row_radius, col_radius) = params.morphology.radii();
    let shape = MorphologyShape::new(input.rows(), input.cols());
    let window = MorphologyWindow::new(row_radius, col_radius);
    output.as_mut_slice().copy_from_slice(input.as_slice());
    let mut workspace = Morphology2DWorkspace::new(input.len());
    let mut tolerance = f64::INFINITY;

    for iter in 0..=params.max_iter {
        average_opening_reflect_into(output.as_slice(), shape, window, &mut workspace);
        for ((target, observed), opened) in workspace
            .next
            .iter_mut()
            .zip(input.as_slice())
            .zip(&workspace.averaged)
        {
            *target = observed.min(*opened);
        }
        tolerance = relative_change(output.as_slice(), &workspace.next);
        if tolerance < params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        output.as_mut_slice().copy_from_slice(&workspace.next);
    }

    Ok(FitReport::new(
        params.max_iter.saturating_add(1),
        false,
        tolerance,
    ))
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

fn validate_imor_input_output(
    input: MatrixView<'_>,
    output: &MatrixViewMut<'_>,
    params: Imor2DParams,
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

#[derive(Debug, Clone, Copy)]
struct MorphologyShape {
    rows: usize,
    cols: usize,
}

impl MorphologyShape {
    fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    fn len(self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Debug, Clone, Copy)]
struct MorphologyWindow {
    row_radius: usize,
    col_radius: usize,
}

impl MorphologyWindow {
    fn new(row_radius: usize, col_radius: usize) -> Self {
        Self {
            row_radius,
            col_radius,
        }
    }
}

#[derive(Debug, Clone)]
struct Morphology2DWorkspace {
    scratch: Vec<f64>,
    opened: Vec<f64>,
    eroded: Vec<f64>,
    dilated: Vec<f64>,
    averaged: Vec<f64>,
    next: Vec<f64>,
}

impl Morphology2DWorkspace {
    fn new(len: usize) -> Self {
        Self {
            scratch: vec![0.0; len],
            opened: vec![0.0; len],
            eroded: vec![0.0; len],
            dilated: vec![0.0; len],
            averaged: vec![0.0; len],
            next: vec![0.0; len],
        }
    }
}

fn opening_reflect_into(
    data: &[f64],
    shape: MorphologyShape,
    window: MorphologyWindow,
    scratch: &mut [f64],
    eroded: &mut [f64],
    output: &mut [f64],
) {
    moving_min_reflect_with_workspace(data, shape, window, scratch, eroded);
    moving_max_reflect_with_workspace(eroded, shape, window, scratch, output);
}

fn average_opening_reflect_into(
    data: &[f64],
    shape: MorphologyShape,
    window: MorphologyWindow,
    workspace: &mut Morphology2DWorkspace,
) {
    opening_reflect_into(
        data,
        shape,
        window,
        &mut workspace.scratch,
        &mut workspace.eroded,
        &mut workspace.opened,
    );
    average_opening_from_opened_reflect_into(
        &workspace.opened,
        shape,
        window,
        &mut workspace.scratch,
        &mut workspace.eroded,
        &mut workspace.dilated,
        &mut workspace.averaged,
    );
}

fn average_opening_from_opened_reflect_into(
    opened: &[f64],
    shape: MorphologyShape,
    window: MorphologyWindow,
    scratch: &mut [f64],
    eroded: &mut [f64],
    dilated: &mut [f64],
    output: &mut [f64],
) {
    moving_max_reflect_with_workspace(opened, shape, window, scratch, dilated);
    moving_min_reflect_with_workspace(opened, shape, window, scratch, eroded);
    for ((target, dilated), eroded) in output.iter_mut().zip(dilated).zip(eroded) {
        *target = 0.5 * (*dilated + *eroded);
    }
}

fn moving_min_reflect_with_workspace(
    data: &[f64],
    shape: MorphologyShape,
    window: MorphologyWindow,
    scratch: &mut [f64],
    output: &mut [f64],
) {
    moving_extreme_reflect_with_workspace(
        data,
        shape,
        window,
        scratch,
        output,
        f64::INFINITY,
        f64::min,
    );
}

fn moving_max_reflect_with_workspace(
    data: &[f64],
    shape: MorphologyShape,
    window: MorphologyWindow,
    scratch: &mut [f64],
    output: &mut [f64],
) {
    moving_extreme_reflect_with_workspace(
        data,
        shape,
        window,
        scratch,
        output,
        f64::NEG_INFINITY,
        f64::max,
    );
}

fn moving_extreme_reflect_with_workspace(
    data: &[f64],
    shape: MorphologyShape,
    window: MorphologyWindow,
    scratch: &mut [f64],
    output: &mut [f64],
    initial: f64,
    combine: fn(f64, f64) -> f64,
) {
    debug_assert_eq!(data.len(), shape.len());
    debug_assert_eq!(scratch.len(), data.len());
    debug_assert_eq!(output.len(), data.len());

    for row in 0..shape.rows {
        let row_start = row * shape.cols;
        for col in 0..shape.cols {
            let mut best = initial;
            for window_col in window_indices(col, shape.cols, window.col_radius) {
                best = combine(best, data[row_start + window_col]);
            }
            scratch[row_start + col] = best;
        }
    }

    for row in 0..shape.rows {
        for col in 0..shape.cols {
            let mut best = initial;
            for window_row in window_indices(row, shape.rows, window.row_radius) {
                best = combine(best, scratch[window_row * shape.cols + col]);
            }
            output[row * shape.cols + col] = best;
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
    let len = len as isize;
    let mut reflected = index;
    while reflected < 0 || reflected >= len {
        if reflected < 0 {
            reflected = -reflected - 1;
        } else {
            reflected = 2 * len - reflected - 1;
        }
    }
    reflected as usize
}

fn relative_change(previous: &[f64], next: &[f64]) -> f64 {
    let numerator = previous
        .iter()
        .zip(next)
        .map(|(left, right)| {
            let diff = right - left;
            diff * diff
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
