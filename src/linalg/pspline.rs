//! Dense P-spline helper for one-dimensional baseline methods.

use crate::linalg::dense::solve_dense;
use crate::{BaselineError, Result};

const DENSE_COMPAT_THRESHOLD: usize = 256;
const GENERAL_BANDED_MIN_BASES: usize = 100;

/// Dense penalized B-spline basis and solver.
#[derive(Debug, Clone)]
pub(crate) struct PenalizedSpline {
    basis: Vec<SparseBasisRow>,
    basis_midpoints: Vec<f64>,
    first_order_penalty: Vec<Vec<f64>>,
    penalty: Vec<Vec<f64>>,
    degree: usize,
    diff_order: usize,
}

/// Reusable buffers for repeated penalized-spline solves.
#[derive(Debug, Clone, Default)]
pub(crate) struct PenalizedSplineWorkspace {
    symmetric_bands: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    intermediate: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Debug, Clone)]
struct SparseBasisRow {
    start: usize,
    values: Vec<f64>,
}

impl PenalizedSplineWorkspace {
    /// Creates an empty P-spline workspace.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn reset_symmetric_bands(&mut self, n: usize, bandwidth: usize) {
        self.symmetric_bands
            .resize_with(bandwidth.saturating_add(1), Vec::new);
        for band in &mut self.symmetric_bands {
            band.resize(n, 0.0);
            band.fill(0.0);
        }
    }

    fn reset_rhs(&mut self, n: usize) {
        self.rhs.resize(n, 0.0);
        self.rhs.fill(0.0);
    }
}

impl PenalizedSpline {
    /// Creates a cubic P-spline basis over `n` equally spaced points in `[-1, 1]`.
    pub(crate) fn new(n: usize, num_knots: usize, degree: usize, diff_order: usize) -> Self {
        let x = scaled_domain(n);
        let knots = spline_knots(&x, num_knots, degree);
        let n_bases = knots.len() - degree - 1;
        let basis_midpoints = basis_midpoints(&knots, degree);
        let basis = x
            .iter()
            .map(|value| sparse_basis_row(*value, &knots, degree, n_bases))
            .collect();
        let first_order_penalty = difference_penalty_bands(n_bases, 1);
        let penalty = difference_penalty_bands(n_bases, diff_order);
        Self {
            basis,
            basis_midpoints,
            first_order_penalty,
            penalty,
            degree,
            diff_order,
        }
    }

    /// Fits a weighted penalized spline and returns the evaluated baseline.
    pub(crate) fn solve(&self, y: &[f64], weights: &[f64], lambda: f64) -> Result<Vec<f64>> {
        self.solve_with_options(y, weights, lambda, None, 0.0, 0.0)
    }

    /// Fits a weighted penalized spline into an existing output buffer.
    pub(crate) fn solve_into(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        output: &mut [f64],
        workspace: &mut PenalizedSplineWorkspace,
    ) -> Result<()> {
        if output.len() != self.basis.len() {
            return Err(BaselineError::LengthMismatch {
                name: "output",
                expected: self.basis.len(),
                actual: output.len(),
            });
        }
        self.solve_coefficients_banded_into(y, weights, lambda, workspace)?;
        self.evaluate_coefficients_into(&workspace.coefficients, output);
        Ok(())
    }

    /// Fits a weighted penalized spline and returns the baseline and coefficients.
    pub(crate) fn solve_with_coefficients(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let coefficients =
            self.solve_coefficients_with_options(y, weights, lambda, None, 0.0, 0.0)?;
        let baseline = self.evaluate_coefficients(&coefficients);
        Ok((baseline, coefficients))
    }

    /// Returns the number of spline basis functions.
    pub(crate) fn basis_count(&self) -> usize {
        self.penalty[0].len()
    }

    /// Fits a weighted penalized spline with row-scaled smoothness penalties.
    pub(crate) fn solve_with_row_scaled_penalty(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        row_scales: &[f64],
    ) -> Result<Vec<f64>> {
        self.solve_with_options(y, weights, lambda, Some(row_scales), 0.0, 0.0)
    }

    /// Fits a weighted penalized spline with drPLS basis penalties.
    pub(crate) fn solve_with_drpls_penalty(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        eta: f64,
        basis_weights: &[f64],
    ) -> Result<Vec<f64>> {
        let n_bases = self.basis_count();
        if basis_weights.len() != n_bases {
            return Err(BaselineError::LengthMismatch {
                name: "basis_weights",
                expected: n_bases,
                actual: basis_weights.len(),
            });
        }

        let row_scales: Vec<f64> = basis_weights
            .iter()
            .map(|weight| 1.0 - eta * weight)
            .collect();
        self.solve_with_options(y, weights, lambda, Some(&row_scales), 1.0, 0.0)
    }

    /// Fits a weighted penalized spline with an added data-domain first-difference penalty.
    pub(crate) fn solve_with_first_difference_penalty(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        first_difference_lambda: f64,
    ) -> Result<Vec<f64>> {
        self.solve_with_options(y, weights, lambda, None, 0.0, first_difference_lambda)
    }

    /// Interpolates sample-domain values onto the spline basis midpoints.
    pub(crate) fn interpolate_to_basis(&self, values: &[f64]) -> Vec<f64> {
        match values.len() {
            0 => vec![0.0; self.basis_midpoints.len()],
            1 => vec![values[0]; self.basis_midpoints.len()],
            len => self
                .basis_midpoints
                .iter()
                .map(|point| interpolate_sample(values, *point, len))
                .collect(),
        }
    }

    fn solve_with_options(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        row_scales: Option<&[f64]>,
        basis_first_difference_lambda: f64,
        data_first_difference_lambda: f64,
    ) -> Result<Vec<f64>> {
        let coefficients = self.solve_coefficients_with_options(
            y,
            weights,
            lambda,
            row_scales,
            basis_first_difference_lambda,
            data_first_difference_lambda,
        )?;
        Ok(self.evaluate_coefficients(&coefficients))
    }

    fn solve_coefficients_with_options(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        row_scales: Option<&[f64]>,
        basis_first_difference_lambda: f64,
        data_first_difference_lambda: f64,
    ) -> Result<Vec<f64>> {
        if row_scales.is_none()
            && basis_first_difference_lambda == 0.0
            && data_first_difference_lambda == 0.0
        {
            return self.solve_coefficients_banded(y, weights, lambda);
        }

        let n_bases = self.basis_count();
        if let Some(scales) = row_scales
            && scales.len() != n_bases
        {
            return Err(BaselineError::LengthMismatch {
                name: "row_scales",
                expected: n_bases,
                actual: scales.len(),
            });
        }
        if n_bases < GENERAL_BANDED_MIN_BASES {
            return self.solve_coefficients_dense_with_options(
                y,
                weights,
                lambda,
                row_scales,
                basis_first_difference_lambda,
                data_first_difference_lambda,
            );
        }

        let bandwidth = if data_first_difference_lambda > 0.0 {
            2 * (self.degree + 1)
        } else {
            self.degree.max(self.diff_order).max(1)
        };
        let mut normal = zero_general_bands(n_bases, bandwidth);
        let mut rhs = vec![0.0; n_bases];

        for ((basis_row, observed), weight) in self.basis.iter().zip(y).zip(weights) {
            for (row, row_value) in basis_row.entries() {
                rhs[row] += row_value * weight * observed;
                for (col, col_value) in basis_row.entries() {
                    add_general_band_value(
                        &mut normal,
                        bandwidth,
                        row,
                        col,
                        row_value * weight * col_value,
                    );
                }
            }
        }

        for row in 0..n_bases {
            let scale = row_scales.map_or(1.0, |scales| scales[row]);
            for col in
                row.saturating_sub(self.diff_order)..=(row + self.diff_order).min(n_bases - 1)
            {
                add_general_band_value(
                    &mut normal,
                    bandwidth,
                    row,
                    col,
                    lambda * scale * symmetric_band_value(&self.penalty, row, col),
                );
            }
        }

        if basis_first_difference_lambda > 0.0 {
            for row in 0..n_bases {
                for col in row.saturating_sub(1)..=(row + 1).min(n_bases - 1) {
                    add_general_band_value(
                        &mut normal,
                        bandwidth,
                        row,
                        col,
                        basis_first_difference_lambda
                            * symmetric_band_value(&self.first_order_penalty, row, col),
                    );
                }
            }
        }

        if data_first_difference_lambda > 0.0 {
            let mut basis_difference = Vec::with_capacity(2 * (self.degree + 1));
            for (basis_pair, observed_pair) in self.basis.windows(2).zip(y.windows(2)) {
                let observed_difference = observed_pair[1] - observed_pair[0];
                SparseBasisRow::difference_entries_into(
                    &basis_pair[0],
                    &basis_pair[1],
                    &mut basis_difference,
                );
                for &(row, basis_row_difference) in &basis_difference {
                    rhs[row] +=
                        data_first_difference_lambda * basis_row_difference * observed_difference;
                    for &(col, basis_col_difference) in &basis_difference {
                        add_general_band_value(
                            &mut normal,
                            bandwidth,
                            row,
                            col,
                            data_first_difference_lambda
                                * basis_row_difference
                                * basis_col_difference,
                        );
                    }
                }
            }
        }

        match solve_general_banded(&mut normal, bandwidth, &rhs) {
            Ok(solution) => Ok(solution),
            Err(error) => {
                if n_bases <= DENSE_COMPAT_THRESHOLD {
                    self.solve_coefficients_dense_with_options(
                        y,
                        weights,
                        lambda,
                        row_scales,
                        basis_first_difference_lambda,
                        data_first_difference_lambda,
                    )
                } else {
                    Err(error)
                }
            }
        }
    }

    fn solve_coefficients_dense_with_options(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        row_scales: Option<&[f64]>,
        basis_first_difference_lambda: f64,
        data_first_difference_lambda: f64,
    ) -> Result<Vec<f64>> {
        let n_bases = self.basis_count();
        let mut normal = vec![vec![0.0; n_bases]; n_bases];
        let mut rhs = vec![0.0; n_bases];

        for ((basis_row, observed), weight) in self.basis.iter().zip(y).zip(weights) {
            for (row, row_value) in basis_row.entries() {
                rhs[row] += row_value * weight * observed;
                for (col, col_value) in basis_row.entries() {
                    normal[row][col] += row_value * weight * col_value;
                }
            }
        }

        let penalty = symmetric_bands_to_dense(&self.penalty);
        for (row, (normal_row, penalty_row)) in normal.iter_mut().zip(&penalty).enumerate() {
            let scale = row_scales.map_or(1.0, |scales| scales[row]);
            for (normal_value, penalty_value) in normal_row.iter_mut().zip(penalty_row) {
                *normal_value += lambda * scale * penalty_value;
            }
        }

        if basis_first_difference_lambda > 0.0 {
            let first_order_penalty = symmetric_bands_to_dense(&self.first_order_penalty);
            for (normal_row, penalty_row) in normal.iter_mut().zip(&first_order_penalty) {
                for (normal_value, penalty_value) in normal_row.iter_mut().zip(penalty_row) {
                    *normal_value += basis_first_difference_lambda * penalty_value;
                }
            }
        }

        if data_first_difference_lambda > 0.0 {
            for (basis_pair, observed_pair) in self.basis.windows(2).zip(y.windows(2)) {
                let observed_difference = observed_pair[1] - observed_pair[0];
                for row in 0..n_bases {
                    let basis_row_difference =
                        basis_pair[1].value_at(row) - basis_pair[0].value_at(row);
                    rhs[row] +=
                        data_first_difference_lambda * basis_row_difference * observed_difference;
                    for (col, normal_value) in normal[row].iter_mut().enumerate() {
                        *normal_value += data_first_difference_lambda
                            * basis_row_difference
                            * (basis_pair[1].value_at(col) - basis_pair[0].value_at(col));
                    }
                }
            }
        }

        solve_dense(normal, rhs)
    }

    fn solve_coefficients_banded(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
    ) -> Result<Vec<f64>> {
        let n_bases = self.basis_count();
        let bandwidth = self.degree.max(self.diff_order);
        let mut normal = zero_symmetric_bands(n_bases, bandwidth);
        let mut rhs = vec![0.0; n_bases];

        for ((basis_row, observed), weight) in self.basis.iter().zip(y).zip(weights) {
            for (row, row_value) in basis_row.entries() {
                rhs[row] += row_value * weight * observed;
                for (col, col_value) in basis_row.entries() {
                    if row >= col {
                        add_symmetric_band_value(
                            &mut normal,
                            row,
                            col,
                            row_value * weight * col_value,
                        );
                    }
                }
            }
        }
        for (offset, penalty_band) in self.penalty.iter().enumerate() {
            for (index, value) in penalty_band.iter().enumerate() {
                normal[offset][index] += lambda * value;
            }
        }

        let dense_fallback = (n_bases <= DENSE_COMPAT_THRESHOLD).then(|| normal.clone());
        match solve_spd_banded(&mut normal, &rhs) {
            Ok(solution) => Ok(solution),
            Err(error) => {
                if let Some(bands) = dense_fallback {
                    solve_dense(symmetric_bands_to_dense(&bands), rhs)
                } else {
                    Err(error)
                }
            }
        }
    }

    fn solve_coefficients_banded_into(
        &self,
        y: &[f64],
        weights: &[f64],
        lambda: f64,
        workspace: &mut PenalizedSplineWorkspace,
    ) -> Result<()> {
        let n_bases = self.basis_count();
        let bandwidth = self.degree.max(self.diff_order);
        workspace.reset_symmetric_bands(n_bases, bandwidth);
        workspace.reset_rhs(n_bases);

        for ((basis_row, observed), weight) in self.basis.iter().zip(y).zip(weights) {
            for (row, row_value) in basis_row.entries() {
                workspace.rhs[row] += row_value * weight * observed;
                for (col, col_value) in basis_row.entries() {
                    if row >= col {
                        add_symmetric_band_value(
                            &mut workspace.symmetric_bands,
                            row,
                            col,
                            row_value * weight * col_value,
                        );
                    }
                }
            }
        }
        for (offset, penalty_band) in self.penalty.iter().enumerate() {
            for (index, value) in penalty_band.iter().enumerate() {
                workspace.symmetric_bands[offset][index] += lambda * value;
            }
        }

        match solve_spd_banded_into(
            &mut workspace.symmetric_bands,
            &workspace.rhs,
            &mut workspace.intermediate,
            &mut workspace.coefficients,
        ) {
            Ok(()) => Ok(()),
            Err(error) => {
                if n_bases <= DENSE_COMPAT_THRESHOLD {
                    workspace.coefficients = self.solve_coefficients_dense_with_options(
                        y, weights, lambda, None, 0.0, 0.0,
                    )?;
                    Ok(())
                } else {
                    Err(error)
                }
            }
        }
    }

    fn evaluate_coefficients(&self, coefficients: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.basis.len()];
        self.evaluate_coefficients_into(coefficients, &mut output);
        output
    }

    fn evaluate_coefficients_into(&self, coefficients: &[f64], output: &mut [f64]) {
        for (target, basis_row) in output.iter_mut().zip(&self.basis) {
            *target = basis_row
                .entries()
                .map(|(index, value)| value * coefficients[index])
                .sum();
        }
    }
}

impl SparseBasisRow {
    fn entries(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.values
            .iter()
            .copied()
            .enumerate()
            .map(|(offset, value)| (self.start + offset, value))
    }

    fn value_at(&self, index: usize) -> f64 {
        if index < self.start || index >= self.start + self.values.len() {
            0.0
        } else {
            self.values[index - self.start]
        }
    }

    fn difference_entries_into(
        left: &SparseBasisRow,
        right: &SparseBasisRow,
        entries: &mut Vec<(usize, f64)>,
    ) {
        entries.clear();
        let start = left.start.min(right.start);
        let end = (left.start + left.values.len()).max(right.start + right.values.len());
        for index in start..end {
            let value = right.value_at(index) - left.value_at(index);
            if value != 0.0 {
                entries.push((index, value));
            }
        }
    }
}

fn scaled_domain(n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => (0..n)
            .map(|index| 2.0 * index as f64 / (n - 1) as f64 - 1.0)
            .collect(),
    }
}

fn spline_knots(x: &[f64], num_knots: usize, degree: usize) -> Vec<f64> {
    let num_knots = num_knots.max(2);
    let x_min = *x.first().unwrap_or(&-1.0);
    let x_max = *x.last().unwrap_or(&1.0);
    let dx = (x_max - x_min) / (num_knots - 1) as f64;
    let mut knots = Vec::with_capacity(num_knots + 2 * degree);
    for index in (1..=degree).rev() {
        knots.push(x_min - index as f64 * dx);
    }
    for index in 0..num_knots {
        knots.push(x_min + index as f64 * dx);
    }
    for index in 1..=degree {
        knots.push(x_max + index as f64 * dx);
    }
    knots
}

fn basis_midpoints(knots: &[f64], degree: usize) -> Vec<f64> {
    if degree % 2 == 1 {
        let start = 1 + degree / 2;
        let end = knots.len() - (degree - degree / 2);
        knots[start..end].to_vec()
    } else {
        let midpoints: Vec<f64> = knots
            .windows(2)
            .map(|pair| 0.5 * (pair[0] + pair[1]))
            .collect();
        let start = degree / 2;
        let end = midpoints.len() - degree / 2;
        midpoints[start..end].to_vec()
    }
}

fn interpolate_sample(values: &[f64], point: f64, len: usize) -> f64 {
    if point <= -1.0 {
        return values[0];
    }
    if point >= 1.0 {
        return values[len - 1];
    }

    let position = 0.5 * (point + 1.0) * (len - 1) as f64;
    let left = position.floor() as usize;
    let right = (left + 1).min(len - 1);
    let fraction = position - left as f64;
    values[left] * (1.0 - fraction) + values[right] * fraction
}

fn sparse_basis_row(x: f64, knots: &[f64], degree: usize, n_bases: usize) -> SparseBasisRow {
    let span = knot_span(x, knots, degree, n_bases);
    SparseBasisRow {
        start: span - degree,
        values: basis_values_for_span(x, knots, span, degree),
    }
}

fn knot_span(x: f64, knots: &[f64], degree: usize, n_bases: usize) -> usize {
    let last_basis = n_bases - 1;
    if x >= knots[last_basis + 1] {
        return last_basis;
    }
    if x <= knots[degree] {
        return degree;
    }

    let mut low = degree;
    let mut high = last_basis + 1;
    let mut mid = (low + high) / 2;
    while x < knots[mid] || x >= knots[mid + 1] {
        if x < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

fn basis_values_for_span(x: f64, knots: &[f64], span: usize, degree: usize) -> Vec<f64> {
    let mut values = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    values[0] = 1.0;
    for level in 1..=degree {
        left[level] = x - knots[span + 1 - level];
        right[level] = knots[span + level] - x;
        let mut saved = 0.0;
        for index in 0..level {
            let denominator = right[index + 1] + left[level - index];
            let temp = if denominator.abs() <= f64::EPSILON {
                0.0
            } else {
                values[index] / denominator
            };
            values[index] = saved + right[index + 1] * temp;
            saved = left[level - index] * temp;
        }
        values[level] = saved;
    }
    values
}

fn difference_penalty_bands(n_bases: usize, diff_order: usize) -> Vec<Vec<f64>> {
    let rows = n_bases.saturating_sub(diff_order);
    let mut penalty = zero_symmetric_bands(n_bases, diff_order);
    let coefficients = difference_coefficients(diff_order);
    for row in 0..rows {
        for (left_offset, left) in coefficients.iter().enumerate() {
            for (right_offset, right) in coefficients[..=left_offset].iter().enumerate() {
                add_symmetric_band_value(
                    &mut penalty,
                    row + left_offset,
                    row + right_offset,
                    left * right,
                );
            }
        }
    }
    penalty
}

fn zero_symmetric_bands(n: usize, bandwidth: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; bandwidth + 1]
}

fn zero_general_bands(n: usize, bandwidth: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; n]; 2 * bandwidth + 1]
}

fn general_band_index(bandwidth: usize, row: usize, col: usize) -> Option<usize> {
    let offset = col as isize - row as isize;
    if offset.unsigned_abs() <= bandwidth {
        Some((offset + bandwidth as isize) as usize)
    } else {
        None
    }
}

fn add_general_band_value(
    bands: &mut [Vec<f64>],
    bandwidth: usize,
    row: usize,
    col: usize,
    value: f64,
) {
    debug_assert!(
        general_band_index(bandwidth, row, col).is_some(),
        "column {col} is outside bandwidth {bandwidth} for row {row}",
    );
    if let Some(index) = general_band_index(bandwidth, row, col) {
        bands[index][row] += value;
    }
}

fn set_general_band_value(
    bands: &mut [Vec<f64>],
    bandwidth: usize,
    row: usize,
    col: usize,
    value: f64,
) {
    debug_assert!(
        general_band_index(bandwidth, row, col).is_some(),
        "column {col} is outside bandwidth {bandwidth} for row {row}",
    );
    if let Some(index) = general_band_index(bandwidth, row, col) {
        bands[index][row] = value;
    }
}

fn general_band_value(bands: &[Vec<f64>], bandwidth: usize, row: usize, col: usize) -> f64 {
    general_band_index(bandwidth, row, col).map_or(0.0, |index| bands[index][row])
}

fn add_symmetric_band_value(bands: &mut [Vec<f64>], row: usize, col: usize, value: f64) {
    let (lower, upper) = if row >= col { (row, col) } else { (col, row) };
    let offset = lower - upper;
    debug_assert!(
        offset < bands.len(),
        "band offset {offset} exceeds bandwidth {}",
        bands.len() - 1
    );
    if offset < bands.len() {
        bands[offset][lower] += value;
    }
}

fn set_symmetric_band_value(bands: &mut [Vec<f64>], row: usize, col: usize, value: f64) {
    let (lower, upper) = if row >= col { (row, col) } else { (col, row) };
    let offset = lower - upper;
    debug_assert!(
        offset < bands.len(),
        "band offset {offset} exceeds bandwidth {}",
        bands.len() - 1
    );
    if offset < bands.len() {
        bands[offset][lower] = value;
    }
}

fn symmetric_band_value(bands: &[Vec<f64>], row: usize, col: usize) -> f64 {
    let (lower, upper) = if row >= col { (row, col) } else { (col, row) };
    let offset = lower - upper;
    if offset < bands.len() {
        bands[offset][lower]
    } else {
        0.0
    }
}

fn symmetric_bands_to_dense(bands: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = bands[0].len();
    let mut matrix = vec![vec![0.0; n]; n];
    let mut entries = Vec::with_capacity(n * bands.len());
    for row in 0..n {
        for col in row.saturating_sub(bands.len() - 1)..=row {
            entries.push((row, col, symmetric_band_value(bands, row, col)));
        }
    }
    for (row, col, value) in entries {
        matrix[row][col] = value;
        matrix[col][row] = value;
    }
    matrix
}

fn solve_general_banded(bands: &mut [Vec<f64>], bandwidth: usize, rhs: &[f64]) -> Result<Vec<f64>> {
    let n = rhs.len();
    let mut rhs = rhs.to_vec();
    for pivot in 0..n {
        let pivot_value = general_band_value(bands, bandwidth, pivot, pivot);
        if pivot_value.abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "singular banded system",
            });
        }
        for row in pivot + 1..=(pivot + bandwidth).min(n - 1) {
            let factor = general_band_value(bands, bandwidth, row, pivot) / pivot_value;
            if factor == 0.0 {
                continue;
            }
            set_general_band_value(bands, bandwidth, row, pivot, 0.0);
            for col in pivot + 1..=(pivot + bandwidth).min(n - 1) {
                let value = general_band_value(bands, bandwidth, row, col)
                    - factor * general_band_value(bands, bandwidth, pivot, col);
                set_general_band_value(bands, bandwidth, row, col, value);
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut output = vec![0.0; n];
    for row in (0..n).rev() {
        let tail = (row + 1..=(row + bandwidth).min(n - 1))
            .map(|col| general_band_value(bands, bandwidth, row, col) * output[col])
            .sum::<f64>();
        let diag = general_band_value(bands, bandwidth, row, row);
        if diag.abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "singular banded system",
            });
        }
        output[row] = (rhs[row] - tail) / diag;
    }
    Ok(output)
}

fn solve_spd_banded(bands: &mut [Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let mut intermediate = Vec::new();
    let mut output = Vec::new();
    solve_spd_banded_into(bands, rhs, &mut intermediate, &mut output)?;
    Ok(output)
}

fn solve_spd_banded_into(
    bands: &mut [Vec<f64>],
    rhs: &[f64],
    intermediate: &mut Vec<f64>,
    output: &mut Vec<f64>,
) -> Result<()> {
    let n = rhs.len();
    let bandwidth = bands.len() - 1;
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        for col in start..row {
            let mut value = symmetric_band_value(bands, row, col);
            let sum_start = start.max(col.saturating_sub(bandwidth));
            for mid in sum_start..col {
                value -=
                    symmetric_band_value(bands, row, mid) * symmetric_band_value(bands, col, mid);
            }
            let col_diag = symmetric_band_value(bands, col, col);
            if col_diag.abs() <= f64::EPSILON {
                return Err(BaselineError::LinearSolve {
                    reason: "singular banded Cholesky factor",
                });
            }
            set_symmetric_band_value(bands, row, col, value / col_diag);
        }

        let mut diag = symmetric_band_value(bands, row, row);
        for col in start..row {
            let value = symmetric_band_value(bands, row, col);
            diag -= value * value;
        }
        if diag <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "matrix was not positive definite",
            });
        }
        set_symmetric_band_value(bands, row, row, diag.sqrt());
    }

    intermediate.resize(n, 0.0);
    intermediate.fill(0.0);
    for row in 0..n {
        let start = row.saturating_sub(bandwidth);
        let tail = (start..row)
            .map(|col| symmetric_band_value(bands, row, col) * intermediate[col])
            .sum::<f64>();
        intermediate[row] = (rhs[row] - tail) / symmetric_band_value(bands, row, row);
    }

    output.resize(n, 0.0);
    output.fill(0.0);
    for row in (0..n).rev() {
        let end = (row + bandwidth).min(n - 1);
        let tail = (row + 1..=end)
            .map(|lower| symmetric_band_value(bands, lower, row) * output[lower])
            .sum::<f64>();
        output[row] = (intermediate[row] - tail) / symmetric_band_value(bands, row, row);
    }
    Ok(())
}

fn difference_coefficients(order: usize) -> Vec<f64> {
    let mut coefficients = vec![1.0];
    for _ in 0..order {
        let mut next = vec![0.0; coefficients.len() + 1];
        for (index, coefficient) in coefficients.iter().enumerate() {
            next[index] -= coefficient;
            next[index + 1] += coefficient;
        }
        coefficients = next;
    }
    coefficients
}
