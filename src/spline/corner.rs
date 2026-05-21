//! Corner-cutting spline baseline construction.

use crate::fit::{Fit, FitReport};
use crate::workspace::validate_signal;
use crate::{BaselineError, Result};

/// Parameters for corner-cutting baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CornerCuttingParams {
    /// Maximum number of corner-removal iterations.
    pub max_iter: usize,
}

impl Default for CornerCuttingParams {
    fn default() -> Self {
        Self { max_iter: 100 }
    }
}

impl CornerCuttingParams {
    fn validate(&self) -> Result<()> {
        if self.max_iter == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "max_iter",
                reason: "must be greater than zero",
            });
        }
        Ok(())
    }
}

/// Fits a corner-cutting baseline.
///
/// # References
///
/// - Y.-J. Liu et al., "A Concise Iterative Method with Bezier Technique for
///   Baseline Construction", *Analyst*, 2015.
/// - `pybaselines.Baseline.corner_cutting` is used as a behavioral reference.
pub fn corner_cutting(y: &[f64], params: CornerCuttingParams) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    if y.len() < 2 {
        return Err(BaselineError::TooShort {
            algorithm: "corner_cutting",
            len: y.len(),
            min: 2,
        });
    }

    let x = scaled_domain(y.len());
    let mut mask = vec![true; y.len()];
    let mut kept_points = vec![0usize; y.len()];
    let mut areas = Vec::with_capacity(params.max_iter);
    let mut old_area = trapezoid(y, &x);
    let mut old_sum = y.len();

    for _ in 0..params.max_iter {
        let active: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(index, keep)| keep.then_some(index))
            .collect();
        if active.len() <= 2 {
            break;
        }

        let mut new_mask = vec![true; active.len()];
        for local in 1..active.len() - 1 {
            let left = active[local - 1];
            let center = active[local];
            let right = active[local + 1];
            let line =
                y[left] + (x[center] - x[left]) * (y[right] - y[left]) / (x[right] - x[left]);
            new_mask[local] = y[center] < line;
        }

        for (index, keep) in active.iter().zip(new_mask) {
            mask[*index] = keep;
        }

        let new_sum = mask.iter().filter(|keep| **keep).count();
        let num_corners = old_sum - new_sum;
        if num_corners == 0 {
            break;
        }
        old_sum = new_sum;

        for (count, keep) in kept_points.iter_mut().zip(&mask) {
            if *keep {
                *count += 1;
            }
        }

        let (xm, ym): (Vec<f64>, Vec<f64>) = mask
            .iter()
            .enumerate()
            .filter_map(|(index, keep)| keep.then_some((x[index], y[index])))
            .unzip();
        let area = trapezoid(&ym, &xm);
        areas.push((old_area - area) / num_corners as f64);
        old_area = area;
    }

    let max_area = areas
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(-1, |(index, _)| index as isize - 1);
    let indices: Vec<usize> = kept_points
        .iter()
        .enumerate()
        .filter_map(|(index, count)| ((*count as isize) >= max_area).then_some(index))
        .collect();
    let baseline = quadratic_bezier_spline(&x, y, &indices);

    Ok(Fit {
        baseline,
        report: FitReport::new(areas.len(), true, 0.0),
    })
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

fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    y.windows(2)
        .zip(x.windows(2))
        .map(|(y_pair, x_pair)| 0.5 * (x_pair[1] - x_pair[0]) * (y_pair[1] + y_pair[0]))
        .sum()
}

fn quadratic_bezier_spline(x: &[f64], y: &[f64], indices: &[usize]) -> Vec<f64> {
    match indices.len() {
        0 | 1 => y.to_vec(),
        2 => {
            let left = indices[0];
            let right = indices[1];
            linear_segment(x, y[left], x[left], y[right], x[right], 0, x.len() - 1)
        }
        3 => {
            let left = indices[0];
            let right = indices[2];
            let denominator = (x[right] - x[left]).max(f64::MIN_POSITIVE);
            x.iter()
                .map(|value| {
                    let t = (value - x[left]) / denominator;
                    quadratic_bezier([y[left], y[indices[1]], y[right]], t)
                })
                .collect()
        }
        _ => multi_segment_quadratic_bezier(x, y, indices),
    }
}

fn multi_segment_quadratic_bezier(x: &[f64], y: &[f64], indices: &[usize]) -> Vec<f64> {
    let mut output = vec![0.0; x.len()];
    let center_idx = indices[1];
    let next_idx = indices[2];
    let left_x = x[indices[0]];
    let mut right_idx = nearest_midpoint_index(x, center_idx, next_idx);
    let mut right_x = x[right_idx];
    let mut right_y = interpolate_line(
        x[center_idx],
        y[center_idx],
        x[next_idx],
        y[next_idx],
        right_x,
    );
    fill_bezier_range(
        x,
        &mut output,
        0,
        next_idx,
        left_x,
        right_x,
        [y[indices[0]], y[center_idx], right_y],
    );

    for position in 2..indices.len() - 2 {
        let left_idx = right_idx;
        let left_x = right_x;
        let left_y = right_y;
        let center_idx = indices[position];
        let next_idx = indices[position + 1];
        right_idx = nearest_midpoint_index(x, center_idx, next_idx);
        right_x = x[right_idx];
        if (right_x - left_x).abs() <= f64::EPSILON {
            continue;
        }

        right_y = interpolate_line(
            x[center_idx],
            y[center_idx],
            x[next_idx],
            y[next_idx],
            right_x,
        );
        fill_bezier_range(
            x,
            &mut output,
            left_idx,
            right_idx,
            left_x,
            right_x,
            [left_y, y[center_idx], right_y],
        );
    }

    let last = *indices.last().expect("indices length was checked");
    fill_bezier_range(
        x,
        &mut output,
        right_idx,
        x.len() - 1,
        right_x,
        x[last],
        [right_y, y[indices[indices.len() - 2]], y[last]],
    );
    output
}

fn nearest_midpoint_index(x: &[f64], center_idx: usize, next_idx: usize) -> usize {
    let midpoint = 0.5 * (x[center_idx] + x[next_idx]);
    center_idx
        + x[center_idx..=next_idx]
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (*left - midpoint)
                    .abs()
                    .total_cmp(&(*right - midpoint).abs())
            })
            .map_or(0, |(offset, _)| offset)
}

fn linear_segment(
    x: &[f64],
    left_y: f64,
    left_x: f64,
    right_y: f64,
    right_x: f64,
    start: usize,
    end: usize,
) -> Vec<f64> {
    let denominator = (right_x - left_x).max(f64::MIN_POSITIVE);
    let mut output = vec![0.0; x.len()];
    for (offset, value) in x[start..=end].iter().enumerate() {
        output[start + offset] = left_y + (value - left_x) * (right_y - left_y) / denominator;
    }
    output
}

fn fill_bezier_range(
    x: &[f64],
    output: &mut [f64],
    start: usize,
    end: usize,
    left_x: f64,
    right_x: f64,
    y_points: [f64; 3],
) {
    let denominator = (right_x - left_x).max(f64::MIN_POSITIVE);
    for index in start..=end {
        let t = (x[index] - left_x) / denominator;
        output[index] = quadratic_bezier(y_points, t);
    }
}

fn quadratic_bezier(y_points: [f64; 3], t: f64) -> f64 {
    let one_minus_t = 1.0 - t;
    y_points[0] * one_minus_t * one_minus_t
        + 2.0 * y_points[1] * one_minus_t * t
        + y_points[2] * t * t
}

fn interpolate_line(left_x: f64, left_y: f64, right_x: f64, right_y: f64, x: f64) -> f64 {
    let denominator = (right_x - left_x).max(f64::MIN_POSITIVE);
    left_y + (x - left_x) * (right_y - left_y) / denominator
}
