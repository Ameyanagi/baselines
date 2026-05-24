//! Symmetric positive-definite pentadiagonal solver.

use crate::{BaselineError, Result};

/// Reusable storage for a second-order Whittaker system.
#[derive(Debug, Clone)]
pub struct PentadiagonalWorkspace {
    diag: Vec<f64>,
    sub1: Vec<f64>,
    sub2: Vec<f64>,
    chol_diag: Vec<f64>,
    chol_sub1: Vec<f64>,
    chol_sub2: Vec<f64>,
    tmp: Vec<f64>,
}

/// Reusable storage for a general nonsymmetric pentadiagonal system.
#[derive(Debug, Clone)]
pub struct GeneralPentadiagonalWorkspace {
    lower2: Vec<f64>,
    lower1: Vec<f64>,
    diag: Vec<f64>,
    upper1: Vec<f64>,
    upper2: Vec<f64>,
    rhs: Vec<f64>,
}

/// Borrowed diagonals for a general nonsymmetric pentadiagonal system.
pub(crate) struct GeneralPentadiagonalSystem<'a> {
    /// Second lower diagonal, where `lower2[i] = A[i + 2, i]`.
    pub lower2: &'a [f64],
    /// First lower diagonal, where `lower1[i] = A[i + 1, i]`.
    pub lower1: &'a [f64],
    /// Main diagonal.
    pub diag: &'a [f64],
    /// First upper diagonal, where `upper1[i] = A[i, i + 1]`.
    pub upper1: &'a [f64],
    /// Second upper diagonal, where `upper2[i] = A[i, i + 2]`.
    pub upper2: &'a [f64],
}

impl GeneralPentadiagonalWorkspace {
    /// Creates a workspace for `n` samples.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut workspace = Self {
            lower2: Vec::new(),
            lower1: Vec::new(),
            diag: Vec::new(),
            upper1: Vec::new(),
            upper2: Vec::new(),
            rhs: Vec::new(),
        };
        workspace.resize(n);
        workspace
    }

    /// Resizes buffers for `n` samples.
    pub fn resize(&mut self, n: usize) {
        self.lower2.resize(n.saturating_sub(2), 0.0);
        self.lower1.resize(n.saturating_sub(1), 0.0);
        self.diag.resize(n, 0.0);
        self.upper1.resize(n.saturating_sub(1), 0.0);
        self.upper2.resize(n.saturating_sub(2), 0.0);
        self.rhs.resize(n, 0.0);
    }
}

impl PentadiagonalWorkspace {
    /// Creates a workspace for `n` samples.
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut workspace = Self {
            diag: Vec::new(),
            sub1: Vec::new(),
            sub2: Vec::new(),
            chol_diag: Vec::new(),
            chol_sub1: Vec::new(),
            chol_sub2: Vec::new(),
            tmp: Vec::new(),
        };
        workspace.resize(n);
        workspace
    }

    /// Resizes buffers for `n` samples.
    pub fn resize(&mut self, n: usize) {
        self.diag.resize(n, 0.0);
        self.sub1.resize(n.saturating_sub(1), 0.0);
        self.sub2.resize(n.saturating_sub(2), 0.0);
        self.chol_diag.resize(n, 0.0);
        self.chol_sub1.resize(n, 0.0);
        self.chol_sub2.resize(n, 0.0);
        self.tmp.resize(n, 0.0);
    }
}

/// Solves `(W + lambda * D'D) z = W y` for second-order differences.
pub fn solve_second_order(
    y: &[f64],
    weights: &[f64],
    lambda: f64,
    baseline: &mut [f64],
    workspace: &mut PentadiagonalWorkspace,
) -> Result<()> {
    let n = y.len();
    if n != weights.len() {
        return Err(BaselineError::LengthMismatch {
            name: "weights",
            expected: n,
            actual: weights.len(),
        });
    }
    if n != baseline.len() {
        return Err(BaselineError::LengthMismatch {
            name: "baseline",
            expected: n,
            actual: baseline.len(),
        });
    }
    if n == 0 {
        return Err(BaselineError::EmptyInput);
    }
    if n < 3 {
        for ((output, observed), weight) in baseline.iter_mut().zip(y).zip(weights) {
            *output = if *weight > 0.0 { *observed } else { 0.0 };
        }
        return Ok(());
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "lambda",
            reason: "must be finite and positive",
        });
    }

    workspace.resize(n);
    fill_second_order_bands(
        weights,
        lambda,
        &mut workspace.diag,
        &mut workspace.sub1,
        &mut workspace.sub2,
    );

    for (rhs, (observed, weight)) in workspace.tmp.iter_mut().zip(y.iter().zip(weights)) {
        *rhs = observed * weight;
    }

    factor_and_solve(baseline, workspace)
}

/// Solves `(W + lambda * D(x)'D(x)) z = W y` for second-order differences.
pub(crate) fn solve_second_order_x(
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    lambda: f64,
    baseline: &mut [f64],
    workspace: &mut PentadiagonalWorkspace,
) -> Result<()> {
    let n = y.len();
    if n != x.len() {
        return Err(BaselineError::LengthMismatch {
            name: "x",
            expected: n,
            actual: x.len(),
        });
    }
    if n != weights.len() {
        return Err(BaselineError::LengthMismatch {
            name: "weights",
            expected: n,
            actual: weights.len(),
        });
    }
    if n != baseline.len() {
        return Err(BaselineError::LengthMismatch {
            name: "baseline",
            expected: n,
            actual: baseline.len(),
        });
    }
    if n < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "whittaker",
            len: n,
            min: 3,
        });
    }
    validate_positive_lambda("lambda", lambda)?;

    workspace.resize(n);
    fill_second_order_x_bands(
        x,
        weights,
        lambda,
        &mut workspace.diag,
        &mut workspace.sub1,
        &mut workspace.sub2,
    )?;

    for (rhs, (observed, weight)) in workspace.tmp.iter_mut().zip(y.iter().zip(weights)) {
        *rhs = observed * weight;
    }

    factor_and_solve(baseline, workspace)
}

/// Solves a second-order Whittaker system with an added first-order penalty.
pub(crate) fn solve_second_order_with_first_order(
    diagonal: &[f64],
    rhs: &[f64],
    lambda_second: f64,
    lambda_first: f64,
    baseline: &mut [f64],
    workspace: &mut PentadiagonalWorkspace,
) -> Result<()> {
    let n = diagonal.len();
    if n != rhs.len() {
        return Err(BaselineError::LengthMismatch {
            name: "rhs",
            expected: n,
            actual: rhs.len(),
        });
    }
    if n != baseline.len() {
        return Err(BaselineError::LengthMismatch {
            name: "baseline",
            expected: n,
            actual: baseline.len(),
        });
    }
    if n < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "whittaker",
            len: n,
            min: 3,
        });
    }
    validate_positive_lambda("lambda_second", lambda_second)?;
    validate_positive_lambda("lambda_first", lambda_first)?;

    workspace.resize(n);
    fill_second_order_bands(
        diagonal,
        lambda_second,
        &mut workspace.diag,
        &mut workspace.sub1,
        &mut workspace.sub2,
    );
    add_first_order_penalty(lambda_first, &mut workspace.diag, &mut workspace.sub1);
    workspace.tmp.copy_from_slice(rhs);

    factor_and_solve(baseline, workspace)
}

/// Solves an x-aware second-order Whittaker system with an added first-order penalty.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_second_order_x_with_first_order(
    x: &[f64],
    diagonal: &[f64],
    rhs: &[f64],
    active_mask: Option<&[bool]>,
    lambda_second: f64,
    lambda_first: f64,
    baseline: &mut [f64],
    workspace: &mut PentadiagonalWorkspace,
) -> Result<()> {
    let n = diagonal.len();
    if n != x.len() {
        return Err(BaselineError::LengthMismatch {
            name: "x",
            expected: n,
            actual: x.len(),
        });
    }
    if n != rhs.len() {
        return Err(BaselineError::LengthMismatch {
            name: "rhs",
            expected: n,
            actual: rhs.len(),
        });
    }
    if n != baseline.len() {
        return Err(BaselineError::LengthMismatch {
            name: "baseline",
            expected: n,
            actual: baseline.len(),
        });
    }
    if let Some(active_mask) = active_mask
        && n != active_mask.len()
    {
        return Err(BaselineError::LengthMismatch {
            name: "active_mask",
            expected: n,
            actual: active_mask.len(),
        });
    }
    if n < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "whittaker",
            len: n,
            min: 3,
        });
    }
    validate_positive_lambda("lambda_second", lambda_second)?;
    validate_positive_lambda("lambda_first", lambda_first)?;

    workspace.resize(n);
    fill_second_order_x_bands(
        x,
        diagonal,
        lambda_second,
        &mut workspace.diag,
        &mut workspace.sub1,
        &mut workspace.sub2,
    )?;
    add_first_order_x_penalty(
        x,
        active_mask,
        lambda_first,
        &mut workspace.diag,
        &mut workspace.sub1,
    )?;
    workspace.tmp.copy_from_slice(rhs);

    factor_and_solve(baseline, workspace)
}

/// Solves a nonsymmetric pentadiagonal system with no pivoting.
pub(crate) fn solve_general_pentadiagonal(
    system: GeneralPentadiagonalSystem<'_>,
    rhs: &[f64],
    output: &mut [f64],
    workspace: &mut GeneralPentadiagonalWorkspace,
) -> Result<()> {
    let n = system.diag.len();
    validate_general_band_lengths(&system, rhs, output)?;
    workspace.resize(n);
    workspace.lower2.copy_from_slice(system.lower2);
    workspace.lower1.copy_from_slice(system.lower1);
    workspace.diag.copy_from_slice(system.diag);
    workspace.upper1.copy_from_slice(system.upper1);
    workspace.upper2.copy_from_slice(system.upper2);
    workspace.rhs.copy_from_slice(rhs);

    for index in 0..n {
        let pivot = workspace.diag[index];
        if !pivot.is_finite() || pivot.abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "matrix pivot was zero",
            });
        }

        if index + 1 < n {
            let factor = workspace.lower1[index] / pivot;
            workspace.diag[index + 1] -= factor * workspace.upper1[index];
            if index + 2 < n {
                workspace.upper1[index + 1] -= factor * workspace.upper2[index];
            }
            workspace.rhs[index + 1] -= factor * workspace.rhs[index];
        }

        if index + 2 < n {
            let factor = workspace.lower2[index] / pivot;
            workspace.lower1[index + 1] -= factor * workspace.upper1[index];
            workspace.diag[index + 2] -= factor * workspace.upper2[index];
            workspace.rhs[index + 2] -= factor * workspace.rhs[index];
        }
    }

    for index in (0..n).rev() {
        let mut value = workspace.rhs[index];
        if index + 1 < n {
            value -= workspace.upper1[index] * output[index + 1];
        }
        if index + 2 < n {
            value -= workspace.upper2[index] * output[index + 2];
        }
        output[index] = value / workspace.diag[index];
    }

    Ok(())
}

fn factor_and_solve(baseline: &mut [f64], workspace: &mut PentadiagonalWorkspace) -> Result<()> {
    cholesky_factor(
        &workspace.diag,
        &workspace.sub1,
        &workspace.sub2,
        &mut workspace.chol_diag,
        &mut workspace.chol_sub1,
        &mut workspace.chol_sub2,
    )?;
    cholesky_solve(
        &workspace.chol_diag,
        &workspace.chol_sub1,
        &workspace.chol_sub2,
        &mut workspace.tmp,
    );
    baseline.copy_from_slice(&workspace.tmp);
    Ok(())
}

fn validate_general_band_lengths(
    system: &GeneralPentadiagonalSystem<'_>,
    rhs: &[f64],
    output: &[f64],
) -> Result<()> {
    let n = system.diag.len();
    if n == 0 {
        return Err(BaselineError::EmptyInput);
    }
    let expected_lower2 = n.saturating_sub(2);
    let expected_lower1 = n.saturating_sub(1);
    for (name, expected, actual) in [
        ("lower2", expected_lower2, system.lower2.len()),
        ("lower1", expected_lower1, system.lower1.len()),
        ("upper1", expected_lower1, system.upper1.len()),
        ("upper2", expected_lower2, system.upper2.len()),
        ("rhs", n, rhs.len()),
        ("output", n, output.len()),
    ] {
        if expected != actual {
            return Err(BaselineError::LengthMismatch {
                name,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

fn validate_positive_lambda(name: &'static str, lambda: f64) -> Result<()> {
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name,
            reason: "must be finite and positive",
        });
    }
    Ok(())
}

fn fill_second_order_bands(
    weights: &[f64],
    lambda: f64,
    diag: &mut [f64],
    sub1: &mut [f64],
    sub2: &mut [f64],
) {
    let n = weights.len();
    for (target, weight) in diag.iter_mut().zip(weights) {
        *target = *weight;
    }

    diag[0] += lambda;
    diag[1] += 5.0 * lambda;
    for value in &mut diag[2..n - 2] {
        *value += 6.0 * lambda;
    }
    diag[n - 2] += 5.0 * lambda;
    diag[n - 1] += lambda;

    sub1[0] = -2.0 * lambda;
    for value in &mut sub1[1..n - 2] {
        *value = -4.0 * lambda;
    }
    sub1[n - 2] = -2.0 * lambda;

    for value in sub2 {
        *value = lambda;
    }
}

pub(crate) fn fill_second_order_x_bands(
    x: &[f64],
    weights: &[f64],
    lambda: f64,
    diag: &mut [f64],
    sub1: &mut [f64],
    sub2: &mut [f64],
) -> Result<()> {
    let n = weights.len();
    for (target, weight) in diag.iter_mut().zip(weights) {
        *target = *weight;
    }
    sub1.fill(0.0);
    sub2.fill(0.0);

    for row in 0..n - 2 {
        let h0 = x[row + 1] - x[row];
        let h1 = x[row + 2] - x[row + 1];
        if !h0.is_finite() || !h1.is_finite() || h0 <= 0.0 || h1 <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "x",
                reason: "must be finite and strictly increasing",
            });
        }
        let sum = h0 + h1;
        let a = 2.0 / (h0 * sum);
        let b = -2.0 / (h0 * h1);
        let c = 2.0 / (h1 * sum);
        let scale = lambda;

        diag[row] += scale * a * a;
        diag[row + 1] += scale * b * b;
        diag[row + 2] += scale * c * c;
        sub1[row] += scale * a * b;
        sub1[row + 1] += scale * b * c;
        sub2[row] += scale * a * c;
    }
    Ok(())
}

fn add_first_order_penalty(lambda: f64, diag: &mut [f64], sub1: &mut [f64]) {
    let n = diag.len();
    diag[0] += lambda;
    diag[n - 1] += lambda;
    for value in &mut diag[1..n - 1] {
        *value += 2.0 * lambda;
    }
    for value in sub1 {
        *value -= lambda;
    }
}

pub(crate) fn add_first_order_x_penalty(
    x: &[f64],
    active_mask: Option<&[bool]>,
    lambda: f64,
    diag: &mut [f64],
    sub1: &mut [f64],
) -> Result<()> {
    for row in 0..x.len() - 1 {
        if active_mask.is_some_and(|mask| !mask[row] || !mask[row + 1]) {
            continue;
        }
        let h = x[row + 1] - x[row];
        if !h.is_finite() || h <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "x",
                reason: "must be finite and strictly increasing",
            });
        }
        let left = -1.0 / h;
        let right = 1.0 / h;
        diag[row] += lambda * left * left;
        diag[row + 1] += lambda * right * right;
        sub1[row] += lambda * left * right;
    }
    Ok(())
}

pub(crate) fn first_order_x_penalty_rhs(
    x: &[f64],
    y: &[f64],
    active_mask: Option<&[bool]>,
    lambda: f64,
    output: &mut [f64],
) -> Result<()> {
    if x.len() != y.len() {
        return Err(BaselineError::LengthMismatch {
            name: "x",
            expected: y.len(),
            actual: x.len(),
        });
    }
    if output.len() != y.len() {
        return Err(BaselineError::LengthMismatch {
            name: "output",
            expected: y.len(),
            actual: output.len(),
        });
    }
    output.fill(0.0);
    for row in 0..x.len() - 1 {
        if active_mask.is_some_and(|mask| !mask[row] || !mask[row + 1]) {
            continue;
        }
        let h = x[row + 1] - x[row];
        if !h.is_finite() || h <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "x",
                reason: "must be finite and strictly increasing",
            });
        }
        let left = -1.0 / h;
        let right = 1.0 / h;
        let dy = left * y[row] + right * y[row + 1];
        output[row] += lambda * left * dy;
        output[row + 1] += lambda * right * dy;
    }
    Ok(())
}

fn cholesky_factor(
    diag: &[f64],
    sub1: &[f64],
    sub2: &[f64],
    chol_diag: &mut [f64],
    chol_sub1: &mut [f64],
    chol_sub2: &mut [f64],
) -> Result<()> {
    let n = diag.len();
    chol_sub1.fill(0.0);
    chol_sub2.fill(0.0);

    for i in 0..n {
        if i >= 2 {
            chol_sub2[i] = sub2[i - 2] / chol_diag[i - 2];
        }
        if i >= 1 {
            let correction = if i >= 2 {
                chol_sub2[i] * chol_sub1[i - 1]
            } else {
                0.0
            };
            chol_sub1[i] = (sub1[i - 1] - correction) / chol_diag[i - 1];
        }
        let value = diag[i] - chol_sub1[i] * chol_sub1[i] - chol_sub2[i] * chol_sub2[i];
        if !value.is_finite() || value <= 0.0 {
            return Err(BaselineError::LinearSolve {
                reason: "matrix was not positive definite",
            });
        }
        chol_diag[i] = value.sqrt();
    }
    Ok(())
}

fn cholesky_solve(diag: &[f64], sub1: &[f64], sub2: &[f64], rhs: &mut [f64]) {
    let n = rhs.len();
    for i in 0..n {
        let mut value = rhs[i];
        if i >= 1 {
            value -= sub1[i] * rhs[i - 1];
        }
        if i >= 2 {
            value -= sub2[i] * rhs[i - 2];
        }
        rhs[i] = value / diag[i];
    }

    for i in (0..n).rev() {
        let mut value = rhs[i];
        if i + 1 < n {
            value -= sub1[i + 1] * rhs[i + 1];
        }
        if i + 2 < n {
            value -= sub2[i + 2] * rhs[i + 2];
        }
        rhs[i] = value / diag[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_signal_is_preserved() {
        let y = vec![3.0; 20];
        let weights = vec![1.0; y.len()];
        let mut baseline = vec![0.0; y.len()];
        let mut workspace = PentadiagonalWorkspace::new(y.len());
        solve_second_order(&y, &weights, 1e5, &mut baseline, &mut workspace).unwrap();

        for value in baseline {
            assert!((value - 3.0).abs() < 1e-8);
        }
    }
}
