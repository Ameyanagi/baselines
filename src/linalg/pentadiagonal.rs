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
