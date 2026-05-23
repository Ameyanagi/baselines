//! Shared Whittaker iteration engine.

use crate::fit::{Fit, FitHistory, FitReport};
use crate::linalg::pentadiagonal::{PentadiagonalWorkspace, solve_second_order};
use crate::workspace::{IterWorkspace, validate_output, validate_signal};
use crate::{BaselineError, Result};

/// Common parameters for Whittaker-style baseline algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhittakerParams {
    /// Smoothness penalty. Larger values produce smoother baselines.
    pub lambda: f64,
    /// Maximum number of reweighting iterations.
    pub max_iter: usize,
    /// Relative weight-change tolerance.
    ///
    /// Values less than or equal to zero disable early convergence and force
    /// `max_iter` iterations, matching pybaselines examples that use
    /// `tol=-1`.
    pub tol: f64,
}

impl Default for WhittakerParams {
    fn default() -> Self {
        Self {
            lambda: 1.0e6,
            max_iter: 50,
            tol: 1.0e-3,
        }
    }
}

impl WhittakerParams {
    /// Validates common Whittaker parameters.
    pub fn validate(&self) -> Result<()> {
        if !self.lambda.is_finite() || self.lambda <= 0.0 {
            return Err(BaselineError::InvalidParameter {
                name: "lambda",
                reason: "must be finite and positive",
            });
        }
        if self.max_iter == 0 {
            return Err(BaselineError::InvalidParameter {
                name: "max_iter",
                reason: "must be greater than zero",
            });
        }
        if !self.tol.is_finite() {
            return Err(BaselineError::InvalidParameter {
                name: "tol",
                reason: "must be finite",
            });
        }
        Ok(())
    }
}

/// Workspace for Whittaker algorithms.
#[derive(Debug, Clone)]
pub struct WhittakerWorkspace {
    /// Iterative algorithm buffers.
    pub iter: IterWorkspace,
    /// Pentadiagonal solver buffers.
    pub solver: PentadiagonalWorkspace,
}

impl WhittakerWorkspace {
    /// Creates a workspace for `n` samples.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            iter: IterWorkspace::new(n),
            solver: PentadiagonalWorkspace::new(n),
        }
    }

    /// Resizes the workspace to `n` samples.
    pub fn resize(&mut self, n: usize) {
        self.iter.resize(n);
        self.solver.resize(n);
    }
}

/// Policy used by the Whittaker IRLS engine.
pub trait Reweighter {
    /// Initializes weights before the first solve.
    fn initialize(&self, y: &[f64], weights: &mut [f64]);

    /// Updates weights and returns a convergence metric.
    fn update(&self, y: &[f64], baseline: &[f64], weights: &mut [f64], iter: usize) -> f64;
}

/// Fits a Whittaker baseline and allocates the output vector.
pub fn fit_alloc<R: Reweighter>(y: &[f64], params: WhittakerParams, reweighter: R) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let mut workspace = WhittakerWorkspace::new(y.len());
    let report = fit_into(y, params, reweighter, &mut baseline, &mut workspace)?;
    Ok(Fit { baseline, report })
}

/// Fits a Whittaker baseline and returns per-iteration tolerance history.
pub fn fit_alloc_with_history<R: Reweighter>(
    y: &[f64],
    params: WhittakerParams,
    reweighter: R,
) -> Result<FitHistory> {
    let mut baseline = vec![0.0; y.len()];
    let mut workspace = WhittakerWorkspace::new(y.len());
    let mut tol_history = Vec::with_capacity(params.max_iter);
    let report = fit_into_with_history(
        y,
        params,
        reweighter,
        &mut baseline,
        &mut workspace,
        &mut tol_history,
    )?;
    Ok(FitHistory {
        baseline,
        report,
        tol_history,
    })
}

/// Fits a Whittaker baseline into an existing output buffer.
pub fn fit_into<R: Reweighter>(
    y: &[f64],
    params: WhittakerParams,
    reweighter: R,
    baseline: &mut [f64],
    workspace: &mut WhittakerWorkspace,
) -> Result<FitReport> {
    fit_into_impl(y, params, reweighter, baseline, workspace, None)
}

/// Fits a Whittaker baseline into an existing output buffer and records tolerance history.
pub fn fit_into_with_history<R: Reweighter>(
    y: &[f64],
    params: WhittakerParams,
    reweighter: R,
    baseline: &mut [f64],
    workspace: &mut WhittakerWorkspace,
    tol_history: &mut Vec<f64>,
) -> Result<FitReport> {
    fit_into_impl(
        y,
        params,
        reweighter,
        baseline,
        workspace,
        Some(tol_history),
    )
}

fn fit_into_impl<R: Reweighter>(
    y: &[f64],
    params: WhittakerParams,
    reweighter: R,
    baseline: &mut [f64],
    workspace: &mut WhittakerWorkspace,
    mut tol_history: Option<&mut Vec<f64>>,
) -> Result<FitReport> {
    validate_signal(y)?;
    validate_output("baseline", y.len(), baseline.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "whittaker",
            len: y.len(),
            min: 3,
        });
    }
    params.validate()?;
    workspace.resize(y.len());
    reweighter.initialize(y, &mut workspace.iter.weights);
    if let Some(history) = tol_history.as_deref_mut() {
        history.clear();
    }

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.max_iter {
        workspace
            .iter
            .previous_weights
            .copy_from_slice(&workspace.iter.weights);

        solve_second_order(
            y,
            &workspace.iter.weights,
            params.lambda,
            baseline,
            &mut workspace.solver,
        )?;

        tolerance = reweighter.update(y, baseline, &mut workspace.iter.weights, iter);
        if let Some(history) = tol_history.as_deref_mut() {
            history.push(tolerance);
        }
        if tolerance <= params.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(params.max_iter, false, tolerance))
}

/// Computes relative L2 change between two slices.
#[must_use]
pub fn relative_change(previous: &[f64], current: &[f64]) -> f64 {
    let numerator = previous
        .iter()
        .zip(current)
        .map(|(old, new)| {
            let diff = new - old;
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
