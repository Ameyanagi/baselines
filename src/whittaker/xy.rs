//! X-aware Whittaker fitting helpers.

use crate::fit::{Fit, FitReport};
use crate::linalg::pentadiagonal::{
    GeneralPentadiagonalSystem, GeneralPentadiagonalWorkspace, add_first_order_x_penalty,
    fill_second_order_x_bands, first_order_x_penalty_rhs, solve_general_pentadiagonal,
    solve_second_order_x, solve_second_order_x_with_first_order,
};
use crate::polynomial::fit_weighted_polynomial;
use crate::whittaker::airpls::{AirPlsParams, AirPlsWeights};
use crate::whittaker::arpls::{ArPlsParams, ArPlsWeights};
use crate::whittaker::asls::{AslsParams, AslsWeights};
use crate::whittaker::engine::{
    WhittakerWorkspace, apply_active_mask, fit_alloc_xy, relative_change,
};
use crate::whittaker::variants::{
    AsPlsParams, BrPlsParams, DerPsalsaParams, DrPlsParams, IarPlsParams, IaslsParams,
    LsrPlsParams, PsalsaParams, asls_weight, aspls_weights_masked, brpls_weights_masked,
    derivative_peak_screening_weights, drpls_weights_masked, standard_deviation,
};
use crate::workspace::{validate_output, validate_signal};
use crate::{BaselineError, Result};

/// Borrowed mask inputs for x-aware Whittaker fits.
#[derive(Debug, Clone, Default)]
pub(crate) struct XyMaskSpec<'a> {
    pub(crate) exclude_ranges: Vec<(f64, f64)>,
    pub(crate) exclude_masks: Vec<&'a [bool]>,
    pub(crate) baseline_masks: Vec<&'a [bool]>,
}

impl<'a> XyMaskSpec<'a> {
    pub(crate) fn exclude_range(&mut self, start: f64, end: f64) {
        self.exclude_ranges.push((start, end));
    }

    pub(crate) fn exclude_ranges<I>(&mut self, ranges: I)
    where
        I: IntoIterator<Item = (f64, f64)>,
    {
        self.exclude_ranges.extend(ranges);
    }

    pub(crate) fn exclude_mask(&mut self, mask: &'a [bool], len: usize) -> Result<()> {
        validate_output("exclude_mask", len, mask.len())?;
        self.exclude_masks.push(mask);
        Ok(())
    }

    pub(crate) fn baseline_mask(&mut self, mask: &'a [bool], len: usize) -> Result<()> {
        validate_output("baseline_mask", len, mask.len())?;
        self.baseline_masks.push(mask);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        self.exclude_ranges.clear();
        self.exclude_masks.clear();
        self.baseline_masks.clear();
    }
}

pub(crate) fn asls_xy(
    x: &[f64],
    y: &[f64],
    params: AslsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    params.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        AslsWeights { p: params.p },
    )
}

pub(crate) fn airpls_xy(
    x: &[f64],
    y: &[f64],
    params: AirPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    params.whittaker.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        AirPlsWeights,
    )
}

pub(crate) fn arpls_xy(
    x: &[f64],
    y: &[f64],
    params: ArPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    params.whittaker.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        ArPlsWeights,
    )
}

pub(crate) fn iarpls_xy(
    x: &[f64],
    y: &[f64],
    params: IarPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    params.whittaker.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        crate::whittaker::variants::IarPlsWeights,
    )
}

pub(crate) fn lsrpls_xy(
    x: &[f64],
    y: &[f64],
    params: LsrPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    params.whittaker.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        crate::whittaker::variants::LsrPlsWeights,
    )
}

pub(crate) fn psalsa_xy(
    x: &[f64],
    y: &[f64],
    params: PsalsaParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    params.validate()?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }
    let prepared = prepare_xy(x, y, masks)?;
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        crate::whittaker::variants::PsalsaWeights { p: params.p, k },
    )
}

pub(crate) fn derpsalsa_xy(
    x: &[f64],
    y: &[f64],
    params: DerPsalsaParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    validate_signal(y)?;
    params.validate()?;
    let k = params.k.unwrap_or_else(|| standard_deviation(y) / 10.0);
    if !k.is_finite() || k <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "k",
            reason: "computed std(y) / 10 must be finite and positive",
        });
    }
    let prepared = prepare_xy(x, y, masks)?;
    let partial_weights = derivative_peak_screening_weights(
        y,
        params.smooth_half_window.unwrap_or(y.len() / 200),
        params.num_smooths,
    );
    fit_alloc_xy(
        &prepared.x,
        y,
        &prepared.active,
        params.whittaker,
        crate::whittaker::variants::DerPsalsaWeights {
            p: params.p,
            k,
            partial_weights,
        },
    )
}

pub(crate) fn iasls_xy(
    x: &[f64],
    y: &[f64],
    params: IaslsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = iasls_xy_into(x, y, params, masks, &mut baseline)?;
    Ok(Fit { baseline, report })
}

pub(crate) fn drpls_xy(
    x: &[f64],
    y: &[f64],
    params: DrPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = drpls_xy_into(x, y, params, masks, &mut baseline)?;
    Ok(Fit { baseline, report })
}

pub(crate) fn aspls_xy(
    x: &[f64],
    y: &[f64],
    params: AsPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = aspls_xy_into(x, y, params, masks, &mut baseline)?;
    Ok(Fit { baseline, report })
}

pub(crate) fn brpls_xy(
    x: &[f64],
    y: &[f64],
    params: BrPlsParams,
    masks: &XyMaskSpec<'_>,
) -> Result<Fit> {
    let mut baseline = vec![0.0; y.len()];
    let report = brpls_xy_into(x, y, params, masks, &mut baseline)?;
    Ok(Fit { baseline, report })
}

fn iasls_xy_into(
    x: &[f64],
    y: &[f64],
    params: IaslsParams,
    masks: &XyMaskSpec<'_>,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    let n = y.len();
    let mut workspace = WhittakerWorkspace::new(n);
    let initial_weights: Vec<f64> = prepared
        .active
        .iter()
        .map(|active| if *active { 1.0 } else { 0.0 })
        .collect();
    if prepared.active_count >= 3 {
        fit_weighted_polynomial(y, &initial_weights, 2, &mut workspace.iter.residual)?;
    } else {
        workspace.iter.residual.copy_from_slice(y);
    }
    for (index, ((weight, observed), fitted)) in workspace
        .iter
        .weights
        .iter_mut()
        .zip(y)
        .zip(&workspace.iter.residual)
        .enumerate()
    {
        *weight = if prepared.active[index] {
            asls_weight(*observed, *fitted, params.p)
        } else {
            0.0
        };
    }

    let mut first_order_rhs = vec![0.0; n];
    first_order_x_penalty_rhs(
        &prepared.x,
        y,
        Some(&prepared.active),
        params.lambda_1,
        &mut first_order_rhs,
    )?;

    let mut tolerance = f64::INFINITY;
    for iter in 0..=params.whittaker.max_iter {
        workspace
            .iter
            .previous_weights
            .copy_from_slice(&workspace.iter.weights);

        for (index, (((diagonal, rhs), weight), (observed, first_order_rhs))) in workspace
            .iter
            .residual
            .iter_mut()
            .zip(workspace.iter.rhs.iter_mut())
            .zip(&workspace.iter.weights)
            .zip(y.iter().zip(&first_order_rhs))
            .enumerate()
        {
            let weight_squared = if prepared.active[index] {
                weight * weight
            } else {
                0.0
            };
            *diagonal = weight_squared;
            *rhs = weight_squared * observed + first_order_rhs;
        }

        solve_second_order_x_with_first_order(
            &prepared.x,
            &workspace.iter.residual,
            &workspace.iter.rhs,
            Some(&prepared.active),
            params.whittaker.lambda,
            params.lambda_1,
            baseline,
            &mut workspace.solver,
        )?;

        for (index, ((weight, observed), fitted)) in workspace
            .iter
            .weights
            .iter_mut()
            .zip(y)
            .zip(baseline.iter())
            .enumerate()
        {
            *weight = if prepared.active[index] {
                asls_weight(*observed, *fitted, params.p)
            } else {
                0.0
            };
        }
        tolerance = relative_change(&workspace.iter.previous_weights, &workspace.iter.weights);
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
    }

    Ok(FitReport::new(
        params.whittaker.max_iter + 1,
        false,
        tolerance,
    ))
}

fn drpls_xy_into(
    x: &[f64],
    y: &[f64],
    params: DrPlsParams,
    masks: &XyMaskSpec<'_>,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    let n = y.len();
    let mut workspace = WhittakerWorkspace::new(n);
    let mut band_workspace = GeneralPentadiagonalWorkspace::new(n);
    let bands = XyPenaltyBands::new(&prepared.x, params.whittaker.lambda)?;
    let mut lower2 = vec![0.0; n - 2];
    let mut lower1 = vec![0.0; n - 1];
    let mut diag = vec![0.0; n];
    let mut upper1 = vec![0.0; n - 1];
    let mut upper2 = vec![0.0; n - 2];
    let mut rhs = vec![0.0; n];
    let mut new_weights = vec![1.0; n];
    workspace.iter.weights.fill(1.0);
    apply_active_mask(&mut workspace.iter.weights, Some(&prepared.active));

    let mut tolerance = f64::INFINITY;
    for iter in 0..=params.whittaker.max_iter {
        fill_drpls_x_bands(
            &workspace.iter.weights,
            params.eta,
            &bands,
            &mut lower2,
            &mut lower1,
            &mut diag,
            &mut upper1,
            &mut upper2,
        );
        for (index, ((target, observed), weight)) in rhs
            .iter_mut()
            .zip(y)
            .zip(&workspace.iter.weights)
            .enumerate()
        {
            *target = if prepared.active[index] {
                observed * weight
            } else {
                0.0
            };
        }

        solve_general_pentadiagonal(
            GeneralPentadiagonalSystem {
                lower2: &lower2,
                lower1: &lower1,
                diag: &diag,
                upper1: &upper1,
                upper2: &upper2,
            },
            &rhs,
            baseline,
            &mut band_workspace,
        )?;

        if !drpls_weights_masked(
            y,
            baseline,
            iter + 1,
            &mut new_weights,
            Some(&prepared.active),
        ) {
            break;
        }
        tolerance = relative_change(&workspace.iter.weights, &new_weights);
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        workspace.iter.weights.copy_from_slice(&new_weights);
    }

    Ok(FitReport::new(
        params.whittaker.max_iter + 1,
        false,
        tolerance,
    ))
}

fn aspls_xy_into(
    x: &[f64],
    y: &[f64],
    params: AsPlsParams,
    masks: &XyMaskSpec<'_>,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    let n = y.len();
    let mut workspace = WhittakerWorkspace::new(n);
    let mut band_workspace = GeneralPentadiagonalWorkspace::new(n);
    let bands = XyPenaltyBands::new(&prepared.x, params.whittaker.lambda)?;
    let mut alpha = vec![1.0; n];
    let mut lower2 = vec![0.0; n - 2];
    let mut lower1 = vec![0.0; n - 1];
    let mut diag = vec![0.0; n];
    let mut upper1 = vec![0.0; n - 1];
    let mut upper2 = vec![0.0; n - 2];
    let mut rhs = vec![0.0; n];
    let mut new_weights = vec![1.0; n];
    workspace.iter.weights.fill(1.0);
    apply_active_mask(&mut workspace.iter.weights, Some(&prepared.active));

    let mut tolerance = f64::INFINITY;
    for iter in 0..params.whittaker.max_iter {
        fill_aspls_x_bands(
            &workspace.iter.weights,
            &alpha,
            &bands,
            &mut lower2,
            &mut lower1,
            &mut diag,
            &mut upper1,
            &mut upper2,
        );
        for (index, ((target, observed), weight)) in rhs
            .iter_mut()
            .zip(y)
            .zip(&workspace.iter.weights)
            .enumerate()
        {
            *target = if prepared.active[index] {
                observed * weight
            } else {
                0.0
            };
        }

        solve_general_pentadiagonal(
            GeneralPentadiagonalSystem {
                lower2: &lower2,
                lower1: &lower1,
                diag: &diag,
                upper1: &upper1,
                upper2: &upper2,
            },
            &rhs,
            baseline,
            &mut band_workspace,
        )?;

        if !aspls_weights_masked(
            y,
            baseline,
            params.asymmetric_coef,
            &mut new_weights,
            &mut workspace.iter.residual,
            Some(&prepared.active),
        ) {
            break;
        }
        tolerance = relative_change(&workspace.iter.weights, &new_weights);
        if tolerance <= params.whittaker.tol {
            return Ok(FitReport::new(iter + 1, true, tolerance));
        }
        workspace.iter.weights.copy_from_slice(&new_weights);
        let max_abs = workspace
            .iter
            .residual
            .iter()
            .zip(&prepared.active)
            .filter(|(_, active)| **active)
            .map(|(value, _)| value.abs())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
        for ((target, residual), active) in alpha
            .iter_mut()
            .zip(&workspace.iter.residual)
            .zip(&prepared.active)
        {
            *target = if *active {
                residual.abs() / max_abs
            } else {
                1.0
            };
        }
    }

    Ok(FitReport::new(params.whittaker.max_iter, false, tolerance))
}

fn brpls_xy_into(
    x: &[f64],
    y: &[f64],
    params: BrPlsParams,
    masks: &XyMaskSpec<'_>,
    baseline: &mut [f64],
) -> Result<FitReport> {
    validate_output("baseline", y.len(), baseline.len())?;
    params.validate()?;
    let prepared = prepare_xy(x, y, masks)?;
    let mut workspace = WhittakerWorkspace::new(y.len());
    workspace.iter.weights.fill(1.0);
    apply_active_mask(&mut workspace.iter.weights, Some(&prepared.active));
    let mut current_baseline = y.to_vec();
    let mut candidate = vec![0.0; y.len()];
    let mut new_weights = vec![1.0; y.len()];
    let mut beta = 0.5;
    let mut tolerance = f64::INFINITY;
    let mut outer_tolerance = f64::INFINITY;
    let mut total_iterations = 0usize;

    'outer: for outer in 0..=params.max_iter_2 {
        for inner in 0..=params.whittaker.max_iter {
            solve_second_order_x(
                &prepared.x,
                y,
                &workspace.iter.weights,
                params.whittaker.lambda,
                &mut candidate,
                &mut workspace.solver,
            )?;
            total_iterations += 1;

            if !brpls_weights_masked(
                y,
                &candidate,
                beta,
                &mut new_weights,
                Some(&prepared.active),
            ) {
                break 'outer;
            }

            tolerance = relative_change(&current_baseline, &candidate);
            if tolerance < params.whittaker.tol {
                if outer == 0 && inner == 0 {
                    current_baseline.copy_from_slice(&candidate);
                }
                break;
            }

            workspace.iter.weights.copy_from_slice(&new_weights);
            current_baseline.copy_from_slice(&candidate);
        }

        workspace.iter.weights.copy_from_slice(&new_weights);
        let weight_mean = workspace
            .iter
            .weights
            .iter()
            .zip(&prepared.active)
            .filter(|(_, active)| **active)
            .map(|(weight, _)| *weight)
            .sum::<f64>()
            / prepared.active_count as f64;
        outer_tolerance = (beta + weight_mean - 1.0).abs();
        if outer_tolerance < params.tol_2 {
            baseline.copy_from_slice(&current_baseline);
            return Ok(FitReport::new(total_iterations, true, outer_tolerance));
        }
        beta = 1.0 - weight_mean;
    }

    baseline.copy_from_slice(&current_baseline);
    Ok(FitReport::new(
        total_iterations,
        outer_tolerance <= params.tol_2,
        outer_tolerance.max(tolerance),
    ))
}

struct PreparedXy {
    x: Vec<f64>,
    active: Vec<bool>,
    active_count: usize,
}

fn prepare_xy(x: &[f64], y: &[f64], masks: &XyMaskSpec<'_>) -> Result<PreparedXy> {
    validate_xy(x, y)?;
    let mean_dx = (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64;
    let normalized_x: Vec<f64> = x.iter().map(|value| (value - x[0]) / mean_dx).collect();
    let mut active = vec![true; y.len()];

    apply_masks(x, y.len(), masks, &mut active)?;
    let active_count = active.iter().filter(|value| **value).count();
    if active_count < 2 {
        return Err(BaselineError::InvalidParameter {
            name: "mask",
            reason: "at least two unmasked points are required",
        });
    }

    Ok(PreparedXy {
        x: normalized_x,
        active,
        active_count,
    })
}

pub(crate) fn validate_xy(x: &[f64], y: &[f64]) -> Result<()> {
    validate_signal(y)?;
    validate_output("x", y.len(), x.len())?;
    if y.len() < 3 {
        return Err(BaselineError::TooShort {
            algorithm: "whittaker",
            len: y.len(),
            min: 3,
        });
    }

    for (index, value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(BaselineError::NonFiniteInput { index });
        }
        if index > 0 && *value <= x[index - 1] {
            return Err(BaselineError::InvalidParameter {
                name: "x",
                reason: "must be strictly increasing",
            });
        }
    }

    let mean_dx = (x[x.len() - 1] - x[0]) / (x.len() - 1) as f64;
    if !mean_dx.is_finite() || mean_dx <= 0.0 {
        return Err(BaselineError::InvalidParameter {
            name: "x",
            reason: "must span a positive finite range",
        });
    }
    Ok(())
}

fn apply_masks(x: &[f64], len: usize, masks: &XyMaskSpec<'_>, active: &mut [bool]) -> Result<()> {
    for &(start, end) in &masks.exclude_ranges {
        if !start.is_finite() || !end.is_finite() || start > end {
            return Err(BaselineError::InvalidParameter {
                name: "exclude_range",
                reason: "range bounds must be finite and sorted",
            });
        }
        for (active, value) in active.iter_mut().zip(x) {
            if *value >= start && *value <= end {
                *active = false;
            }
        }
    }

    for mask in &masks.exclude_masks {
        validate_output("exclude_mask", len, mask.len())?;
        for (active, excluded) in active.iter_mut().zip(*mask) {
            if *excluded {
                *active = false;
            }
        }
    }

    for mask in &masks.baseline_masks {
        validate_output("baseline_mask", len, mask.len())?;
        for (active, baseline_point) in active.iter_mut().zip(*mask) {
            if !*baseline_point {
                *active = false;
            }
        }
    }
    Ok(())
}

struct XyPenaltyBands {
    first_diag: Vec<f64>,
    first_sub1: Vec<f64>,
    second_diag: Vec<f64>,
    second_sub1: Vec<f64>,
    second_sub2: Vec<f64>,
}

impl XyPenaltyBands {
    fn new(x: &[f64], lambda: f64) -> Result<Self> {
        let n = x.len();
        let mut first_diag = vec![0.0; n];
        let mut first_sub1 = vec![0.0; n - 1];
        let mut second_diag = vec![0.0; n];
        let mut second_sub1 = vec![0.0; n - 1];
        let mut second_sub2 = vec![0.0; n - 2];
        let zero_weights = vec![0.0; n];

        add_first_order_x_penalty(x, None, 1.0, &mut first_diag, &mut first_sub1)?;
        fill_second_order_x_bands(
            x,
            &zero_weights,
            lambda,
            &mut second_diag,
            &mut second_sub1,
            &mut second_sub2,
        )?;

        Ok(Self {
            first_diag,
            first_sub1,
            second_diag,
            second_sub1,
            second_sub2,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_drpls_x_bands(
    weights: &[f64],
    eta: f64,
    bands: &XyPenaltyBands,
    lower2: &mut [f64],
    lower1: &mut [f64],
    diag: &mut [f64],
    upper1: &mut [f64],
    upper2: &mut [f64],
) {
    for (i, target) in diag.iter_mut().enumerate() {
        *target =
            bands.first_diag[i] + bands.second_diag[i] * (1.0 - eta * weights[i]) + weights[i];
    }
    for i in 0..weights.len() - 1 {
        upper1[i] = bands.first_sub1[i] + bands.second_sub1[i] * (1.0 - eta * weights[i]);
        lower1[i] = bands.first_sub1[i] + bands.second_sub1[i] * (1.0 - eta * weights[i + 1]);
    }
    for i in 0..weights.len() - 2 {
        upper2[i] = bands.second_sub2[i] * (1.0 - eta * weights[i]);
        lower2[i] = bands.second_sub2[i] * (1.0 - eta * weights[i + 2]);
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_aspls_x_bands(
    weights: &[f64],
    alpha: &[f64],
    bands: &XyPenaltyBands,
    lower2: &mut [f64],
    lower1: &mut [f64],
    diag: &mut [f64],
    upper1: &mut [f64],
    upper2: &mut [f64],
) {
    for (i, target) in diag.iter_mut().enumerate() {
        *target = weights[i] + alpha[i] * bands.second_diag[i];
    }
    for i in 0..weights.len() - 1 {
        upper1[i] = alpha[i] * bands.second_sub1[i];
        lower1[i] = alpha[i + 1] * bands.second_sub1[i];
    }
    for i in 0..weights.len() - 2 {
        upper2[i] = alpha[i] * bands.second_sub2[i];
        lower2[i] = alpha[i + 2] * bands.second_sub2[i];
    }
}
