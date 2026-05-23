//! Ruviz counterpart for pybaselines' P-spline Whittaker gallery.
//!
//! The data formula, data sizes, lambda-search grid, `arpls` and
//! `pspline_arpls` functions, and `tol=1e-2, max_iter=50` settings mirror:
//! <https://pybaselines.readthedocs.io/en/latest/generated/examples/spline/plot_pspline_whittaker.html>.
//! pybaselines is used as a behavioral and documentation reference only; this
//! example calls this crate's native Rust implementations.

mod common;

use baselines::BaselineError;
use baselines::spline::pspline_arpls;
use baselines::whittaker::{ArPlsParams, WhittakerParams, arpls};
use common::{
    PybaselinesBaseline, ensure_output_dir, output_path, print_output, pybaselines_make_data,
};
use ruviz::prelude::*;
use std::error::Error;

const NUM_POINTS: [usize; 6] = [499, 1045, 2186, 4573, 9563, 20000];

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    let algorithms = [
        SplineWhittakerAlgorithm::Arpls,
        SplineWhittakerAlgorithm::PsplineArpls,
    ];
    let results = calculate_lam_sweeps(&algorithms)?;
    save_plot(&results)?;

    Ok(())
}

fn calculate_lam_sweeps(
    algorithms: &[SplineWhittakerAlgorithm],
) -> std::result::Result<Vec<AlgorithmSweep>, Box<dyn Error>> {
    let mut results = Vec::with_capacity(algorithms.len());
    println!("Function, intercept & slope of log(N) vs log(lam) fit");
    println!("{}", "-".repeat(60));

    for &algorithm in algorithms {
        let mut best_lams = Vec::with_capacity(NUM_POINTS.as_slice().len());
        let mut previous_min = None;
        for &num_x in &NUM_POINTS {
            let (_, y, baseline) = pybaselines_make_data(num_x, PybaselinesBaseline::Exponential);
            let best_lam = optimize_lam(&y, &baseline, algorithm, previous_min, 1.0e-2, 50)?;
            previous_min = Some(best_lam);
            best_lams.push(best_lam);
        }

        let log_n = log10_usizes(&NUM_POINTS);
        let fit = LinearFit::fit(&log_n, &best_lams);
        println!(
            "{:<16} [{:.6}, {:.6}]",
            algorithm.name(),
            fit.intercept,
            fit.slope
        );
        results.push(AlgorithmSweep {
            algorithm,
            best_lams,
            fit,
        });
    }

    Ok(results)
}

fn optimize_lam(
    y: &[f64],
    known_baseline: &[f64],
    algorithm: SplineWhittakerAlgorithm,
    previous_min: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> baselines::Result<f64> {
    let min_lam = previous_min.map_or(-1.0, |value| value - 0.5);
    let coarse_lams = arange(min_lam, 13.5, 0.5);
    let coarse_best = minimize_l2(y, known_baseline, algorithm, &coarse_lams, tol, max_iter)?;
    let fine_lams = arange(coarse_best - 0.5, coarse_best + 0.7, 0.2);
    minimize_l2(y, known_baseline, algorithm, &fine_lams, tol, max_iter)
}

fn minimize_l2(
    y: &[f64],
    known_baseline: &[f64],
    algorithm: SplineWhittakerAlgorithm,
    log_lambdas: &[f64],
    tol: f64,
    max_iter: usize,
) -> baselines::Result<f64> {
    let mut min_error = f64::INFINITY;
    let mut best_lam = log_lambdas[0];
    for &log_lambda in log_lambdas {
        let lambda = 10.0_f64.powf(log_lambda);
        let Ok(baseline) = algorithm.fit(y, lambda, tol, max_iter) else {
            continue;
        };
        let fit_error = l2_norm_difference(known_baseline, &baseline);
        if fit_error < min_error {
            min_error = fit_error;
            best_lam = log_lambda;
        }
    }
    if min_error.is_finite() {
        Ok(best_lam)
    } else {
        Err(BaselineError::LinearSolve {
            reason: "all lambda candidates failed",
        })
    }
}

fn save_plot(results: &[AlgorithmSweep]) -> std::result::Result<(), Box<dyn Error>> {
    let x = log10_usizes(&NUM_POINTS);
    let colors = [Color::new(43, 70, 104), Color::new(218, 111, 76)];
    let mut plot = Plot::new()
        .title("pybaselines P-spline Whittaker lam vs data size")
        .xlabel("log10(Input Array Size, N)")
        .ylabel("log10(Optimal lam)")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best);
    for (index, result) in results.iter().enumerate() {
        let line_y: Vec<f64> = x.iter().map(|value| result.fit.predict(*value)).collect();
        plot = plot
            .line(&x, &line_y)
            .label(result.algorithm.name())
            .color(colors[index])
            .into();
        plot = plot
            .scatter(&x, &result.best_lams)
            .label(format!("{} samples", result.algorithm.name()))
            .color(colors[index])
            .into();
    }

    let path = output_path("pybaselines_gallery_pspline_whittaker.png");
    plot.save(&path)?;
    print_output(&path);
    Ok(())
}

fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut value = start;
    while value < stop {
        values.push(value);
        value += step;
    }
    values
}

fn log10_usizes(values: &[usize]) -> Vec<f64> {
    values.iter().map(|value| (*value as f64).log10()).collect()
}

fn l2_norm_difference(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let diff = left - right;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

#[derive(Debug, Clone, Copy)]
struct LinearFit {
    intercept: f64,
    slope: f64,
}

impl LinearFit {
    fn fit(x: &[f64], y: &[f64]) -> Self {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let numerator = x
            .iter()
            .zip(y)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
        let denominator = x
            .iter()
            .map(|x| {
                let centered = x - mean_x;
                centered * centered
            })
            .sum::<f64>();
        Self {
            intercept: mean_y - numerator / denominator * mean_x,
            slope: numerator / denominator,
        }
    }

    fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
}

#[derive(Debug)]
struct AlgorithmSweep {
    algorithm: SplineWhittakerAlgorithm,
    best_lams: Vec<f64>,
    fit: LinearFit,
}

#[derive(Debug, Clone, Copy)]
enum SplineWhittakerAlgorithm {
    Arpls,
    PsplineArpls,
}

impl SplineWhittakerAlgorithm {
    fn name(self) -> &'static str {
        match self {
            Self::Arpls => "arpls",
            Self::PsplineArpls => "pspline_arpls",
        }
    }

    fn fit(self, y: &[f64], lambda: f64, tol: f64, max_iter: usize) -> baselines::Result<Vec<f64>> {
        let params = ArPlsParams {
            whittaker: WhittakerParams {
                lambda,
                tol,
                max_iter,
            },
        };
        let fit = match self {
            Self::Arpls => arpls(y, params)?,
            Self::PsplineArpls => pspline_arpls(y, params)?,
        };
        Ok(fit.baseline)
    }
}
