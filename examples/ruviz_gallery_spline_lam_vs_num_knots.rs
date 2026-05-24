//! Spline lambda and knot-count sweep rendered with ruviz.
//!
//! The data formula, knot counts, data sizes, lambda-search grid,
//! `mixture_model`, `diff_order=2`, and `tol=1e-2, max_iter=50` settings mirror:
//! <https://pybaselines.readthedocs.io/en/latest/generated/examples/spline/plot_lam_vs_num_knots.html>.
//! Inspired by the linked pybaselines example; pybaselines is used as a
//! behavioral and documentation reference only, and this example calls this
//! crate's native Rust implementation.

mod common;

use baselines::BaselineError;
use baselines::spline::{MixtureModelParams, mixture_model};
use common::{
    LineSeries, ReferenceBaseline, ensure_output_dir, output_path, print_output,
    reference_make_data, save_lines,
};
use ruviz::prelude::*;
use std::error::Error;

const NUM_KNOTS: [usize; 5] = [20, 53, 141, 376, 1000];
const NUM_POINTS: [usize; 6] = [500, 1045, 2186, 4573, 9563, 20000];

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    save_baseline_reference()?;
    let best_lams = calculate_best_lambdas()?;
    save_lam_vs_data_size_plot(&best_lams)?;
    save_lam_vs_num_knots_plot(&best_lams)?;

    Ok(())
}

fn save_baseline_reference() -> std::result::Result<(), Box<dyn Error>> {
    let (x, _, baseline) = reference_make_data(1000, ReferenceBaseline::Exponential);
    let path = output_path("gallery_spline_lam_vs_num_knots_baseline.png");
    save_lines(
        "Spline Lambda vs Number of Knots Baseline",
        "x",
        "baseline",
        &x,
        &[LineSeries {
            label: "exponential baseline",
            y: &baseline,
            color: Color::new(43, 70, 104),
        }],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn calculate_best_lambdas() -> baselines::Result<Vec<Vec<f64>>> {
    let mut best_lams = vec![vec![0.0; NUM_POINTS.as_slice().len()]; NUM_KNOTS.as_slice().len()];
    for (knot_index, &num_knot) in NUM_KNOTS.iter().enumerate() {
        let mut previous_min = Some(0.0);
        for (point_index, &num_x) in NUM_POINTS.iter().enumerate() {
            let (_, y, baseline) = reference_make_data(num_x, ReferenceBaseline::Exponential);
            let best_lam = optimize_lam(&y, &baseline, num_knot, previous_min, 1.0e-2, 50)?;
            previous_min = Some(best_lam);
            best_lams[knot_index][point_index] = best_lam;
            println!("num_knots={num_knot:<5} n={num_x:<6} best_log10_lam={best_lam:.3}");
        }
    }
    Ok(best_lams)
}

fn optimize_lam(
    y: &[f64],
    known_baseline: &[f64],
    num_knots: usize,
    previous_min: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> baselines::Result<f64> {
    let min_lam = previous_min.map_or(-1.0, |value| value - 0.5);
    let coarse_lams = arange(min_lam, 13.5, 0.5);
    let coarse_best = minimize_l2(y, known_baseline, num_knots, &coarse_lams, tol, max_iter)?;
    let fine_lams = arange(coarse_best - 0.5, coarse_best + 0.7, 0.2);
    minimize_l2(y, known_baseline, num_knots, &fine_lams, tol, max_iter)
}

fn minimize_l2(
    y: &[f64],
    known_baseline: &[f64],
    num_knots: usize,
    log_lambdas: &[f64],
    tol: f64,
    max_iter: usize,
) -> baselines::Result<f64> {
    let mut min_error = f64::INFINITY;
    let mut best_lam = log_lambdas[0];
    for &log_lambda in log_lambdas {
        let lambda = 10.0_f64.powf(log_lambda);
        let Ok(fit) = mixture_model(
            y,
            MixtureModelParams {
                lambda,
                num_knots,
                diff_order: 2,
                tol,
                max_iter,
                ..MixtureModelParams::default()
            },
        ) else {
            continue;
        };
        let fit_error = l2_norm_difference(known_baseline, &fit.baseline);
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

fn save_lam_vs_data_size_plot(best_lams: &[Vec<f64>]) -> std::result::Result<(), Box<dyn Error>> {
    println!("Number of knots, intercept & slope of log(N) vs log(lam) fit");
    println!("{}", "-".repeat(60));
    let x = log10_usizes(&NUM_POINTS);
    let colors = colors();
    let mut plot = Plot::new()
        .title("Spline Lambda vs Data Size")
        .xlabel("log10(Input Array Size, N)")
        .ylabel("log10(Optimal lam)")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best);
    for (index, &num_knot) in NUM_KNOTS.iter().enumerate() {
        let fit = LinearFit::fit(&x, &best_lams[index]);
        println!("{num_knot:<6} [{:.6}, {:.6}]", fit.intercept, fit.slope);
        let line_y: Vec<f64> = x.iter().map(|value| fit.predict(*value)).collect();
        plot = plot
            .line(&x, &line_y)
            .label(format!("num_knots={num_knot}"))
            .color(colors[index])
            .into();
        plot = plot
            .scatter(&x, &best_lams[index])
            .label(format!("num_knots={num_knot} samples"))
            .color(colors[index])
            .into();
    }

    let path = output_path("gallery_spline_lam_vs_num_knots_data_size.png");
    plot.save(&path)?;
    print_output(&path);
    Ok(())
}

fn save_lam_vs_num_knots_plot(best_lams: &[Vec<f64>]) -> std::result::Result<(), Box<dyn Error>> {
    println!("Number of points, intercept & slope of log(number of knots) vs log(lam) fit");
    println!("{}", "-".repeat(80));
    let x = log10_usizes(&NUM_KNOTS);
    let colors = colors();
    let mut plot = Plot::new()
        .title("Spline Lambda vs Number of Knots")
        .xlabel("log10(Number of Knots)")
        .ylabel("log10(Optimal lam)")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best);
    for (point_index, &num_x) in NUM_POINTS.iter().enumerate() {
        let y: Vec<f64> = best_lams.iter().map(|row| row[point_index]).collect();
        let fit = LinearFit::fit(&x, &y);
        println!("{num_x:<6} [{:.6}, {:.6}]", fit.intercept, fit.slope);
        let line_y: Vec<f64> = x.iter().map(|value| fit.predict(*value)).collect();
        plot = plot
            .line(&x, &line_y)
            .label(format!("data size={num_x}"))
            .color(colors[point_index])
            .into();
        plot = plot
            .scatter(&x, &y)
            .label(format!("data size={num_x} samples"))
            .color(colors[point_index])
            .into();
    }

    let path = output_path("gallery_spline_lam_vs_num_knots_knots.png");
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

fn colors() -> [Color; 6] {
    [
        Color::new(43, 70, 104),
        Color::new(218, 111, 76),
        Color::new(84, 151, 160),
        Color::new(118, 85, 148),
        Color::new(232, 168, 72),
        Color::new(80, 145, 110),
    ]
}
