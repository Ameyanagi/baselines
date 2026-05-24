//! Whittaker lambda and data-size sweep rendered with ruviz.
//!
//! The data formula, algorithm list, data sizes, lambda-search grid, and
//! `tol=1e-2, max_iter=50` settings mirror:
//! <https://pybaselines.readthedocs.io/en/latest/generated/examples/whittaker/plot_lam_vs_data_size.html>.
//! Inspired by the linked pybaselines example; pybaselines is used as a
//! behavioral and documentation reference only, and this example calls this
//! crate's native Rust implementations.

mod common;

use baselines::BaselineError;
use baselines::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IaslsParams, LsrPlsParams, PsalsaParams, WhittakerParams, airpls, arpls, asls, aspls, brpls,
    derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};
use common::{
    ReferenceBaseline, ensure_output_dir, output_path, print_output, reference_make_data,
};
use ruviz::prelude::*;
use std::error::Error;

const NUM_POINTS: [usize; 6] = [499, 1045, 2186, 4573, 9563, 20000];

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    let algorithms = [
        WhittakerAlgorithm::Asls,
        WhittakerAlgorithm::Iasls,
        WhittakerAlgorithm::Airpls,
        WhittakerAlgorithm::Arpls,
        WhittakerAlgorithm::Iarpls,
        WhittakerAlgorithm::Drpls,
        WhittakerAlgorithm::Aspls,
        WhittakerAlgorithm::Psalsa,
        WhittakerAlgorithm::Derpsalsa,
        WhittakerAlgorithm::Brpls,
        WhittakerAlgorithm::Lsrpls,
    ];
    let baselines = [
        BaselineCase {
            name: "exponential",
            kind: ReferenceBaseline::Exponential,
            color: Color::new(43, 70, 104),
        },
        BaselineCase {
            name: "gaussian",
            kind: ReferenceBaseline::Gaussian,
            color: Color::new(218, 111, 76),
        },
        BaselineCase {
            name: "sine",
            kind: ReferenceBaseline::Sine,
            color: Color::new(84, 151, 160),
        },
    ];

    let results = calculate_whittaker_lam_sweeps(&algorithms, &baselines)?;
    save_algorithm_plots(&results, &baselines)?;
    save_baseline_summary_plots(&results, &algorithms, &baselines)?;

    Ok(())
}

fn calculate_whittaker_lam_sweeps(
    algorithms: &[WhittakerAlgorithm],
    baselines: &[BaselineCase],
) -> std::result::Result<Vec<AlgorithmSweep>, Box<dyn Error>> {
    let mut results = Vec::with_capacity(algorithms.len());
    println!("Function, baseline type, intercept & slope of log(N) vs log(lam) fit");
    println!("{}", "-".repeat(60));

    for &algorithm in algorithms {
        let mut algorithm_result = AlgorithmSweep {
            algorithm,
            cases: Vec::with_capacity(baselines.len()),
        };
        for case in baselines {
            let mut best_lams = Vec::with_capacity(NUM_POINTS.as_slice().len());
            let mut previous_min = None;
            for &num_x in &NUM_POINTS {
                let (_, y, baseline) = reference_make_data(num_x, case.kind);
                let best_lam = optimize_lam(&y, &baseline, algorithm, previous_min, 1.0e-2, 50)?;
                previous_min = Some(best_lam);
                best_lams.push(best_lam);
            }

            let log_n = log10_usizes(&NUM_POINTS);
            let fit = LinearFit::fit(&log_n, &best_lams);
            println!(
                "{:<11} {:<13} [{:.6}, {:.6}]",
                algorithm.name(),
                case.name,
                fit.intercept,
                fit.slope
            );
            algorithm_result.cases.push(CaseSweep {
                baseline_name: case.name,
                best_lams,
                fit,
            });
        }
        results.push(algorithm_result);
    }

    Ok(results)
}

fn optimize_lam(
    y: &[f64],
    known_baseline: &[f64],
    algorithm: WhittakerAlgorithm,
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
    algorithm: WhittakerAlgorithm,
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

fn save_algorithm_plots(
    results: &[AlgorithmSweep],
    baselines: &[BaselineCase],
) -> std::result::Result<(), Box<dyn Error>> {
    let x = log10_usizes(&NUM_POINTS);
    for result in results {
        let mut plot = Plot::new()
            .title(format!("{}: lam vs data size", result.algorithm.name()))
            .xlabel("log10(Input Array Size, N)")
            .ylabel("log10(Optimal lam)")
            .max_resolution(1800, 1200)
            .legend_position(LegendPosition::Best);
        for (case, baseline) in result.cases.iter().zip(baselines) {
            let line_y: Vec<f64> = x.iter().map(|value| case.fit.predict(*value)).collect();
            plot = plot
                .line(&x, &line_y)
                .label(case.baseline_name)
                .color(baseline.color)
                .into();
            plot = plot
                .scatter(&x, &case.best_lams)
                .label(format!("{} samples", case.baseline_name))
                .color(baseline.color)
                .into();
        }

        let path = output_path(&format!(
            "gallery_lam_vs_data_size_{}.png",
            result.algorithm.name()
        ));
        plot.save(&path)?;
        print_output(&path);
    }
    Ok(())
}

fn save_baseline_summary_plots(
    results: &[AlgorithmSweep],
    algorithms: &[WhittakerAlgorithm],
    baselines: &[BaselineCase],
) -> std::result::Result<(), Box<dyn Error>> {
    let x = log10_usizes(&NUM_POINTS);
    let colors = colors();
    for (case_index, case) in baselines.iter().enumerate() {
        let mut plot = Plot::new()
            .title(format!("{} baseline: lam vs data size", case.name))
            .xlabel("log10(Input Array Size, N)")
            .ylabel("log10(Optimal lam)")
            .max_resolution(1800, 1200)
            .legend_position(LegendPosition::Best);
        for (algorithm_index, algorithm) in algorithms.iter().enumerate() {
            let sweep = &results[algorithm_index].cases[case_index];
            let line_y: Vec<f64> = x.iter().map(|value| sweep.fit.predict(*value)).collect();
            plot = plot
                .line(&x, &line_y)
                .label(algorithm.name())
                .color(colors[algorithm_index])
                .into();
        }

        let path = output_path(&format!(
            "gallery_lam_vs_data_size_{}_baseline.png",
            case.name
        ));
        plot.save(&path)?;
        print_output(&path);
    }
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

#[derive(Debug, Clone, Copy)]
struct BaselineCase {
    name: &'static str,
    kind: ReferenceBaseline,
    color: Color,
}

#[derive(Debug)]
struct AlgorithmSweep {
    algorithm: WhittakerAlgorithm,
    cases: Vec<CaseSweep>,
}

#[derive(Debug)]
struct CaseSweep {
    baseline_name: &'static str,
    best_lams: Vec<f64>,
    fit: LinearFit,
}

#[derive(Debug, Clone, Copy)]
enum WhittakerAlgorithm {
    Asls,
    Iasls,
    Airpls,
    Arpls,
    Iarpls,
    Drpls,
    Aspls,
    Psalsa,
    Derpsalsa,
    Brpls,
    Lsrpls,
}

impl WhittakerAlgorithm {
    fn name(self) -> &'static str {
        match self {
            Self::Asls => "asls",
            Self::Iasls => "iasls",
            Self::Airpls => "airpls",
            Self::Arpls => "arpls",
            Self::Iarpls => "iarpls",
            Self::Drpls => "drpls",
            Self::Aspls => "aspls",
            Self::Psalsa => "psalsa",
            Self::Derpsalsa => "derpsalsa",
            Self::Brpls => "brpls",
            Self::Lsrpls => "lsrpls",
        }
    }

    fn fit(self, y: &[f64], lambda: f64, tol: f64, max_iter: usize) -> baselines::Result<Vec<f64>> {
        let whittaker = WhittakerParams {
            lambda,
            tol,
            max_iter,
        };
        let fit = match self {
            Self::Asls => asls(y, AslsParams { whittaker, p: 0.01 })?,
            Self::Iasls => iasls(
                y,
                IaslsParams {
                    whittaker,
                    p: 0.05,
                    lambda_1: 1.0e-8 * lambda,
                },
            )?,
            Self::Airpls => airpls(y, AirPlsParams { whittaker })?,
            Self::Arpls => arpls(y, ArPlsParams { whittaker })?,
            Self::Iarpls => iarpls(y, ArPlsParams { whittaker })?,
            Self::Drpls => drpls(
                y,
                DrPlsParams {
                    whittaker,
                    eta: 0.5,
                },
            )?,
            Self::Aspls => aspls(
                y,
                AsPlsParams {
                    whittaker,
                    asymmetric_coef: 0.5,
                },
            )?,
            Self::Psalsa => psalsa(
                y,
                PsalsaParams {
                    whittaker,
                    p: 0.5,
                    k: None,
                },
            )?,
            Self::Derpsalsa => derpsalsa(
                y,
                DerPsalsaParams {
                    whittaker,
                    p: 0.01,
                    k: None,
                    smooth_half_window: None,
                    num_smooths: 16,
                },
            )?,
            Self::Brpls => brpls(
                y,
                BrPlsParams {
                    whittaker,
                    max_iter_2: 50,
                    tol_2: 1.0e-3,
                },
            )?,
            Self::Lsrpls => lsrpls(y, LsrPlsParams { whittaker })?,
        };
        Ok(fit.baseline)
    }
}

fn colors() -> [Color; 11] {
    [
        Color::new(43, 70, 104),
        Color::new(218, 111, 76),
        Color::new(84, 151, 160),
        Color::new(118, 85, 148),
        Color::new(232, 168, 72),
        Color::new(80, 145, 110),
        Color::new(173, 80, 116),
        Color::new(91, 113, 65),
        Color::new(142, 90, 55),
        Color::new(96, 120, 174),
        Color::new(76, 76, 76),
    ]
}
