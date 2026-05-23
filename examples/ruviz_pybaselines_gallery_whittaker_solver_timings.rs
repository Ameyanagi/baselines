//! Ruviz counterpart for pybaselines' Whittaker solver timings gallery.
//!
//! The data sizes, repeat count, lambda equation, AsLS algorithm, and
//! `max_iter=8` setting mirror:
//! <https://pybaselines.readthedocs.io/en/latest/generated/examples/whittaker/plot_whittaker_solvers.html>.
//! pybaselines compares SciPy/pentapy solver backends; this Rust-native example
//! compares the allocating API with the reusable-workspace API over the same
//! workload because this crate uses one native pentadiagonal solver.

mod common;

use baselines::whittaker::{AslsParams, WhittakerParams, WhittakerWorkspace, asls, asls_into};
use common::{PybaselinesBaseline, pybaselines_make_data as make_data};
use common::{ensure_output_dir, output_path, print_output};
use ruviz::prelude::*;
use std::error::Error;
use std::time::Instant;

const REPEATS: usize = 25;
const DATA_SIZES: [usize; 8] = [499, 935, 1748, 3270, 6115, 11437, 21388, 40000];

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    let allocating = time_allocating_api()?;
    let reusable = time_reusable_workspace_api()?;
    save_timing_plot(&allocating, &reusable)?;
    save_relative_reduction_plot(&allocating, &reusable)?;

    Ok(())
}

fn time_allocating_api() -> baselines::Result<TimingSeries> {
    time_series("asls allocating", |y, params, _baseline, _workspace| {
        asls(y, params).map(|_| ())
    })
}

fn time_reusable_workspace_api() -> baselines::Result<TimingSeries> {
    time_series("asls_into workspace", |y, params, baseline, workspace| {
        asls_into(y, params, baseline, workspace).map(|_| ())
    })
}

fn time_series<F>(label: &'static str, mut fit: F) -> baselines::Result<TimingSeries>
where
    F: FnMut(&[f64], AslsParams, &mut [f64], &mut WhittakerWorkspace) -> baselines::Result<()>,
{
    let mut medians = Vec::with_capacity(DATA_SIZES.as_slice().len());
    let mut std_devs = Vec::with_capacity(DATA_SIZES.as_slice().len());
    for &num_x in &DATA_SIZES {
        let (_, y, _) = make_data(num_x, PybaselinesBaseline::Exponential);
        let lambda = lam_equation(num_x);
        let params = AslsParams {
            whittaker: WhittakerParams {
                lambda,
                // pybaselines uses tol=-1 to force the same number of
                // iterations. The public Rust API requires positive tolerance,
                // so use the smallest positive value available.
                tol: f64::MIN_POSITIVE,
                max_iter: 8,
            },
            p: 0.01,
        };
        let mut baseline = vec![0.0; y.len()];
        let mut workspace = WhittakerWorkspace::new(y.len());
        let mut times = Vec::with_capacity(REPEATS);
        for repeat in 0..=REPEATS {
            let start = Instant::now();
            fit(&y, params, &mut baseline, &mut workspace)?;
            let elapsed = start.elapsed().as_secs_f64();
            if repeat > 0 {
                times.push(elapsed);
            }
        }
        let median = median(&times);
        let std_dev = sample_std(&times);
        println!("{label:<20} n={num_x:<6} median={median:.6e} std={std_dev:.6e}");
        medians.push(median);
        std_devs.push(std_dev);
    }

    Ok(TimingSeries {
        label,
        medians,
        std_devs,
    })
}

fn save_timing_plot(
    allocating: &TimingSeries,
    reusable: &TimingSeries,
) -> std::result::Result<(), Box<dyn Error>> {
    let x = log10_usizes(&DATA_SIZES);
    let allocating_y = log10_values(&allocating.medians);
    let reusable_y = log10_values(&reusable.medians);
    let allocating_upper = log10_sum(&allocating.medians, &allocating.std_devs);
    let reusable_upper = log10_sum(&reusable.medians, &reusable.std_devs);
    let path = output_path("pybaselines_gallery_whittaker_solver_timings.png");
    Plot::new()
        .title("pybaselines Whittaker solver timings")
        .xlabel("log10(Input Array Size)")
        .ylabel("log10(Median Time, seconds)")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best)
        .line(&x, &allocating_y)
        .label(allocating.label)
        .color(Color::new(43, 70, 104))
        .scatter(&x, &allocating_y)
        .label("allocating samples")
        .color(Color::new(43, 70, 104))
        .line(&x, &allocating_upper)
        .label("allocating + std")
        .color(Color::new(96, 120, 174))
        .line(&x, &reusable_y)
        .label(reusable.label)
        .color(Color::new(218, 111, 76))
        .scatter(&x, &reusable_y)
        .label("workspace samples")
        .color(Color::new(218, 111, 76))
        .line(&x, &reusable_upper)
        .label("workspace + std")
        .color(Color::new(232, 168, 72))
        .save(&path)?;
    print_output(&path);
    Ok(())
}

fn save_relative_reduction_plot(
    allocating: &TimingSeries,
    reusable: &TimingSeries,
) -> std::result::Result<(), Box<dyn Error>> {
    let x = log10_usizes(&DATA_SIZES);
    let relative_reduction: Vec<f64> = allocating
        .medians
        .iter()
        .zip(&reusable.medians)
        .map(|(allocating, reusable)| 100.0 * (allocating - reusable) / allocating)
        .collect();
    let path = output_path("pybaselines_gallery_whittaker_solver_relative_reduction.png");
    Plot::new()
        .title("pybaselines Whittaker solver relative timing")
        .xlabel("log10(Input Array Size)")
        .ylabel("Relative Time Reduction (%)")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best)
        .line(&x, &relative_reduction)
        .label("workspace vs allocating")
        .color(Color::new(84, 151, 160))
        .scatter(&x, &relative_reduction)
        .label("samples")
        .color(Color::new(84, 151, 160))
        .save(&path)?;
    print_output(&path);
    Ok(())
}

fn lam_equation(n: usize) -> f64 {
    10.0_f64.powf(-6.35 + (n as f64).log10() * 4.17)
}

fn log10_usizes(values: &[usize]) -> Vec<f64> {
    values.iter().map(|value| (*value as f64).log10()).collect()
}

fn log10_values(values: &[f64]) -> Vec<f64> {
    values.iter().map(|value| value.log10()).collect()
}

fn log10_sum(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left + right).log10())
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn sample_std(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

#[derive(Debug)]
struct TimingSeries {
    label: &'static str,
    medians: Vec<f64>,
    std_devs: Vec<f64>,
}
