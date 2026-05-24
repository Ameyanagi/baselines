//! BEADS preprocessing gallery rendered with ruviz.
//!
//! The signal, three baseline formulas, 1000-point x grid, noise scale, and
//! BEADS parameter sets mirror:
//! <https://pybaselines.readthedocs.io/en/latest/generated/examples/misc/plot_beads_preprocessing.html>.
//! Inspired by the linked pybaselines example; pybaselines is used as a
//! behavioral and documentation reference only, and this example calls this
//! crate's native Rust implementation.

mod common;

use baselines::misc::{BeadsParams, beads};
use common::{LineSeries, NormalNoise, ensure_output_dir, gaussian, linspace, output_path};
use common::{print_output, save_lines, standard_signal};
use ruviz::prelude::*;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    let x = linspace(0.0, 1000.0, 1000);
    save_preprocessing_inputs(&x)?;
    save_beads_fit(
        &x,
        0,
        BeadsParams {
            lam_0: 0.005,
            lam_1: 0.01,
            lam_2: 1.0,
            ..BeadsParams::default()
        },
        "gallery_beads_baseline_1.png",
    )?;
    save_beads_fit(
        &x,
        1,
        BeadsParams {
            lam_0: 0.015,
            lam_1: 0.1,
            lam_2: 1.0,
            ..BeadsParams::default()
        },
        "gallery_beads_baseline_2.png",
    )?;
    save_beads_fit(
        &x,
        2,
        BeadsParams {
            lam_0: 0.00006,
            lam_1: 0.00008,
            lam_2: 0.05,
            tol: 1.0e-3,
            freq_cutoff: 0.04,
            asymmetry: 3.0,
            ..BeadsParams::default()
        },
        "gallery_beads_baseline_3.png",
    )?;

    Ok(())
}

fn save_preprocessing_inputs(x: &[f64]) -> std::result::Result<(), Box<dyn Error>> {
    for baseline_type in 0..3 {
        let (y, true_baseline) = make_beads_data(x, baseline_type);
        let parabola = endpoint_parabola(&y);
        let path = output_path(&format!(
            "gallery_beads_preprocessing_baseline_{}.png",
            baseline_type + 1
        ));
        save_lines(
            &format!("BEADS Endpoint Parabola {}", baseline_type + 1),
            "x",
            "intensity",
            x,
            &[
                LineSeries {
                    label: "raw data",
                    y: &y,
                    color: Color::new(43, 70, 104),
                },
                LineSeries {
                    label: "fit parabola",
                    y: &parabola,
                    color: Color::new(218, 111, 76),
                },
                LineSeries {
                    label: "true baseline",
                    y: &true_baseline,
                    color: Color::new(80, 145, 110),
                },
            ],
            &path,
        )?;
        print_output(&path);
    }
    Ok(())
}

fn save_beads_fit(
    x: &[f64],
    baseline_type: usize,
    params: BeadsParams,
    filename: &str,
) -> std::result::Result<(), Box<dyn Error>> {
    let (y, true_baseline) = make_beads_data(x, baseline_type);
    let without_parabola = beads(
        &y,
        BeadsParams {
            fit_parabola: false,
            ..params
        },
    )?;
    let with_parabola = beads(
        &y,
        BeadsParams {
            fit_parabola: true,
            ..params
        },
    )?;
    let path = output_path(filename);
    save_lines(
        &format!("BEADS Baseline {}", baseline_type + 1),
        "x",
        "intensity",
        x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "fit_parabola=false",
                y: &without_parabola.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "fit_parabola=true",
                y: &with_parabola.baseline,
                color: Color::new(84, 151, 160),
            },
            LineSeries {
                label: "true baseline",
                y: &true_baseline,
                color: Color::new(80, 145, 110),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn make_beads_data(x: &[f64], baseline_type: usize) -> (Vec<f64>, Vec<f64>) {
    let signal = standard_signal(x);
    let baseline: Vec<f64> = match baseline_type {
        0 => x
            .iter()
            .map(|value| 2.0e-5 * (value - 500.0).powi(2) - 5.0)
            .collect(),
        1 => x
            .iter()
            .map(|value| 10.0 - 10.0 * (-value / 600.0).exp())
            .collect(),
        _ => {
            let mut cumulative = 0.0;
            x.iter()
                .map(|&value| {
                    cumulative += gaussian(value, 0.05, 400.0, 100.0);
                    -cumulative + gaussian(value, 3.0, 800.0, 100.0) - 5.0
                })
                .collect()
        }
    };
    let mut noise = NormalNoise::new(0);
    let y = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.2))
        .collect();
    (y, baseline)
}

fn endpoint_parabola(y: &[f64]) -> Vec<f64> {
    let min_y = y.iter().copied().fold(f64::INFINITY, f64::min);
    let y1 = y[0] - min_y;
    let y2 = y[y.len() - 1] - min_y;
    let c = 0.5 * (y2 + y1);
    let b = c - y1;
    linspace(-1.0, 1.0, y.len())
        .iter()
        .map(|x| min_y + b * x + c * x * x)
        .collect()
}
