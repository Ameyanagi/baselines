use baselines::whittaker::{ArPlsParams, AslsParams, WhittakerParams, arpls, asls};
use ruviz::prelude::*;
use std::error::Error;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

const N: usize = 420;
const OUTPUT_DIR: &str = "target/baselines-ruviz";

fn main() -> std::result::Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(OUTPUT_DIR)?;

    let (x, y, true_baseline) = synthetic_spectrum();
    let whittaker = WhittakerParams {
        lambda: 5.0e5,
        max_iter: 60,
        tol: 1.0e-4,
    };

    let asls_fit = asls(&y, AslsParams { whittaker, p: 0.01 })?;
    let arpls_fit = arpls(&y, ArPlsParams { whittaker })?;
    let asls_corrected = asls_fit.corrected(&y)?;
    let arpls_corrected = arpls_fit.corrected(&y)?;
    let true_corrected: Vec<f64> = y
        .iter()
        .zip(&true_baseline)
        .map(|(observed, baseline)| observed - baseline)
        .collect();

    let baseline_path = output_path("1d_baselines.png");
    Plot::new()
        .title("1D Whittaker Baselines")
        .xlabel("x")
        .ylabel("intensity")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best)
        .line(&x, &y)
        .label("observed")
        .color(Color::new(43, 70, 104))
        .line(&x, &true_baseline)
        .label("true baseline")
        .color(Color::new(80, 145, 110))
        .line(&x, &asls_fit.baseline)
        .label("AsLS")
        .color(Color::new(218, 111, 76))
        .line(&x, &arpls_fit.baseline)
        .label("arPLS")
        .color(Color::new(118, 85, 148))
        .save(&baseline_path)?;

    let corrected_path = output_path("1d_corrected.png");
    Plot::new()
        .title("1D Corrected Signals")
        .xlabel("x")
        .ylabel("corrected intensity")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best)
        .line(&x, &true_corrected)
        .label("observed - true baseline")
        .color(Color::new(80, 145, 110))
        .line(&x, &asls_corrected)
        .label("AsLS corrected")
        .color(Color::new(218, 111, 76))
        .line(&x, &arpls_corrected)
        .label("arPLS corrected")
        .color(Color::new(118, 85, 148))
        .save(&corrected_path)?;

    print_output(&baseline_path);
    print_output(&corrected_path);
    Ok(())
}

fn synthetic_spectrum() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..N).map(|i| i as f64 / (N - 1) as f64).collect();
    let true_baseline: Vec<f64> = x
        .iter()
        .map(|&value| 0.45 + 0.85 * value + 0.18 * (4.0 * PI * value).sin())
        .collect();
    let y = x
        .iter()
        .zip(&true_baseline)
        .map(|(&value, &baseline)| {
            baseline
                + gaussian(value, 0.18, 0.016, 2.1)
                + gaussian(value, 0.44, 0.026, 1.35)
                + gaussian(value, 0.72, 0.020, 1.7)
                + 0.025 * (97.0 * value).sin()
                + 0.015 * (53.0 * value).cos()
        })
        .collect();
    (x, y, true_baseline)
}

fn gaussian(x: f64, center: f64, sigma: f64, height: f64) -> f64 {
    height * (-0.5 * ((x - center) / sigma).powi(2)).exp()
}

fn output_path(name: &str) -> PathBuf {
    Path::new(OUTPUT_DIR).join(name)
}

fn print_output(path: &Path) {
    println!("wrote {}", path.display());
}
