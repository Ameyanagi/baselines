use baselines::MatrixView;
use baselines::two_d::whittaker::{Asls2DParams, Whittaker2DParams, asls};
use ruviz::prelude::*;
use std::error::Error;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

const ROWS: usize = 48;
const COLS: usize = 72;
const OUTPUT_DIR: &str = "target/baselines-ruviz";

fn main() -> std::result::Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(OUTPUT_DIR)?;

    let (observed, true_baseline) = synthetic_surface();
    let input = MatrixView::row_major(&observed, ROWS, COLS)?;
    let fit = asls(
        input,
        Asls2DParams {
            whittaker: Whittaker2DParams {
                lambda: 8.0e3,
                max_iter: 40,
                tol: 1.0e-3,
                cg_max_iter: 500,
                cg_tol: 1.0e-6,
            },
            p: 0.01,
        },
    )?;
    let corrected = fit.corrected(&observed)?;

    let observed_path = output_path("2d_observed.png");
    save_heatmap(
        &observed,
        "2D Observed Surface",
        "intensity",
        ColorMap::viridis(),
        &observed_path,
    )?;

    let baseline_path = output_path("2d_asls_baseline.png");
    save_heatmap(
        &fit.baseline,
        "2D AsLS Baseline",
        "baseline",
        ColorMap::magma(),
        &baseline_path,
    )?;

    let corrected_path = output_path("2d_corrected.png");
    save_heatmap(
        &corrected,
        "2D Corrected Surface",
        "corrected",
        ColorMap::plasma(),
        &corrected_path,
    )?;

    let truth_path = output_path("2d_true_baseline.png");
    save_heatmap(
        &true_baseline,
        "2D True Baseline",
        "baseline",
        ColorMap::magma(),
        &truth_path,
    )?;

    print_output(&observed_path);
    print_output(&baseline_path);
    print_output(&corrected_path);
    print_output(&truth_path);
    Ok(())
}

fn synthetic_surface() -> (Vec<f64>, Vec<f64>) {
    let mut observed = Vec::with_capacity(ROWS * COLS);
    let mut baseline = Vec::with_capacity(ROWS * COLS);

    for row in 0..ROWS {
        let y = row as f64 / (ROWS - 1) as f64;
        for col in 0..COLS {
            let x = col as f64 / (COLS - 1) as f64;
            let base = 0.55 + 0.55 * x + 0.38 * y + 0.15 * (2.0 * PI * x).sin() * (PI * y).cos();
            let signal = gaussian2d(x, y, 0.24, 0.28, 0.055, 0.080, 2.2)
                + gaussian2d(x, y, 0.58, 0.63, 0.070, 0.045, 1.6)
                + gaussian2d(x, y, 0.80, 0.32, 0.045, 0.060, 1.25);
            let ripple = 0.018 * (31.0 * x + 17.0 * y).sin() + 0.012 * (43.0 * x * (y + 0.2)).cos();

            baseline.push(base);
            observed.push(base + signal + ripple);
        }
    }

    (observed, baseline)
}

fn gaussian2d(
    x: f64,
    y: f64,
    center_x: f64,
    center_y: f64,
    sigma_x: f64,
    sigma_y: f64,
    height: f64,
) -> f64 {
    let x_term = ((x - center_x) / sigma_x).powi(2);
    let y_term = ((y - center_y) / sigma_y).powi(2);
    height * (-0.5 * (x_term + y_term)).exp()
}

fn save_heatmap(
    flat: &[f64],
    title: &str,
    label: &str,
    colormap: ColorMap,
    path: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let rows = matrix_rows(flat);
    let config = HeatmapConfig::new()
        .colormap(colormap)
        .interpolation(Interpolation::Nearest)
        .colorbar_label(label);

    Plot::new()
        .title(title)
        .xlabel("column")
        .ylabel("row")
        .max_resolution(1800, 1200)
        .heatmap(&rows, Some(config))
        .save(path)?;
    Ok(())
}

fn matrix_rows(flat: &[f64]) -> Vec<Vec<f64>> {
    flat.chunks(COLS).map(<[f64]>::to_vec).collect()
}

fn output_path(name: &str) -> PathBuf {
    Path::new(OUTPUT_DIR).join(name)
}

fn print_output(path: &Path) {
    println!("wrote {}", path.display());
}
