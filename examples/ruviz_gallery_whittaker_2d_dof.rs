//! 2D Whittaker eigendecomposition gallery rendered with ruviz.
//!
//! The grid, Gaussian peaks, polynomial and sinusoidal baselines, lambda
//! tuples, eigen counts, `return_dof=True`, `tol=1e-3`, and `max_iter=50`
//! mirror:
//! <https://pybaselines.readthedocs.io/en/latest/generated/examples/two_d/plot_whittaker_2d_dof.html>
//! Inspired by the linked pybaselines example; pybaselines is used as a
//! behavioral and documentation reference only.

mod common;

use baselines::MatrixView;
use baselines::two_d::whittaker::{
    ArPls2DEigenParams, ArPls2DParams, Whittaker2DEigenFit, Whittaker2DEigenParams,
    Whittaker2DParams, arpls, arpls_eigen,
};
use common::{NormalNoise, ensure_output_dir, linspace, output_path, print_output, save_heatmap};
use std::error::Error;
use std::time::Instant;

const ROWS: usize = 100;
const COLS: usize = 100;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    let x = linspace(-20.0, 20.0, ROWS);
    let z = linspace(-20.0, 30.0, COLS);
    let data = make_data(&x, &z);

    save_surface(
        &data.polynomial_baseline,
        "Actual Polynomial Baseline",
        "gallery_2d_whittaker_actual_polynomial.png",
    )?;
    save_surface(
        &data.sine_baseline,
        "Actual Sinusoidal Baseline",
        "gallery_2d_whittaker_actual_sinusoidal.png",
    )?;

    let lam_poly = (1.0e2, 1.0e4);
    let lam_sine = (1.0e2, 1.0e0);

    let start = Instant::now();
    let analytical_poly = fit_analytical(&data.y_poly, lam_poly)?;
    let analytical_sine = fit_analytical(&data.y_sine, lam_sine)?;
    println!("Analytical solutions:");
    println!("  Time: {:.3} seconds", start.elapsed().as_secs_f64());
    println!(
        "  Mean-squared-error, polynomial: {:.5}",
        mean_squared_error(&analytical_poly.baseline, &data.polynomial_baseline)
    );
    println!(
        "  Mean-squared-error, sinusoidal: {:.5}",
        mean_squared_error(&analytical_sine.baseline, &data.sine_baseline)
    );

    save_surface(
        &analytical_poly.baseline,
        "Analytical Polynomial Baseline",
        "gallery_2d_whittaker_analytical_polynomial.png",
    )?;
    save_surface(
        &analytical_sine.baseline,
        "Analytical Sinusoidal Baseline",
        "gallery_2d_whittaker_analytical_sinusoidal.png",
    )?;

    let start = Instant::now();
    let eigen_poly_40 = fit_eigen(&data.y_poly, lam_poly, (40, 40))?;
    let eigen_sine_40 = fit_eigen(&data.y_sine, lam_sine, (40, 40))?;
    println!("40x40 Eigenvalues:");
    println!("  Time: {:.3} seconds", start.elapsed().as_secs_f64());
    println!(
        "  Mean-squared-error, polynomial: {:.5}",
        mean_squared_error(&eigen_poly_40.baseline, &data.polynomial_baseline)
    );
    println!(
        "  Mean-squared-error, sinusoidal: {:.5}",
        mean_squared_error(&eigen_sine_40.baseline, &data.sine_baseline)
    );

    save_surface(
        &eigen_poly_40.baseline,
        "40x40 Eigenvalues Polynomial Baseline",
        "gallery_2d_whittaker_eigen_40_polynomial.png",
    )?;
    save_surface(
        &eigen_sine_40.baseline,
        "40x40 Eigenvalues Sinusoidal Baseline",
        "gallery_2d_whittaker_eigen_40_sinusoidal.png",
    )?;
    save_dof(
        &eigen_poly_40,
        "Effective Degrees of Freedom for Polynomial Baseline",
        "gallery_2d_whittaker_dof_polynomial.png",
    )?;
    save_dof(
        &eigen_sine_40,
        "Effective Degrees of Freedom for Sinusoidal Baseline",
        "gallery_2d_whittaker_dof_sinusoidal.png",
    )?;

    let start = Instant::now();
    let eigen_poly_reduced = fit_eigen(&data.y_poly, lam_poly, (10, 4))?;
    let eigen_sine_reduced = fit_eigen(&data.y_sine, lam_sine, (8, 35))?;
    println!("10x4 Eigenvalues for polynomial, 8x35 for sinusoidal:");
    println!("  Time: {:.3} seconds", start.elapsed().as_secs_f64());
    println!(
        "  Mean-squared-error, polynomial: {:.5}",
        mean_squared_error(&eigen_poly_reduced.baseline, &data.polynomial_baseline)
    );
    println!(
        "  Mean-squared-error, sinusoidal: {:.5}",
        mean_squared_error(&eigen_sine_reduced.baseline, &data.sine_baseline)
    );

    save_surface(
        &eigen_poly_reduced.baseline,
        "10x4 Eigenvalues Polynomial Baseline",
        "gallery_2d_whittaker_eigen_reduced_polynomial.png",
    )?;
    save_surface(
        &eigen_sine_reduced.baseline,
        "8x35 Eigenvalues Sinusoidal Baseline",
        "gallery_2d_whittaker_eigen_reduced_sinusoidal.png",
    )?;

    let start = Instant::now();
    let eigen_poly_underfit = fit_eigen(&data.y_poly, lam_poly, (3, 3))?;
    let eigen_sine_underfit = fit_eigen(&data.y_sine, lam_sine, (5, 12))?;
    println!("3x3 Eigenvalues for polynomial, 5x12 for sinusoidal:");
    println!("  Time: {:.3} seconds", start.elapsed().as_secs_f64());
    println!(
        "  Mean-squared-error, polynomial: {:.5}",
        mean_squared_error(&eigen_poly_underfit.baseline, &data.polynomial_baseline)
    );
    println!(
        "  Mean-squared-error, sinusoidal: {:.5}",
        mean_squared_error(&eigen_sine_underfit.baseline, &data.sine_baseline)
    );

    save_surface(
        &eigen_poly_underfit.baseline,
        "3x3 Eigenvalues Polynomial Baseline",
        "gallery_2d_whittaker_eigen_underfit_polynomial.png",
    )?;
    save_surface(
        &eigen_sine_underfit.baseline,
        "5x12 Eigenvalues Sinusoidal Baseline",
        "gallery_2d_whittaker_eigen_underfit_sinusoidal.png",
    )?;

    Ok(())
}

fn fit_analytical(
    y: &[f64],
    lambda: (f64, f64),
) -> std::result::Result<baselines::Fit2D, Box<dyn Error>> {
    let input = MatrixView::row_major(y, ROWS, COLS)?;
    Ok(arpls(
        input,
        ArPls2DParams {
            whittaker: Whittaker2DParams {
                lambda: 1.0,
                lambda_rows: Some(lambda.0),
                lambda_cols: Some(lambda.1),
                max_iter: 50,
                tol: 1.0e-3,
                cg_max_iter: 1000,
                cg_tol: 1.0e-6,
            },
        },
    )?)
}

fn fit_eigen(
    y: &[f64],
    lambda: (f64, f64),
    num_eigens: (usize, usize),
) -> std::result::Result<Whittaker2DEigenFit, Box<dyn Error>> {
    let input = MatrixView::row_major(y, ROWS, COLS)?;
    Ok(arpls_eigen(
        input,
        ArPls2DEigenParams {
            whittaker: Whittaker2DEigenParams {
                lambda: 1.0,
                lambda_rows: Some(lambda.0),
                lambda_cols: Some(lambda.1),
                num_eigens,
                return_dof: true,
                max_iter: 50,
                tol: 1.0e-3,
                cg_max_iter: 500,
                cg_tol: 1.0e-7,
                ..Whittaker2DEigenParams::default()
            },
        },
    )?)
}

fn save_surface(
    values: &[f64],
    title: &str,
    file_name: &str,
) -> std::result::Result<(), Box<dyn Error>> {
    let path = output_path(file_name);
    save_heatmap(values, ROWS, COLS, title, "intensity", &path)?;
    print_output(&path);
    Ok(())
}

fn save_dof(
    fit: &Whittaker2DEigenFit,
    title: &str,
    file_name: &str,
) -> std::result::Result<(), Box<dyn Error>> {
    let Some(dof) = &fit.dof else {
        return Ok(());
    };
    let (rows, cols) = fit.dof_shape();
    let path = output_path(file_name);
    save_heatmap(dof, rows, cols, title, "degrees of freedom", &path)?;
    print_output(&path);
    Ok(())
}

struct ExampleData {
    polynomial_baseline: Vec<f64>,
    sine_baseline: Vec<f64>,
    y_poly: Vec<f64>,
    y_sine: Vec<f64>,
}

fn make_data(x: &[f64], z: &[f64]) -> ExampleData {
    let mut signal = Vec::with_capacity(ROWS * COLS);
    let mut polynomial_baseline = Vec::with_capacity(ROWS * COLS);
    let mut sine_baseline = Vec::with_capacity(ROWS * COLS);
    let mut noise = NormalNoise::new(0);

    for &x_value in x {
        for &z_value in z {
            signal.push(
                gaussian2d(x_value, z_value, 12.0, -5.0, -5.0)
                    + gaussian2d(x_value, z_value, 11.0, 3.0, 2.0)
                    + gaussian2d(x_value, z_value, 13.0, 8.0, 8.0)
                    + gaussian2d(x_value, z_value, 8.0, 9.0, 18.0)
                    + gaussian2d(x_value, z_value, 16.0, -8.0, 8.0),
            );
            polynomial_baseline.push(
                0.1 + 0.05 * x_value + 0.005 * z_value - 0.008 * x_value * z_value
                    + 0.0006 * x_value.powi(2)
                    + 0.0003 * z_value.powi(2),
            );
            sine_baseline.push((x_value / 5.0).sin() + (z_value / 2.0).cos());
        }
    }

    let mut y_poly = Vec::with_capacity(ROWS * COLS);
    let mut y_sine = Vec::with_capacity(ROWS * COLS);
    for ((signal_value, poly), sine) in signal.iter().zip(&polynomial_baseline).zip(&sine_baseline)
    {
        let noise_value = noise.sample(0.1);
        y_poly.push(signal_value + poly + noise_value);
        y_sine.push(signal_value + sine + noise_value);
    }

    ExampleData {
        polynomial_baseline,
        sine_baseline,
        y_poly,
        y_sine,
    }
}

fn gaussian2d(x: f64, z: f64, height: f64, center_x: f64, center_z: f64) -> f64 {
    height * (-0.5 * ((x - center_x).powi(2) + (z - center_z).powi(2))).exp()
}

fn mean_squared_error(fit_baseline: &[f64], real_baseline: &[f64]) -> f64 {
    fit_baseline
        .iter()
        .zip(real_baseline)
        .map(|(fit, real)| {
            let difference = fit - real;
            difference * difference
        })
        .sum::<f64>()
        / fit_baseline.len() as f64
}
