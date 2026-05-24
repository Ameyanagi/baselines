mod common;

use baselines::prelude::*;
use common::{LineSeries, NormalNoise, ensure_output_dir, gaussian, output_path, print_output};
use ruviz::prelude::*;
use std::error::Error;

const N: usize = 520;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    let (x, y, true_baseline) = synthetic_irregular_spectrum();
    let range_mask = x_range_mask(&x, &[(135.0, 205.0), (505.0, 565.0)]);
    let bool_exclude_mask = x_range_mask(&x, &[(695.0, 750.0), (825.0, 885.0)]);
    let combined_exclude_mask: Vec<bool> = range_mask
        .iter()
        .zip(&bool_exclude_mask)
        .map(|(range, boolean)| *range || *boolean)
        .collect();
    let baseline_mask: Vec<bool> = combined_exclude_mask
        .iter()
        .map(|excluded| !*excluded)
        .collect();

    let index_fit = Baseline::new(&y)
        .arpls()
        .lambda(2.5e5)
        .max_iter(60)
        .tol(1.0e-4)
        .fit()?;

    let xy_range_and_bool_fit = Baseline::new_xy(&x, &y)?
        .arpls()
        .lambda(2.5e5)
        .max_iter(60)
        .tol(1.0e-4)
        .exclude_ranges([(135.0, 205.0), (505.0, 565.0)])
        .exclude_mask(&bool_exclude_mask)?
        .fit()?;

    let xy_baseline_mask_fit = Baseline::new_xy(&x, &y)?
        .asls()
        .lambda(2.5e5)
        .max_iter(60)
        .tol(1.0e-4)
        .p(0.01)
        .baseline_mask(&baseline_mask)?
        .fit()?;

    let fit_path = output_path("xy_whittaker_masks.png");
    save_lines(
        "X-Aware Whittaker With Masks",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "observed irregular-grid data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "true baseline",
                y: &true_baseline,
                color: Color::new(80, 145, 110),
            },
            LineSeries {
                label: "index-space arPLS",
                y: &index_fit.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "x-aware arPLS + exclude masks",
                y: &xy_range_and_bool_fit.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "x-aware AsLS + baseline mask",
                y: &xy_baseline_mask_fit.baseline,
                color: Color::new(84, 151, 160),
            },
        ],
        &fit_path,
    )?;
    print_output(&fit_path);

    let (mask_base, mask_height) = mask_band(&[&y]);
    let range_ribbon = mask_ribbon(&range_mask, mask_base, mask_height);
    let bool_ribbon = mask_ribbon(
        &bool_exclude_mask,
        mask_base - 1.35 * mask_height,
        mask_height,
    );
    let baseline_ribbon = mask_ribbon(&baseline_mask, mask_base - 2.70 * mask_height, mask_height);
    let mask_path = output_path("xy_whittaker_mask_ribbons.png");
    save_lines(
        "X-Aware Whittaker Mask Inputs",
        "x",
        "intensity / mask ribbon",
        &x,
        &[
            LineSeries {
                label: "observed irregular-grid data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "excluded x ranges",
                y: &range_ribbon,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "boolean exclude mask",
                y: &bool_ribbon,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "boolean baseline mask",
                y: &baseline_ribbon,
                color: Color::new(84, 151, 160),
            },
        ],
        &mask_path,
    )?;
    print_output(&mask_path);
    Ok(())
}

fn synthetic_irregular_spectrum() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(N);
    let mut position = 0.0;
    for index in 0..N {
        if index > 0 {
            let t = index as f64 / (N - 1) as f64;
            position += 1.0
                + 0.35 * (9.0 * std::f64::consts::PI * t).sin()
                + 0.20 * (17.0 * std::f64::consts::PI * t).cos();
        }
        x.push(position);
    }
    let scale = 1000.0 / x[N - 1];
    for value in &mut x {
        *value *= scale;
    }

    let true_baseline: Vec<f64> = x
        .iter()
        .map(|&value| {
            3.0 + 0.0045 * value + 1.2 * (-value / 720.0).exp() + 0.35 * (value / 95.0).sin()
        })
        .collect();
    let mut noise = NormalNoise::new(42);
    let y = x
        .iter()
        .zip(&true_baseline)
        .map(|(&value, &baseline)| {
            baseline
                + gaussian(value, 7.5, 170.0, 17.0)
                + gaussian(value, 4.5, 330.0, 22.0)
                + gaussian(value, 6.0, 535.0, 15.0)
                + gaussian(value, 5.2, 720.0, 18.0)
                + gaussian(value, 4.8, 855.0, 13.0)
                + noise.sample(0.08)
        })
        .collect();
    (x, y, true_baseline)
}

fn x_range_mask(x: &[f64], ranges: &[(f64, f64)]) -> Vec<bool> {
    x.iter()
        .map(|value| {
            ranges
                .iter()
                .any(|(start, end)| *value >= *start && *value <= *end)
        })
        .collect()
}

fn mask_band(series: &[&[f64]]) -> (f64, f64) {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for values in series {
        for &value in *values {
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }
    }
    let span = (max_value - min_value).max(1.0);
    (min_value - 0.12 * span, 0.04 * span)
}

fn mask_ribbon(mask: &[bool], base: f64, height: f64) -> Vec<f64> {
    mask.iter()
        .map(|is_set| if *is_set { base + height } else { base })
        .collect()
}

fn save_lines(
    title: &str,
    x_label: &str,
    y_label: &str,
    x: &[f64],
    series: &[LineSeries<'_>],
    path: &std::path::Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let mut plot = Plot::new()
        .title(title)
        .xlabel(x_label)
        .ylabel(y_label)
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best);
    for item in series {
        plot = plot
            .line(&x, &item.y)
            .label(item.label)
            .color(item.color)
            .into();
    }
    plot.save(path)?;
    Ok(())
}
