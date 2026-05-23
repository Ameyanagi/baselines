mod common;

use baselines::classification::{
    FastChromParams, StdDistributionParams, fastchrom_with_mask, std_distribution,
    std_distribution_with_mask,
};
use baselines::morphology::{MorphologyParams, mor};
use baselines::optimizers::{CustomBcParams, custom_bc_with};
use baselines::polynomial::{ImodPolyParams, ModPolyParams, imodpoly, modpoly};
use baselines::smoothing::{SmoothingParams, ria};
use baselines::spline::pspline_arpls;
use baselines::whittaker::{
    ArPlsParams, AsPlsParams, AslsParams, WhittakerParams, arpls, asls_with_history,
    aspls_with_history, iarpls,
};
use common::{
    LineSeries, NormalNoise, PadMode, add3, ensure_output_dir, gaussian, linear_interpolate_masked,
    linspace, median, output_path, pad_edges, percentile, print_output, rolling_std_reflect,
    save_heatmap, save_lines, standard_signal, uniform_filter_reflect,
};
use ruviz::prelude::*;
use std::error::Error;

fn main() -> std::result::Result<(), Box<dyn Error>> {
    ensure_output_dir()?;

    general_algorithm_convergence()?;
    general_noisy_data()?;
    general_masked_data()?;
    general_padding()?;
    general_padding_extrapolate()?;
    general_reuse_baseline()?;
    general_sorted_data()?;
    morphology_half_window_effects()?;
    classification_masks()?;
    classification_fastchrom_threshold()?;
    optimizer_custom_bc()?;
    two_d_individual_axes()?;

    Ok(())
}

fn general_algorithm_convergence() -> std::result::Result<(), Box<dyn Error>> {
    let (x, y, _) = standard_noisy_dataset(1000, 600.0, 0.1, 0);
    let whittaker = WhittakerParams {
        lambda: 5.0e6,
        tol: 1.0e-3,
        max_iter: 20,
    };
    let asls_20 = asls_with_history(&y, AslsParams { whittaker, p: 0.01 })?;
    let aspls_20 = aspls_with_history(
        &y,
        AsPlsParams {
            whittaker,
            asymmetric_coef: 0.5,
        },
    )?;

    let whittaker_100 = WhittakerParams {
        max_iter: 100,
        ..whittaker
    };
    let asls_100 = asls_with_history(
        &y,
        AslsParams {
            whittaker: whittaker_100,
            p: 0.01,
        },
    )?;
    let aspls_100 = aspls_with_history(
        &y,
        AsPlsParams {
            whittaker: whittaker_100,
            asymmetric_coef: 0.5,
        },
    )?;

    let path = output_path("pybaselines_gallery_algorithm_convergence.png");
    save_lines(
        "pybaselines algorithm convergence",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "asls max_iter=20",
                y: &asls_20.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "aspls max_iter=20",
                y: &aspls_20.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "asls max_iter=100",
                y: &asls_100.baseline,
                color: Color::new(232, 168, 72),
            },
            LineSeries {
                label: "aspls max_iter=100",
                y: &aspls_100.baseline,
                color: Color::new(84, 151, 160),
            },
        ],
        &path,
    )?;
    print_output(&path);

    let history_path = output_path("pybaselines_gallery_algorithm_convergence_tolerance.png");
    save_history_lines(
        "pybaselines algorithm convergence tolerance",
        "iteration",
        "log10(relative difference)",
        &[
            HistorySeries {
                label: "asls max_iter=20",
                y: &asls_20.tol_history,
                color: Color::new(218, 111, 76),
            },
            HistorySeries {
                label: "aspls max_iter=20",
                y: &aspls_20.tol_history,
                color: Color::new(118, 85, 148),
            },
            HistorySeries {
                label: "asls max_iter=100",
                y: &asls_100.tol_history,
                color: Color::new(232, 168, 72),
            },
            HistorySeries {
                label: "aspls max_iter=100",
                y: &aspls_100.tol_history,
                color: Color::new(84, 151, 160),
            },
        ],
        &history_path,
    )?;
    print_output(&history_path);
    Ok(())
}

fn general_noisy_data() -> std::result::Result<(), Box<dyn Error>> {
    let (x, y, baseline) = standard_noisy_dataset(1000, 600.0, 0.6, 0);
    let smooth_y = uniform_filter_reflect(&y, 11);
    let regular_modpoly = modpoly(
        &y,
        ModPolyParams {
            order: 3,
            ..ModPolyParams::default()
        },
    )?;
    let smoothed_modpoly = modpoly(
        &smooth_y,
        ModPolyParams {
            order: 3,
            ..ModPolyParams::default()
        },
    )?;
    let regular_imodpoly = imodpoly(
        &y,
        ImodPolyParams {
            order: 3,
            num_std: 0.7,
            ..ImodPolyParams::default()
        },
    )?;
    let smoothed_imodpoly = imodpoly(
        &smooth_y,
        ImodPolyParams {
            order: 3,
            num_std: 0.7,
            ..ImodPolyParams::default()
        },
    )?;

    let path = output_path("pybaselines_gallery_noisy_data.png");
    save_lines(
        "pybaselines noisy data",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "modpoly poly_order=3",
                y: &regular_modpoly.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "smoothed modpoly",
                y: &smoothed_modpoly.baseline,
                color: Color::new(232, 168, 72),
            },
            LineSeries {
                label: "imodpoly poly_order=3 num_std=0.7",
                y: &regular_imodpoly.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "smoothed imodpoly num_std=0.7",
                y: &smoothed_imodpoly.baseline,
                color: Color::new(84, 151, 160),
            },
            LineSeries {
                label: "true baseline",
                y: &baseline,
                color: Color::new(80, 145, 110),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn general_masked_data() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(500.0, 4000.0, 1000);
    let signal: Vec<f64> = x
        .iter()
        .map(|&value| {
            gaussian(value, 8.0, 650.0, 16.0)
                + gaussian(value, 9.0, 1100.0, 50.0)
                + gaussian(value, 8.0, 1350.0, 20.0)
                + gaussian(value, 11.0, 2800.0, 20.0)
                + gaussian(value, 8.0, 2900.0, 20.0)
                + gaussian(value, 5.0, 3400.0, 40.0)
        })
        .collect();
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| 0.08 + 0.00004 * (value - 1000.0) + gaussian(value, 10.0, 1900.0, 800.0))
        .collect();
    let mut noise = NormalNoise::new(123);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.1))
        .collect();
    let bad_region: Vec<bool> = x
        .iter()
        .map(|&value| value > 2000.0 && value < 2500.0)
        .collect();
    let mut y_bad = y.clone();
    for (value, is_bad) in y_bad.iter_mut().zip(&bad_region) {
        if *is_bad {
            *value = 0.5 + noise.sample(0.25);
        }
    }
    let fit_mask: Vec<bool> = x
        .iter()
        .map(|&value| !(1900.0..=2550.0).contains(&value))
        .collect();
    let y_linear = linear_interpolate_masked(&x, &y_bad, &fit_mask);
    let non_masked_arpls = arpls(
        &y_bad,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                ..WhittakerParams::default()
            },
        },
    )?;
    let interpolated_arpls = arpls(
        &y_linear,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                ..WhittakerParams::default()
            },
        },
    )?;
    let mor_linear = mor(&y_linear, MorphologyParams { window_size: 71 })?;

    let path = output_path("pybaselines_gallery_masked_data.png");
    save_lines(
        "pybaselines masked data",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "problematic data",
                y: &y_bad,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "linear interpolation",
                y: &y_linear,
                color: Color::new(80, 145, 110),
            },
            LineSeries {
                label: "arpls non-masked lam=1e5",
                y: &non_masked_arpls.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "arpls interpolated lam=1e5",
                y: &interpolated_arpls.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "mor interpolated half_window=35",
                y: &mor_linear.baseline,
                color: Color::new(84, 151, 160),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn general_padding() -> std::result::Result<(), Box<dyn Error>> {
    let half_window = 80;
    let num_points = 1000;
    let x = linspace(0.0, 1000.0, num_points);
    let line: Vec<f64> = x
        .iter()
        .map(|&value| 10.0 * (-value / 150.0).exp())
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = line.iter().map(|line| line + noise.sample(0.5)).collect();
    let pad_len = 2 * half_window + 1;
    let reflect = pad_edges(&y, pad_len, PadMode::Reflect, (100, 100));
    let edge = pad_edges(&y, pad_len, PadMode::Edge, (100, 100));
    let extrapolate = pad_edges(&y, pad_len, PadMode::Extrapolate, (100, 100));
    let padded_x = linspace(
        -(pad_len as f64),
        (num_points + pad_len - 1) as f64,
        reflect.len(),
    );

    let path = output_path("pybaselines_gallery_padding.png");
    save_lines(
        "pybaselines padding",
        "sample",
        "offset intensity",
        &padded_x,
        &[
            LineSeries {
                label: "reflect",
                y: &reflect,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "edge",
                y: &offset(&edge, 5.0),
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "extrapolate",
                y: &offset(&extrapolate, 10.0),
                color: Color::new(118, 85, 148),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn general_padding_extrapolate() -> std::result::Result<(), Box<dyn Error>> {
    let num_points = 1000;
    let pad_len = 100;
    let x = linspace(0.0, 1000.0, num_points);
    let line: Vec<f64> = x
        .iter()
        .map(|&value| {
            5.0 * (-value / 200.0).exp()
                + gaussian(value, 5.0, 900.0, 20.0)
                + gaussian(value, 5.0, 200.0, 20.0)
        })
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = line.iter().map(|line| line + noise.sample(0.1)).collect();
    let ex_1 = pad_edges(&y, pad_len, PadMode::Extrapolate, (1, 1));
    let ex_100 = pad_edges(&y, pad_len, PadMode::Extrapolate, (100, 100));
    let ex_asym = pad_edges(&y, pad_len, PadMode::Extrapolate, (100, 40));
    let padded_x = linspace(
        -(pad_len as f64),
        (num_points + pad_len - 1) as f64,
        ex_1.len(),
    );

    let path = output_path("pybaselines_gallery_padding_extrapolate.png");
    save_lines(
        "pybaselines padding extrapolate",
        "sample",
        "offset intensity",
        &padded_x,
        &[
            LineSeries {
                label: "extrapolate_window=1",
                y: &ex_1,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "extrapolate_window=100",
                y: &offset(&ex_100, 5.0),
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "extrapolate_window=[100, 40]",
                y: &offset(&ex_asym, 10.0),
                color: Color::new(118, 85, 148),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn general_reuse_baseline() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(0.0, 1000.0, 10_000);
    let signal: Vec<f64> = x
        .iter()
        .map(|&value| {
            gaussian(value, 9.0, 100.0, 12.0)
                + gaussian(value, 6.0, 150.0, 5.0)
                + gaussian(value, 8.0, 350.0, 11.0)
                + gaussian(value, 6.0, 550.0, 6.0)
                + gaussian(value, 13.0, 700.0, 8.0)
                + gaussian(value, 9.0, 880.0, 7.0)
        })
        .collect();
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| 5.0 + 10.0 * (-value / 600.0).exp() + gaussian(value, 15.0, 1000.0, 400.0))
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.1))
        .collect();
    let iarpls_fit = iarpls(
        &y,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: 1.0e5,
                ..WhittakerParams::default()
            },
        },
    )?;
    let mor_fit = mor(&y, MorphologyParams { window_size: 61 })?;
    let ria_fit = ria(
        &y,
        SmoothingParams {
            window_size: 41,
            ..SmoothingParams::default()
        },
    )?;
    let std_fit = std_distribution(
        &y,
        StdDistributionParams {
            half_window: 25,
            ..StdDistributionParams::default()
        },
    )?;

    let path = output_path("pybaselines_gallery_reuse_baseline.png");
    save_lines(
        "pybaselines reuse Baseline",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "iarpls lam=1e5",
                y: &iarpls_fit.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "mor half_window=30",
                y: &mor_fit.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "ria half_window=20",
                y: &ria_fit.baseline,
                color: Color::new(84, 151, 160),
            },
            LineSeries {
                label: "std_distribution half_window=25",
                y: &std_fit.baseline,
                color: Color::new(80, 145, 110),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn general_sorted_data() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(500.0, 4000.0, 1000);
    let signal: Vec<f64> = x
        .iter()
        .map(|&value| {
            gaussian(value, 8.0, 650.0, 16.0)
                + gaussian(value, 9.0, 1100.0, 50.0)
                + gaussian(value, 8.0, 1350.0, 20.0)
                + gaussian(value, 11.0, 2800.0, 20.0)
                + gaussian(value, 8.0, 2900.0, 20.0)
                + gaussian(value, 5.0, 3400.0, 120.0)
        })
        .collect();
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| 0.08 + 0.00004 * value + gaussian(value, 3.0, 3500.0, 1000.0))
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.1))
        .collect();
    let fit = iarpls(
        &y,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: 1.0e6,
                ..WhittakerParams::default()
            },
        },
    )?;
    let mut y_reversed = y.clone();
    y_reversed.reverse();
    let mut fit_reversed = iarpls(
        &y_reversed,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: 1.0e6,
                ..WhittakerParams::default()
            },
        },
    )?
    .baseline;
    fit_reversed.reverse();

    let path = output_path("pybaselines_gallery_sorted_data.png");
    save_lines(
        "pybaselines sorted data",
        "x",
        "absorbance",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "sorted iarpls lam=1e6",
                y: &fit.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "reversed iarpls lam=1e6",
                y: &fit_reversed,
                color: Color::new(118, 85, 148),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn morphology_half_window_effects() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(0.0, 1000.0, 2000);
    let signal: Vec<f64> = x
        .iter()
        .map(|&value| {
            gaussian(value, 9.0, 100.0, 12.0)
                + gaussian(value, 6.0, 180.0, 5.0)
                + gaussian(value, 8.0, 300.0, 11.0)
                + gaussian(value, 15.0, 400.0, 12.0)
                + gaussian(value, 6.0, 550.0, 6.0)
                + gaussian(value, 13.0, 700.0, 8.0)
                + gaussian(value, 9.0, 800.0, 9.0)
                + gaussian(value, 9.0, 880.0, 7.0)
        })
        .collect();
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| 5.0 + gaussian(value, 10.0, 650.0, 150.0))
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.1))
        .collect();
    let fit_30 = mor(&y, MorphologyParams { window_size: 61 })?;
    let fit_60 = mor(&y, MorphologyParams { window_size: 121 })?;
    let fit_120 = mor(&y, MorphologyParams { window_size: 241 })?;

    let path = output_path("pybaselines_gallery_half_window_effects.png");
    save_lines(
        "pybaselines half_window effects",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "half_window=30",
                y: &fit_30.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "half_window=60",
                y: &fit_60.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "half_window=120",
                y: &fit_120.baseline,
                color: Color::new(84, 151, 160),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn classification_masks() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(0.0, 1000.0, 1000);
    let signal = standard_signal(&x);
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| gaussian(value, -6.0, 700.0, 500.0))
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.1))
        .collect();
    let fit_15 = std_distribution_with_mask(
        &y,
        StdDistributionParams {
            half_window: 15,
            smooth_half_window: 10,
            ..StdDistributionParams::default()
        },
    )?;
    let fit_45 = std_distribution_with_mask(
        &y,
        StdDistributionParams {
            half_window: 45,
            smooth_half_window: 10,
            ..StdDistributionParams::default()
        },
    )?;

    let path = output_path("pybaselines_gallery_classifier_masks.png");
    save_lines(
        "pybaselines classification masks",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "std_distribution half_window=15",
                y: &fit_15.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "std_distribution half_window=45",
                y: &fit_45.baseline,
                color: Color::new(118, 85, 148),
            },
        ],
        &path,
    )?;
    print_output(&path);

    let mask_15 = mask_as_float(&fit_15.mask);
    let mask_45 = mask_as_float(&fit_45.mask);
    let mask_path = output_path("pybaselines_gallery_classifier_mask_diagnostics.png");
    save_lines(
        "pybaselines classification mask diagnostics",
        "x",
        "baseline mask",
        &x,
        &[
            LineSeries {
                label: "half_window=15 mask",
                y: &mask_15,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "half_window=45 mask",
                y: &mask_45,
                color: Color::new(118, 85, 148),
            },
        ],
        &mask_path,
    )?;
    print_output(&mask_path);
    Ok(())
}

fn classification_fastchrom_threshold() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(0.0, 1000.0, 1000);
    let signal = standard_signal(&x);
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| gaussian(value, 6.0, 400.0, 500.0))
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.2))
        .collect();
    let rolling_std = rolling_std_reflect(&y, 15);
    let fixed_threshold = 1.5;
    let default_threshold = percentile(&rolling_std, 15.0);
    let custom_threshold = median(&rolling_std);
    let fit_default = fastchrom_with_mask(
        &y,
        FastChromParams {
            half_window: 15,
            threshold: None,
            ..FastChromParams::default()
        },
    )?;
    let fit_fixed = fastchrom_with_mask(
        &y,
        FastChromParams {
            half_window: 15,
            threshold: Some(fixed_threshold),
            ..FastChromParams::default()
        },
    )?;
    let fit_custom = fastchrom_with_mask(
        &y,
        FastChromParams {
            half_window: 15,
            threshold: Some(custom_threshold),
            ..FastChromParams::default()
        },
    )?;

    let path = output_path("pybaselines_gallery_fastchrom_threshold.png");
    save_lines(
        "pybaselines fastchrom threshold",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "default threshold",
                y: &fit_default.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "fixed threshold=1.5",
                y: &fit_fixed.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "custom median threshold",
                y: &fit_custom.baseline,
                color: Color::new(84, 151, 160),
            },
        ],
        &path,
    )?;
    print_output(&path);

    let thresholds_default = vec![default_threshold; rolling_std.len()];
    let thresholds_fixed = vec![fixed_threshold; rolling_std.len()];
    let thresholds_custom = vec![custom_threshold; rolling_std.len()];
    let threshold_path = output_path("pybaselines_gallery_fastchrom_rolling_std.png");
    save_lines(
        "pybaselines fastchrom rolling std",
        "x",
        "standard deviation",
        &x,
        &[
            LineSeries {
                label: "rolling std",
                y: &rolling_std,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "default percentile",
                y: &thresholds_default,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "fixed",
                y: &thresholds_fixed,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "custom median",
                y: &thresholds_custom,
                color: Color::new(84, 151, 160),
            },
        ],
        &threshold_path,
    )?;
    print_output(&threshold_path);

    let mask_default = mask_as_float(&fit_default.mask);
    let mask_fixed = mask_as_float(&fit_fixed.mask);
    let mask_custom = mask_as_float(&fit_custom.mask);
    let mask_path = output_path("pybaselines_gallery_fastchrom_masks.png");
    save_lines(
        "pybaselines fastchrom masks",
        "x",
        "baseline mask",
        &x,
        &[
            LineSeries {
                label: "default mask",
                y: &mask_default,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "fixed mask",
                y: &mask_fixed,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "custom median mask",
                y: &mask_custom,
                color: Color::new(84, 151, 160),
            },
        ],
        &mask_path,
    )?;
    print_output(&mask_path);
    Ok(())
}

fn optimizer_custom_bc() -> std::result::Result<(), Box<dyn Error>> {
    let x = linspace(20.0, 1000.0, 1000);
    let signal: Vec<f64> = x
        .iter()
        .map(|&value| {
            gaussian(value, 6.0, 240.0, 5.0)
                + gaussian(value, 8.0, 350.0, 11.0)
                + gaussian(value, 15.0, 400.0, 18.0)
                + gaussian(value, 6.0, 550.0, 6.0)
                + gaussian(value, 13.0, 700.0, 8.0)
                + gaussian(value, 9.0, 800.0, 9.0)
                + gaussian(value, 9.0, 880.0, 7.0)
        })
        .collect();
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| {
            5.0 + 6.0 * (-(value - 40.0) / 30.0).exp() + gaussian(value, 5.0, 1000.0, 300.0)
        })
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(signal, baseline)| signal + baseline + noise.sample(0.1))
        .collect();
    let lam_flexible = 1.0e2;
    let lam_stiff = 5.0e5;
    let flexible = arpls(
        &y,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: lam_flexible,
                ..WhittakerParams::default()
            },
        },
    )?;
    let stiff = arpls(
        &y,
        ArPlsParams {
            whittaker: WhittakerParams {
                lambda: lam_stiff,
                ..WhittakerParams::default()
            },
        },
    )?;
    let crossover_index = x
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| (*left - 160.0).abs().total_cmp(&(*right - 160.0).abs()))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let fit = custom_bc_with(
        &y,
        CustomBcParams {
            regions: vec![(Some(crossover_index), None)],
            sampling: 15,
            asls: AslsParams::default(),
            smooth_lambda: Some(1.0e1),
        },
        |values| {
            arpls(
                values,
                ArPlsParams {
                    whittaker: WhittakerParams {
                        lambda: lam_flexible,
                        ..WhittakerParams::default()
                    },
                },
            )
        },
    )?;

    let path = output_path("pybaselines_gallery_custom_bc_whittaker.png");
    save_lines(
        "pybaselines custom_bc whittaker",
        "x",
        "intensity",
        &x,
        &[
            LineSeries {
                label: "data",
                y: &y,
                color: Color::new(43, 70, 104),
            },
            LineSeries {
                label: "flexible lam=1e2",
                y: &flexible.baseline,
                color: Color::new(218, 111, 76),
            },
            LineSeries {
                label: "stiff lam=5e5",
                y: &stiff.baseline,
                color: Color::new(118, 85, 148),
            },
            LineSeries {
                label: "custom_bc",
                y: &fit.baseline,
                color: Color::new(84, 151, 160),
            },
            LineSeries {
                label: "true baseline",
                y: &baseline,
                color: Color::new(80, 145, 110),
            },
        ],
        &path,
    )?;
    print_output(&path);
    Ok(())
}

fn two_d_individual_axes() -> std::result::Result<(), Box<dyn Error>> {
    let len_temperature = 25;
    let wavenumber = linspace(50.0, 300.0, 1000);
    let temperature = linspace(25.0, 100.0, len_temperature);
    let mut data = Vec::with_capacity(len_temperature * wavenumber.len());
    let mut noise = NormalNoise::new(0);
    for (index, &t_value) in temperature.iter().enumerate() {
        for &wave in &wavenumber {
            let signal = gaussian(
                wave,
                11.0 * (1.0 - index as f64 / len_temperature as f64),
                90.0,
                3.0,
            ) + gaussian(
                wave,
                12.0 * (1.0 - index as f64 / len_temperature as f64),
                110.0,
                6.0,
            ) + gaussian(wave, 13.0, 210.0, 8.0);
            let baseline = 100.0 + 0.005 * wave + 0.0001 * (wave - 120.0).powi(2) + 0.08 * t_value;
            data.push(signal + baseline + noise.sample(0.1));
        }
    }
    let mut baseline = vec![0.0; data.len()];
    for row in 0..len_temperature {
        let start = row * wavenumber.len();
        let end = start + wavenumber.len();
        let fit = pspline_arpls(
            &data[start..end],
            ArPlsParams {
                whittaker: WhittakerParams {
                    lambda: 1.0e4,
                    ..WhittakerParams::default()
                },
            },
        )?;
        baseline[start..end].copy_from_slice(&fit.baseline);
    }
    let corrected: Vec<f64> = data
        .iter()
        .zip(&baseline)
        .map(|(observed, baseline)| observed - baseline)
        .collect();

    let observed_path = output_path("pybaselines_gallery_2d_individual_axes_observed.png");
    save_heatmap(
        &data,
        len_temperature,
        wavenumber.len(),
        "pybaselines 2D individual_axes observed",
        "intensity",
        &observed_path,
    )?;
    print_output(&observed_path);

    let corrected_path = output_path("pybaselines_gallery_2d_individual_axes_corrected.png");
    save_heatmap(
        &corrected,
        len_temperature,
        wavenumber.len(),
        "pybaselines 2D individual_axes corrected",
        "corrected",
        &corrected_path,
    )?;
    print_output(&corrected_path);
    Ok(())
}

fn standard_noisy_dataset(
    count: usize,
    decay: f64,
    noise_sigma: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x = linspace(0.0, 1000.0, count);
    let signal = standard_signal(&x);
    let baseline: Vec<f64> = x
        .iter()
        .map(|&value| 5.0 + 10.0 * (-value / decay).exp())
        .collect();
    let mut noise = NormalNoise::new(seed);
    let y = add3(
        &signal,
        &baseline,
        &baseline
            .iter()
            .map(|_| noise.sample(noise_sigma))
            .collect::<Vec<_>>(),
    );
    (x, y, baseline)
}

fn offset(values: &[f64], amount: f64) -> Vec<f64> {
    values.iter().map(|value| value + amount).collect()
}

fn mask_as_float(mask: &[bool]) -> Vec<f64> {
    mask.iter()
        .map(|is_baseline| if *is_baseline { 1.0 } else { 0.0 })
        .collect()
}

struct HistorySeries<'a> {
    label: &'a str,
    y: &'a [f64],
    color: Color,
}

fn save_history_lines(
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[HistorySeries<'_>],
    path: &std::path::Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let mut plot = Plot::new()
        .title(title)
        .xlabel(x_label)
        .ylabel(y_label)
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best);
    for item in series {
        let x = history_iterations(item.y);
        let y = log10_history(item.y);
        plot = plot.line(&x, &y).label(item.label).color(item.color).into();
    }
    plot.save(path)?;
    Ok(())
}

fn history_iterations(values: &[f64]) -> Vec<f64> {
    (1..=values.len()).map(|index| index as f64).collect()
}

fn log10_history(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|value| value.max(f64::MIN_POSITIVE).log10())
        .collect()
}
