use baselines::classification::{FastChromParams, fastchrom_with_mask};
use baselines::morphology::{MorphologyParams, mor};
use baselines::polynomial::{ModPolyParams, modpoly};
use baselines::prelude::*;
use baselines::smoothing::{SmoothingParams, noise_median};
use baselines::spline::pspline_asls;
use baselines::two_d;
use baselines::whittaker::{AslsParams, WhittakerParams, asls};

#[test]
fn whittaker_builder_matches_free_function_defaults() {
    let y = signal();

    let chained = Baseline::new(&y).asls().fit().unwrap();
    let explicit = asls(&y, AslsParams::default()).unwrap();

    assert_eq!(chained, explicit);
}

#[test]
fn whittaker_builder_setters_match_manual_params() {
    let y = signal();
    let params = AslsParams {
        whittaker: WhittakerParams {
            lambda: 2.5e5,
            max_iter: 12,
            tol: 1.0e-4,
        },
        p: 0.02,
    };

    let chained = Baseline::new(&y)
        .asls()
        .lambda(params.whittaker.lambda)
        .max_iter(params.whittaker.max_iter)
        .tol(params.whittaker.tol)
        .p(params.p)
        .fit()
        .unwrap();
    let explicit = asls(&y, params).unwrap();

    assert_eq!(chained, explicit);
}

#[test]
fn non_whittaker_builders_match_free_functions() {
    let y = signal();
    let morph = MorphologyParams { window_size: 9 };
    let smooth = SmoothingParams {
        window_size: 9,
        max_iter: 4,
    };

    assert_eq!(
        Baseline::new(&y)
            .modpoly()
            .order(2)
            .max_iter(20)
            .tol(1.0e-3)
            .fit()
            .unwrap(),
        modpoly(
            &y,
            ModPolyParams {
                order: 2,
                max_iter: 20,
                tol: 1.0e-3,
            },
        )
        .unwrap()
    );
    assert_eq!(
        Baseline::new(&y)
            .mor()
            .window_size(morph.window_size)
            .fit()
            .unwrap(),
        mor(&y, morph).unwrap()
    );
    assert_eq!(
        Baseline::new(&y)
            .noise_median()
            .window_size(smooth.window_size)
            .max_iter(smooth.max_iter)
            .fit()
            .unwrap(),
        noise_median(&y, smooth).unwrap()
    );
    assert_eq!(
        Baseline::new(&y)
            .pspline_asls()
            .lambda(1.0e5)
            .p(0.02)
            .fit()
            .unwrap(),
        pspline_asls(
            &y,
            AslsParams {
                whittaker: WhittakerParams {
                    lambda: 1.0e5,
                    ..WhittakerParams::default()
                },
                p: 0.02,
            },
        )
        .unwrap()
    );
}

#[test]
fn classification_builder_exposes_mask_result() {
    let y = signal();
    let params = FastChromParams {
        half_window: 6,
        ..FastChromParams::default()
    };

    let chained = Baseline::new(&y)
        .fastchrom()
        .half_window(params.half_window)
        .fit_with_mask()
        .unwrap();
    let explicit = fastchrom_with_mask(&y, params).unwrap();

    assert_eq!(chained, explicit);
}

#[test]
fn two_d_builder_matches_free_function() {
    let data = surface();
    let params = two_d::whittaker::Asls2DParams {
        whittaker: two_d::whittaker::Whittaker2DParams {
            lambda: 8.0e3,
            max_iter: 12,
            tol: 1.0e-3,
            cg_max_iter: 100,
            cg_tol: 1.0e-6,
            ..two_d::whittaker::Whittaker2DParams::default()
        },
        p: 0.02,
    };
    let view = MatrixView::row_major(&data, 8, 9).unwrap();

    let chained = Baseline2D::row_major(&data, 8, 9)
        .unwrap()
        .asls()
        .lambda(params.whittaker.lambda)
        .max_iter(params.whittaker.max_iter)
        .tol(params.whittaker.tol)
        .cg_max_iter(params.whittaker.cg_max_iter)
        .cg_tol(params.whittaker.cg_tol)
        .p(params.p)
        .fit()
        .unwrap();
    let explicit = two_d::whittaker::asls(view, params).unwrap();

    assert_eq!(chained, explicit);
}

fn signal() -> Vec<f64> {
    (0..72)
        .map(|i| {
            let x = i as f64 / 71.0;
            0.7 + 0.25 * x + (-(x - 0.35).powi(2) / 0.002).exp()
        })
        .collect()
}

fn surface() -> Vec<f64> {
    (0..8)
        .flat_map(|row| {
            (0..9).map(move |col| {
                let x = col as f64 / 8.0;
                let y = row as f64 / 7.0;
                0.5 + 0.2 * x + 0.15 * y + (-(x - 0.4).powi(2) / 0.02).exp()
            })
        })
        .collect()
}
