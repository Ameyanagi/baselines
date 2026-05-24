use approx::assert_abs_diff_eq;
use baselines::classification::{FastChromParams, fastchrom_with_mask};
use baselines::morphology::{MorphologyParams, mor};
use baselines::polynomial::{ModPolyParams, modpoly};
use baselines::prelude::*;
use baselines::smoothing::{SmoothingParams, noise_median};
use baselines::spline::pspline_asls;
use baselines::two_d;
use baselines::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, WhittakerParams, airpls, arpls, asls,
    aspls, brpls, derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};

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
fn xy_whittaker_uniform_grid_matches_existing_methods() {
    let y = signal();
    let x: Vec<f64> = (0..y.len()).map(|index| 5.0 + 2.5 * index as f64).collect();
    let xy = Baseline::new_xy(&x, &y).unwrap();

    assert_fit_close(
        &xy.asls().fit().unwrap(),
        &asls(&y, AslsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.airpls().fit().unwrap(),
        &airpls(&y, AirPlsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.arpls().fit().unwrap(),
        &arpls(&y, ArPlsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.iasls().fit().unwrap(),
        &iasls(&y, IaslsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.drpls().fit().unwrap(),
        &drpls(&y, DrPlsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.iarpls().fit().unwrap(),
        &iarpls(&y, IarPlsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.aspls().fit().unwrap(),
        &aspls(&y, AsPlsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.psalsa().fit().unwrap(),
        &psalsa(&y, PsalsaParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.derpsalsa().fit().unwrap(),
        &derpsalsa(&y, DerPsalsaParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.brpls().fit().unwrap(),
        &brpls(&y, BrPlsParams::default()).unwrap(),
    );
    assert_fit_close(
        &xy.lsrpls().fit().unwrap(),
        &lsrpls(&y, LsrPlsParams::default()).unwrap(),
    );
}

#[test]
fn xy_whittaker_masks_accept_ranges_and_boolean_slices() {
    let y = signal();
    let x: Vec<f64> = (0..y.len()).map(|index| index as f64).collect();
    let mut exclude = vec![false; y.len()];
    for value in &mut exclude[24..34] {
        *value = true;
    }
    let baseline_mask: Vec<bool> = exclude.iter().map(|excluded| !excluded).collect();

    let range_fit = Baseline::new_xy(&x, &y)
        .unwrap()
        .asls()
        .lambda(1.0e5)
        .exclude_range(24.0, 33.0)
        .fit()
        .unwrap();
    let exclude_fit = Baseline::new_xy(&x, &y)
        .unwrap()
        .asls()
        .lambda(1.0e5)
        .exclude_mask(&exclude)
        .unwrap()
        .fit()
        .unwrap();
    let baseline_fit = Baseline::new_xy(&x, &y)
        .unwrap()
        .asls()
        .lambda(1.0e5)
        .baseline_mask(&baseline_mask)
        .unwrap()
        .fit()
        .unwrap();

    assert_fit_close(&range_fit, &exclude_fit);
    assert_fit_close(&exclude_fit, &baseline_fit);
}

#[test]
fn xy_whittaker_all_methods_accept_masks() {
    let y = signal();
    let x: Vec<f64> = (0..y.len())
        .map(|index| {
            let base = index as f64;
            if index < y.len() / 2 {
                base
            } else {
                base + 0.25
            }
        })
        .collect();
    let mut exclude = vec![false; y.len()];
    exclude[30] = true;
    exclude[31] = true;
    let xy = Baseline::new_xy(&x, &y).unwrap();

    let fits = [
        xy.asls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.airpls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.arpls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.iasls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.drpls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.iarpls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.aspls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.psalsa().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.derpsalsa()
            .exclude_mask(&exclude)
            .unwrap()
            .fit()
            .unwrap(),
        xy.brpls().exclude_mask(&exclude).unwrap().fit().unwrap(),
        xy.lsrpls().exclude_mask(&exclude).unwrap().fit().unwrap(),
    ];

    for fit in fits {
        assert_eq!(fit.baseline.len(), y.len());
        assert!(fit.baseline.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn xy_whittaker_validates_x_and_masks() {
    let y = signal();
    let x: Vec<f64> = (0..y.len()).map(|index| index as f64).collect();
    let mut non_increasing = x.clone();
    non_increasing[10] = non_increasing[9];

    assert!(Baseline::new_xy(&non_increasing, &y).is_err());
    assert!(
        Baseline::new_xy(&x, &y)
            .unwrap()
            .asls()
            .exclude_mask(&[true, false])
            .is_err()
    );
    assert!(
        Baseline::new_xy(&x, &y)
            .unwrap()
            .asls()
            .exclude_range(x[0], x[y.len() - 1])
            .fit()
            .is_err()
    );
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

fn assert_fit_close(left: &Fit, right: &Fit) {
    assert_eq!(left.baseline.len(), right.baseline.len());
    for (left, right) in left.baseline.iter().zip(&right.baseline) {
        assert_abs_diff_eq!(*left, *right, epsilon = 1.0e-6);
    }
}
