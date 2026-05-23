use std::collections::BTreeMap;

use baselines::MatrixView;
use baselines::two_d::morphology::{
    Morphology2DParams, imor, mor, noise_median, rolling_ball, tophat,
};
use baselines::two_d::optimizers::{
    AdaptiveMinmax2DParams, CollabPls2DParams, IndividualAxes2DParams, adaptive_minmax, collab_pls,
    individual_axes,
};
use baselines::two_d::polynomial::{
    ImodPoly2DParams, ModPoly2DParams, PenalizedPoly2DParams, Poly2DParams, QuantReg2DParams,
    imodpoly, modpoly, penalized_poly, poly, quant_reg,
};
use baselines::two_d::spline::{
    Irsqr2DParams, MixtureModel2DParams, PsplineAirPls2DParams, PsplineArPls2DParams,
    PsplineAsls2DParams, PsplineBrPls2DParams, PsplineIarPls2DParams, PsplineIasls2DParams,
    PsplineLsrPls2DParams, PsplinePsalsa2DParams, Spline2DParams, irsqr, mixture_model,
    pspline_airpls, pspline_arpls, pspline_asls, pspline_brpls, pspline_iarpls, pspline_iasls,
    pspline_lsrpls, pspline_psalsa,
};
use baselines::two_d::whittaker::{
    AirPls2DParams, ArPls2DParams, AsPls2DParams, Asls2DParams, BrPls2DParams, DrPls2DParams,
    IarPls2DParams, Iasls2DParams, LsrPls2DParams, Psalsa2DParams, Whittaker2DParams, airpls,
    arpls, asls, aspls, brpls, drpls, iarpls, iasls, lsrpls, psalsa,
};
use baselines::whittaker::{AslsParams, WhittakerParams};
use serde::Deserialize;

const EXPECTED_PYBASELINES_2D_METHODS: &[&str] = &[
    "adaptive_minmax",
    "airpls",
    "arpls",
    "asls",
    "aspls",
    "brpls",
    "collab_pls",
    "drpls",
    "iarpls",
    "iasls",
    "imodpoly",
    "imor",
    "individual_axes",
    "irsqr",
    "lsrpls",
    "mixture_model",
    "modpoly",
    "mor",
    "noise_median",
    "penalized_poly",
    "poly",
    "psalsa",
    "pspline_airpls",
    "pspline_arpls",
    "pspline_asls",
    "pspline_brpls",
    "pspline_iarpls",
    "pspline_iasls",
    "pspline_lsrpls",
    "pspline_psalsa",
    "quant_reg",
    "rolling_ball",
    "tophat",
];

#[derive(Debug, Deserialize)]
struct Fixture2D {
    pybaselines_version: String,
    pybaselines_methods: Vec<String>,
    shape: [usize; 2],
    signal: Vec<f64>,
    baselines: BTreeMap<String, Vec<f64>>,
    cases: BTreeMap<String, Fixture2DCase>,
}

#[derive(Debug, Deserialize)]
struct Fixture2DCase {
    shape: [usize; 2],
    signal: Vec<f64>,
    baselines: BTreeMap<String, Vec<f64>>,
}

#[test]
fn pybaselines_2d_method_list_has_not_drifted() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();
    assert_eq!(fixture.pybaselines_version, "1.2.1");
    assert_eq!(
        fixture.pybaselines_methods, EXPECTED_PYBASELINES_2D_METHODS,
        "pinned pybaselines Baseline2D public method list changed"
    );
}

#[test]
fn reference_2d_fixture_represents_every_method() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();

    let represented = fixture.baselines.keys().collect::<Vec<_>>();
    for method in EXPECTED_PYBASELINES_2D_METHODS {
        if *method == "collab_pls" {
            assert!(
                fixture.baselines.contains_key("collab_pls_0")
                    && fixture.baselines.contains_key("collab_pls_1"),
                "collab_pls must be represented by collaborative output baselines"
            );
        } else {
            assert!(
                fixture.baselines.contains_key(*method),
                "missing 2D baseline fixture for {method}"
            );
        }
    }
    for name in represented {
        assert!(
            EXPECTED_PYBASELINES_2D_METHODS.contains(&name.as_str())
                || name == "collab_pls_0"
                || name == "collab_pls_1",
            "unexpected 2D baseline fixture {name}"
        );
    }
}

#[test]
fn two_d_fixture_arrays_are_row_major_and_finite() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();

    assert_case_arrays(
        "reference",
        fixture.shape,
        &fixture.signal,
        &fixture.baselines,
    );
    assert_eq!(
        fixture.cases.keys().map(String::as_str).collect::<Vec<_>>(),
        ["noisy", "reference", "ridge_valley", "tilted_plane"]
    );
    for (case_name, case) in &fixture.cases {
        assert_case_arrays(case_name, case.shape, &case.signal, &case.baselines);
    }
}

#[test]
fn native_2d_morphology_tracks_reference_fixture() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();
    let [rows, cols] = fixture.shape;
    let input = MatrixView::row_major(&fixture.signal, rows, cols).unwrap();
    let params = Morphology2DParams {
        window_rows: 7,
        window_cols: 7,
    };

    assert_baseline_close(
        "rolling_ball",
        &fixture.baselines,
        rolling_ball(input, params).unwrap().baseline,
        1e-12,
    );
    assert_baseline_close(
        "tophat",
        &fixture.baselines,
        tophat(input, params).unwrap().baseline,
        1e-12,
    );
    assert_baseline_close(
        "mor",
        &fixture.baselines,
        mor(input, params).unwrap().baseline,
        1e-12,
    );
    assert_baseline_close(
        "imor",
        &fixture.baselines,
        imor(input, params).unwrap().baseline,
        1.4e-1,
    );
    assert_baseline_close(
        "noise_median",
        &fixture.baselines,
        noise_median(input, params).unwrap().baseline,
        6e-2,
    );
}

#[test]
fn native_2d_polynomial_tracks_reference_fixture() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();
    let [rows, cols] = fixture.shape;
    let input = MatrixView::row_major(&fixture.signal, rows, cols).unwrap();

    assert_baseline_close(
        "poly",
        &fixture.baselines,
        poly(input, Poly2DParams { order: 2 }).unwrap().baseline,
        1e-12,
    );
    assert_baseline_close(
        "modpoly",
        &fixture.baselines,
        modpoly(
            input,
            ModPoly2DParams {
                order: 2,
                max_iter: 20,
                tol: 1e-3,
            },
        )
        .unwrap()
        .baseline,
        6e-2,
    );
    assert_baseline_close(
        "imodpoly",
        &fixture.baselines,
        imodpoly(
            input,
            ImodPoly2DParams {
                order: 2,
                max_iter: 20,
                tol: 1e-3,
            },
        )
        .unwrap()
        .baseline,
        3e-2,
    );
    assert_baseline_close(
        "penalized_poly",
        &fixture.baselines,
        penalized_poly(
            input,
            PenalizedPoly2DParams {
                order: 2,
                max_iter: 20,
                ..PenalizedPoly2DParams::default()
            },
        )
        .unwrap()
        .baseline,
        1e-12,
    );
    assert_baseline_close(
        "quant_reg",
        &fixture.baselines,
        quant_reg(
            input,
            QuantReg2DParams {
                order: 2,
                quantile: 0.05,
                max_iter: 20,
                ..QuantReg2DParams::default()
            },
        )
        .unwrap()
        .baseline,
        2e-3,
    );
}

#[test]
fn native_2d_whittaker_tracks_reference_fixture() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();
    let [rows, cols] = fixture.shape;
    let input = MatrixView::row_major(&fixture.signal, rows, cols).unwrap();
    let whittaker = Whittaker2DParams {
        lambda: 1e4,
        max_iter: 50,
        tol: 1e-3,
        cg_max_iter: 500,
        cg_tol: 1e-6,
    };

    assert_baseline_close(
        "asls",
        &fixture.baselines,
        asls(input, Asls2DParams { whittaker, p: 0.01 })
            .unwrap()
            .baseline,
        2e-3,
    );
    assert_baseline_close(
        "iasls",
        &fixture.baselines,
        iasls(
            input,
            Iasls2DParams {
                whittaker,
                p: 0.01,
                lambda_1: 1e-4,
            },
        )
        .unwrap()
        .baseline,
        2e-2,
    );
    assert_baseline_close(
        "airpls",
        &fixture.baselines,
        airpls(input, AirPls2DParams { whittaker })
            .unwrap()
            .baseline,
        2e-2,
    );
    assert_baseline_close(
        "arpls",
        &fixture.baselines,
        arpls(input, ArPls2DParams { whittaker }).unwrap().baseline,
        1e-4,
    );
    assert_baseline_close(
        "drpls",
        &fixture.baselines,
        drpls(
            input,
            DrPls2DParams {
                whittaker,
                eta: 0.5,
            },
        )
        .unwrap()
        .baseline,
        3e-2,
    );
    assert_baseline_close(
        "iarpls",
        &fixture.baselines,
        iarpls(input, IarPls2DParams { whittaker })
            .unwrap()
            .baseline,
        5e-3,
    );
    assert_baseline_close(
        "aspls",
        &fixture.baselines,
        aspls(
            input,
            AsPls2DParams {
                whittaker: Whittaker2DParams {
                    max_iter: 100,
                    ..whittaker
                },
                asymmetric_coef: 0.5,
            },
        )
        .unwrap()
        .baseline,
        3e-2,
    );
    assert_baseline_close(
        "psalsa",
        &fixture.baselines,
        psalsa(
            input,
            Psalsa2DParams {
                whittaker,
                p: 0.5,
                k: None,
            },
        )
        .unwrap()
        .baseline,
        1e-3,
    );
    let low_lambda = Whittaker2DParams {
        lambda: 1e3,
        ..whittaker
    };
    assert_baseline_close(
        "brpls",
        &fixture.baselines,
        brpls(
            input,
            BrPls2DParams {
                whittaker: low_lambda,
                max_iter_2: 50,
                tol_2: 1e-3,
            },
        )
        .unwrap()
        .baseline,
        3e-3,
    );
    assert_baseline_close(
        "lsrpls",
        &fixture.baselines,
        lsrpls(
            input,
            LsrPls2DParams {
                whittaker: low_lambda,
            },
        )
        .unwrap()
        .baseline,
        1e-5,
    );
}

#[test]
fn native_2d_spline_tracks_reference_fixture() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();
    let [rows, cols] = fixture.shape;
    let input = MatrixView::row_major(&fixture.signal, rows, cols).unwrap();
    let spline = Spline2DParams {
        lambda: 1e3,
        max_iter: 50,
        tol: 1e-3,
        num_knots_rows: 8,
        num_knots_cols: 8,
    };

    assert_baseline_close(
        "pspline_asls",
        &fixture.baselines,
        pspline_asls(input, PsplineAsls2DParams { spline, p: 0.01 })
            .unwrap()
            .baseline,
        6e-2,
    );
    assert_baseline_close(
        "pspline_iasls",
        &fixture.baselines,
        pspline_iasls(
            input,
            PsplineIasls2DParams {
                spline,
                p: 0.01,
                lambda_1: 1e-4,
            },
        )
        .unwrap()
        .baseline,
        6e-2,
    );
    assert_baseline_close(
        "pspline_airpls",
        &fixture.baselines,
        pspline_airpls(input, PsplineAirPls2DParams { spline })
            .unwrap()
            .baseline,
        7e-2,
    );
    assert_baseline_close(
        "pspline_arpls",
        &fixture.baselines,
        pspline_arpls(input, PsplineArPls2DParams { spline })
            .unwrap()
            .baseline,
        3e-2,
    );
    assert_baseline_close(
        "pspline_iarpls",
        &fixture.baselines,
        pspline_iarpls(input, PsplineIarPls2DParams { spline })
            .unwrap()
            .baseline,
        6e-2,
    );
    assert_baseline_close(
        "pspline_psalsa",
        &fixture.baselines,
        pspline_psalsa(
            input,
            PsplinePsalsa2DParams {
                spline,
                p: 0.5,
                k: None,
            },
        )
        .unwrap()
        .baseline,
        3e-2,
    );
    assert_baseline_close(
        "pspline_brpls",
        &fixture.baselines,
        pspline_brpls(
            input,
            PsplineBrPls2DParams {
                spline: Spline2DParams {
                    max_iter: 20,
                    ..spline
                },
                max_iter_2: 10,
                tol_2: 1e-3,
            },
        )
        .unwrap()
        .baseline,
        3e-2,
    );
    assert_baseline_close(
        "pspline_lsrpls",
        &fixture.baselines,
        pspline_lsrpls(input, PsplineLsrPls2DParams { spline })
            .unwrap()
            .baseline,
        3e-2,
    );
    let short_spline = Spline2DParams {
        max_iter: 20,
        ..spline
    };
    assert_baseline_close(
        "irsqr",
        &fixture.baselines,
        irsqr(
            input,
            Irsqr2DParams {
                spline: short_spline,
                quantile: 0.05,
                epsilon: None,
            },
        )
        .unwrap()
        .baseline,
        2e-2,
    );
    assert_baseline_close(
        "mixture_model",
        &fixture.baselines,
        mixture_model(
            input,
            MixtureModel2DParams {
                spline: short_spline,
                p: 0.01,
            },
        )
        .unwrap()
        .baseline,
        5e-2,
    );
}

#[test]
fn native_2d_optimizer_tracks_reference_fixture() {
    let fixture: Fixture2D =
        serde_json::from_str(include_str!("fixtures/pybaselines_2d_reference.json")).unwrap();
    let [rows, cols] = fixture.shape;
    let input = MatrixView::row_major(&fixture.signal, rows, cols).unwrap();

    assert_baseline_close(
        "adaptive_minmax",
        &fixture.baselines,
        adaptive_minmax(
            input,
            AdaptiveMinmax2DParams {
                order: 2,
                max_iter: 20,
                tol: 1e-3,
            },
        )
        .unwrap()
        .baseline,
        3e-2,
    );
    assert_baseline_close(
        "individual_axes",
        &fixture.baselines,
        individual_axes(
            input,
            IndividualAxes2DParams {
                asls: AslsParams {
                    whittaker: WhittakerParams {
                        lambda: 1e4,
                        ..WhittakerParams::default()
                    },
                    p: 0.01,
                },
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );

    let collab = collab_surface(&fixture.signal, rows, cols);
    let collab_input = MatrixView::row_major(&collab, rows, cols).unwrap();
    let fits = collab_pls(&[input, collab_input], CollabPls2DParams::default()).unwrap();
    assert_eq!(fits.len(), 2);
    assert_baseline_close(
        "collab_pls_0",
        &fixture.baselines,
        fits[0].baseline.clone(),
        6e-2,
    );
    assert_baseline_close(
        "collab_pls_1",
        &fixture.baselines,
        fits[1].baseline.clone(),
        6e-2,
    );
}

fn assert_case_arrays(
    case_name: &str,
    shape: [usize; 2],
    signal: &[f64],
    baselines: &BTreeMap<String, Vec<f64>>,
) {
    let [rows, cols] = shape;
    let view = MatrixView::row_major(signal, rows, cols).unwrap_or_else(|error| {
        panic!("{case_name} signal should be a valid row-major matrix: {error}")
    });
    assert_eq!(view.len(), signal.len());

    for (method, baseline) in baselines {
        MatrixView::row_major(baseline, rows, cols).unwrap_or_else(|error| {
            panic!("{case_name}.{method} baseline should be a valid row-major matrix: {error}")
        });
    }
}

fn collab_surface(values: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut output = Vec::with_capacity(values.len());
    for row in 0..rows {
        let y = row as f64 / (rows - 1) as f64;
        for col in 0..cols {
            let x = col as f64 / (cols - 1) as f64;
            output.push(values[row * cols + col] + 0.03 * x + 0.02 * y);
        }
    }
    output
}

fn assert_baseline_close(
    name: &str,
    baselines: &BTreeMap<String, Vec<f64>>,
    actual: Vec<f64>,
    tolerance: f64,
) {
    let expected = baselines
        .get(name)
        .unwrap_or_else(|| panic!("missing 2D fixture for {name}"));
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max);
    assert!(
        max_error <= tolerance,
        "{name} max error {max_error} exceeded tolerance {tolerance}"
    );
}
