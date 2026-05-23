use std::collections::BTreeMap;

use baselines::MatrixView;
use baselines::two_d::morphology::{
    Morphology2DParams, imor, mor, noise_median, rolling_ball, tophat,
};
use baselines::two_d::polynomial::{
    ImodPoly2DParams, ModPoly2DParams, PenalizedPoly2DParams, Poly2DParams, QuantReg2DParams,
    imodpoly, modpoly, penalized_poly, poly, quant_reg,
};
use baselines::two_d::whittaker::{
    AirPls2DParams, ArPls2DParams, AsPls2DParams, Asls2DParams, BrPls2DParams, DrPls2DParams,
    IarPls2DParams, Iasls2DParams, LsrPls2DParams, Psalsa2DParams, Whittaker2DParams, airpls,
    arpls, asls, aspls, brpls, drpls, iarpls, iasls, lsrpls, psalsa,
};
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
        3e-1,
    );
    assert_baseline_close(
        "tophat",
        &fixture.baselines,
        tophat(input, params).unwrap().baseline,
        3e-1,
    );
    assert_baseline_close(
        "mor",
        &fixture.baselines,
        mor(input, params).unwrap().baseline,
        3e-1,
    );
    assert_baseline_close(
        "imor",
        &fixture.baselines,
        imor(input, params).unwrap().baseline,
        3e-1,
    );
    assert_baseline_close(
        "noise_median",
        &fixture.baselines,
        noise_median(input, params).unwrap().baseline,
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
    );
    assert_baseline_close(
        "airpls",
        &fixture.baselines,
        airpls(input, AirPls2DParams { whittaker })
            .unwrap()
            .baseline,
        3e-1,
    );
    assert_baseline_close(
        "arpls",
        &fixture.baselines,
        arpls(input, ArPls2DParams { whittaker }).unwrap().baseline,
        3e-1,
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
        3e-1,
    );
    assert_baseline_close(
        "iarpls",
        &fixture.baselines,
        iarpls(input, IarPls2DParams { whittaker })
            .unwrap()
            .baseline,
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
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
        3e-1,
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
