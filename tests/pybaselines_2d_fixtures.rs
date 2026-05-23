use std::collections::BTreeMap;

use baselines::MatrixView;
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
