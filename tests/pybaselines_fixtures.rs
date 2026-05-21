use std::collections::BTreeMap;

use baselines::classification::rubberband;
use baselines::morphology::{
    MorphologyParams, imor, jbcd, mor, mormol, mpls, mwmv, rolling_ball, snip, tophat,
};
use baselines::polynomial::{
    GoldindecParams, ImodPolyParams, ModPolyParams, PenalizedPolyParams, PolyParams,
    QuantRegParams, goldindec, imodpoly, modpoly, penalized_poly, poly, quant_reg,
};
use baselines::smoothing::{SmoothingParams, noise_median};
use baselines::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, WhittakerParams, airpls, arpls, asls,
    aspls, brpls, derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    pybaselines_version: String,
    signal: Vec<f64>,
    baselines: BTreeMap<String, Vec<f64>>,
}

#[test]
fn core_algorithms_track_pybaselines_fixtures() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/pybaselines_1d_reference.json")).unwrap();
    assert_eq!(fixture.pybaselines_version, "1.2.1");

    assert_close(
        "poly",
        &fixture,
        poly(&fixture.signal, PolyParams { order: 2 })
            .unwrap()
            .baseline,
        1e-10,
    );
    assert_close(
        "modpoly",
        &fixture,
        modpoly(&fixture.signal, ModPolyParams::default())
            .unwrap()
            .baseline,
        6e-2,
    );
    assert_close(
        "imodpoly",
        &fixture,
        imodpoly(&fixture.signal, ImodPolyParams::default())
            .unwrap()
            .baseline,
        4e-2,
    );
    assert_close(
        "penalized_poly",
        &fixture,
        penalized_poly(&fixture.signal, PenalizedPolyParams::default())
            .unwrap()
            .baseline,
        1e-12,
    );
    assert_close(
        "quant_reg",
        &fixture,
        quant_reg(&fixture.signal, QuantRegParams::default())
            .unwrap()
            .baseline,
        1e-3,
    );
    assert_close(
        "goldindec",
        &fixture,
        goldindec(&fixture.signal, GoldindecParams::default())
            .unwrap()
            .baseline,
        1e-12,
    );

    let whittaker = WhittakerParams {
        lambda: 1e5,
        max_iter: 50,
        tol: 1e-3,
    };
    assert_close(
        "asls",
        &fixture,
        asls(&fixture.signal, AslsParams { whittaker, p: 0.01 })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "iasls",
        &fixture,
        iasls(
            &fixture.signal,
            IaslsParams {
                whittaker,
                p: 0.01,
                lambda_1: 1e-4,
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    assert_close(
        "airpls",
        &fixture,
        airpls(&fixture.signal, AirPlsParams { whittaker })
            .unwrap()
            .baseline,
        2.5e-2,
    );
    assert_close(
        "arpls",
        &fixture,
        arpls(&fixture.signal, ArPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-3,
    );
    assert_close(
        "drpls",
        &fixture,
        drpls(
            &fixture.signal,
            DrPlsParams {
                whittaker,
                eta: 0.5,
            },
        )
        .unwrap()
        .baseline,
        1e-2,
    );
    assert_close(
        "iarpls",
        &fixture,
        iarpls(&fixture.signal, IarPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "aspls",
        &fixture,
        aspls(
            &fixture.signal,
            AsPlsParams {
                whittaker: WhittakerParams {
                    lambda: 1e5,
                    max_iter: 100,
                    tol: 1e-3,
                },
                asymmetric_coef: 0.5,
            },
        )
        .unwrap()
        .baseline,
        1e-2,
    );
    assert_close(
        "psalsa",
        &fixture,
        psalsa(
            &fixture.signal,
            PsalsaParams {
                whittaker,
                p: 0.5,
                k: None,
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    assert_close(
        "derpsalsa",
        &fixture,
        derpsalsa(
            &fixture.signal,
            DerPsalsaParams {
                whittaker,
                p: 0.01,
                k: None,
                smooth_half_window: None,
                num_smooths: 16,
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    assert_close(
        "brpls",
        &fixture,
        brpls(
            &fixture.signal,
            BrPlsParams {
                whittaker,
                max_iter_2: 50,
                tol_2: 1e-3,
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    assert_close(
        "lsrpls",
        &fixture,
        lsrpls(&fixture.signal, LsrPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-8,
    );

    let morphology = MorphologyParams { window_size: 17 };
    assert_close(
        "rolling_ball",
        &fixture,
        rolling_ball(&fixture.signal, morphology).unwrap().baseline,
        1e-12,
    );
    assert_close(
        "mwmv",
        &fixture,
        mwmv(&fixture.signal, morphology).unwrap().baseline,
        1e-12,
    );
    assert_close(
        "tophat",
        &fixture,
        tophat(&fixture.signal, morphology).unwrap().baseline,
        1e-12,
    );
    assert_close(
        "mor",
        &fixture,
        mor(&fixture.signal, morphology).unwrap().baseline,
        1e-12,
    );
    assert_close(
        "mpls",
        &fixture,
        mpls(&fixture.signal, morphology).unwrap().baseline,
        1e-8,
    );
    assert_close(
        "imor",
        &fixture,
        imor(&fixture.signal, morphology).unwrap().baseline,
        1e-12,
    );
    assert_close(
        "mormol",
        &fixture,
        mormol(&fixture.signal, morphology).unwrap().baseline,
        3e-4,
    );
    assert_close(
        "jbcd",
        &fixture,
        jbcd(&fixture.signal, morphology).unwrap().baseline,
        1e-8,
    );
    assert_close(
        "snip",
        &fixture,
        snip(
            &fixture.signal,
            baselines::SnipParams { max_half_window: 8 },
        )
        .unwrap()
        .baseline,
        7e-2,
    );
    assert_close(
        "noise_median",
        &fixture,
        noise_median(
            &fixture.signal,
            SmoothingParams {
                window_size: 17,
                max_iter: 1,
            },
        )
        .unwrap()
        .baseline,
        1.3e-1,
    );
    assert_close(
        "rubberband",
        &fixture,
        rubberband(&fixture.signal).unwrap().baseline,
        1e-12,
    );
}

fn assert_close(name: &str, fixture: &Fixture, actual: Vec<f64>, tolerance: f64) {
    let expected = fixture
        .baselines
        .get(name)
        .unwrap_or_else(|| panic!("missing fixture for {name}"));
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
