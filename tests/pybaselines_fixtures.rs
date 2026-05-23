use std::collections::BTreeMap;

use baselines::classification::{
    CwtBrParams, DietrichParams, FabcParams, FastChromParams, GolotvinParams,
    StdDistributionParams, cwt_br, dietrich, fabc, fastchrom, golotvin, rubberband,
    std_distribution,
};
use baselines::misc::{BeadsParams, beads, interp_pts};
use baselines::morphology::{
    MorphologyParams, amormol, imor, jbcd, mor, mormol, mpls, mpspline, mwmv, rolling_ball, snip,
    tophat,
};
use baselines::optimizers::{
    AdaptiveMinmaxParams, CustomBcParams, LambdaSearchParams, adaptive_minmax, collab_pls,
    custom_bc, optimize_extended_range,
};
use baselines::polynomial::{
    GoldindecParams, ImodPolyParams, LoessParams, ModPolyParams, PenalizedPolyParams, PolyParams,
    QuantRegParams, goldindec, imodpoly, loess, modpoly, penalized_poly, poly, quant_reg,
};
use baselines::smoothing::{SmoothingParams, ipsa, noise_median, peak_filling, ria, swima};
use baselines::spline::{
    CornerCuttingParams, IrsqrParams, MixtureModelParams, corner_cutting, irsqr, mixture_model,
    pspline_airpls, pspline_arpls, pspline_asls, pspline_aspls, pspline_brpls, pspline_derpsalsa,
    pspline_drpls, pspline_iarpls, pspline_iasls, pspline_lsrpls, pspline_mpls, pspline_psalsa,
};
use baselines::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, WhittakerParams, airpls, arpls, asls,
    aspls, brpls, derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Fixture {
    pybaselines_version: String,
    pybaselines_methods: Vec<String>,
    signal: Vec<f64>,
    baselines: BTreeMap<String, Vec<f64>>,
    cases: BTreeMap<String, FixtureCase>,
}

#[derive(Debug, Deserialize)]
struct FixtureCase {
    signal: Vec<f64>,
    baselines: BTreeMap<String, Vec<f64>>,
}

const EXPECTED_PYBASELINES_METHODS: &[&str] = &[
    "adaptive_minmax",
    "airpls",
    "amormol",
    "arpls",
    "asls",
    "aspls",
    "beads",
    "brpls",
    "collab_pls",
    "corner_cutting",
    "custom_bc",
    "cwt_br",
    "derpsalsa",
    "dietrich",
    "drpls",
    "fabc",
    "fastchrom",
    "goldindec",
    "golotvin",
    "iarpls",
    "iasls",
    "imodpoly",
    "imor",
    "interp_pts",
    "ipsa",
    "irsqr",
    "jbcd",
    "loess",
    "lsrpls",
    "mixture_model",
    "modpoly",
    "mor",
    "mormol",
    "mpls",
    "mpspline",
    "mwmv",
    "noise_median",
    "optimize_extended_range",
    "peak_filling",
    "penalized_poly",
    "poly",
    "psalsa",
    "pspline_airpls",
    "pspline_arpls",
    "pspline_asls",
    "pspline_aspls",
    "pspline_brpls",
    "pspline_derpsalsa",
    "pspline_drpls",
    "pspline_iarpls",
    "pspline_iasls",
    "pspline_lsrpls",
    "pspline_mpls",
    "pspline_psalsa",
    "quant_reg",
    "ria",
    "rolling_ball",
    "rubberband",
    "snip",
    "std_distribution",
    "swima",
    "tophat",
];

#[test]
fn pybaselines_method_list_has_not_drifted() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/pybaselines_1d_reference.json")).unwrap();
    assert_eq!(fixture.pybaselines_version, "1.2.1");
    let expected: Vec<String> = EXPECTED_PYBASELINES_METHODS
        .iter()
        .map(|method| (*method).to_owned())
        .collect();
    assert_eq!(
        fixture.pybaselines_methods, expected,
        "pinned pybaselines public method list changed"
    );
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
        "loess",
        &fixture,
        loess(&fixture.signal, LoessParams { window_size: 26 })
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
    assert_close(
        "pspline_asls",
        &fixture,
        pspline_asls(&fixture.signal, AslsParams { whittaker, p: 0.01 })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "pspline_iasls",
        &fixture,
        pspline_iasls(
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
        "pspline_airpls",
        &fixture,
        pspline_airpls(&fixture.signal, AirPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "pspline_arpls",
        &fixture,
        pspline_arpls(&fixture.signal, ArPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "pspline_drpls",
        &fixture,
        pspline_drpls(
            &fixture.signal,
            DrPlsParams {
                whittaker,
                eta: 0.5,
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    assert_close(
        "pspline_iarpls",
        &fixture,
        pspline_iarpls(&fixture.signal, IarPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "pspline_aspls",
        &fixture,
        pspline_aspls(
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
        1e-4,
    );
    assert_close(
        "pspline_psalsa",
        &fixture,
        pspline_psalsa(
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
        "pspline_derpsalsa",
        &fixture,
        pspline_derpsalsa(
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
        "pspline_lsrpls",
        &fixture,
        pspline_lsrpls(&fixture.signal, LsrPlsParams { whittaker })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "pspline_brpls",
        &fixture,
        pspline_brpls(
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
        "pspline_mpls",
        &fixture,
        pspline_mpls(&fixture.signal, MorphologyParams { window_size: 17 })
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "irsqr",
        &fixture,
        irsqr(&fixture.signal, IrsqrParams::default())
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "mixture_model",
        &fixture,
        mixture_model(&fixture.signal, MixtureModelParams::default())
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
        "amormol",
        &fixture,
        amormol(&fixture.signal, morphology).unwrap().baseline,
        2e-1,
    );
    assert_close(
        "jbcd",
        &fixture,
        jbcd(&fixture.signal, morphology).unwrap().baseline,
        1e-8,
    );
    assert_close(
        "mpspline",
        &fixture,
        mpspline(&fixture.signal, morphology).unwrap().baseline,
        1e-11,
    );
    assert_close(
        "snip",
        &fixture,
        snip(
            &fixture.signal,
            baselines::morphology::SnipParams { max_half_window: 8 },
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
        1e-3,
    );
    let smooth = SmoothingParams {
        window_size: 17,
        max_iter: 20,
    };
    assert_close(
        "swima",
        &fixture,
        swima(&fixture.signal, smooth).unwrap().baseline,
        3.5e-1,
    );
    assert_close(
        "ipsa",
        &fixture,
        ipsa(&fixture.signal, smooth).unwrap().baseline,
        1e-3,
    );
    assert_close(
        "ria",
        &fixture,
        ria(&fixture.signal, smooth).unwrap().baseline,
        4e-1,
    );
    assert_close(
        "peak_filling",
        &fixture,
        peak_filling(&fixture.signal, smooth).unwrap().baseline,
        1e-3,
    );
    assert_close(
        "corner_cutting",
        &fixture,
        corner_cutting(&fixture.signal, CornerCuttingParams::default())
            .unwrap()
            .baseline,
        1e-12,
    );
    assert_close(
        "dietrich",
        &fixture,
        dietrich(&fixture.signal, DietrichParams::default())
            .unwrap()
            .baseline,
        1e-10,
    );
    assert_close(
        "golotvin",
        &fixture,
        golotvin(&fixture.signal, GolotvinParams::default())
            .unwrap()
            .baseline,
        1e-12,
    );
    assert_close(
        "std_distribution",
        &fixture,
        std_distribution(&fixture.signal, StdDistributionParams::default())
            .unwrap()
            .baseline,
        1e-12,
    );
    assert_close(
        "fastchrom",
        &fixture,
        fastchrom(&fixture.signal, FastChromParams::default())
            .unwrap()
            .baseline,
        1e-12,
    );
    assert_close(
        "cwt_br",
        &fixture,
        cwt_br(
            &fixture.signal,
            CwtBrParams {
                poly_order: 2,
                scales: Some(vec![8]),
                num_std: 1.0,
                min_length: 2,
                max_iter: 50,
                tol: 1e-3,
                symmetric: false,
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    assert_close(
        "fabc",
        &fixture,
        fabc(&fixture.signal, FabcParams::default())
            .unwrap()
            .baseline,
        1e-10,
    );
    assert_close(
        "adaptive_minmax",
        &fixture,
        adaptive_minmax(&fixture.signal, AdaptiveMinmaxParams::default())
            .unwrap()
            .baseline,
        1e-10,
    );
    assert_close(
        "optimize_extended_range",
        &fixture,
        optimize_extended_range(
            &fixture.signal,
            LambdaSearchParams {
                start_exp: 2.0,
                stop_exp: 4.0,
                steps: 2,
            },
        )
        .unwrap()
        .baseline,
        1e-9,
    );
    assert_close(
        "custom_bc",
        &fixture,
        custom_bc(
            &fixture.signal,
            CustomBcParams {
                sampling: 4,
                asls: AslsParams { whittaker, p: 0.01 },
                ..CustomBcParams::default()
            },
        )
        .unwrap()
        .baseline,
        1e-8,
    );
    let collab = collab_pls(
        &[fixture.signal.clone(), collab_signal(&fixture.signal)],
        AslsParams { whittaker, p: 0.01 },
    )
    .unwrap();
    assert_close("collab_pls_0", &fixture, collab[0].baseline.clone(), 1e-8);
    assert_close("collab_pls_1", &fixture, collab[1].baseline.clone(), 1e-8);
    assert_close(
        "rubberband",
        &fixture,
        rubberband(&fixture.signal).unwrap().baseline,
        1e-12,
    );
    assert_close(
        "beads",
        &fixture,
        beads(&fixture.signal, BeadsParams::default())
            .unwrap()
            .baseline,
        1e-8,
    );
    assert_close(
        "interp_pts",
        &fixture,
        interp_pts(
            &fixture.signal,
            &[
                (0, fixture.signal[0]),
                (
                    fixture.signal.len() / 2,
                    fixture.signal[fixture.signal.len() / 2],
                ),
                (
                    fixture.signal.len() - 1,
                    fixture.signal[fixture.signal.len() - 1],
                ),
            ],
        )
        .unwrap()
        .baseline,
        1e-12,
    );
}

#[test]
fn targeted_signal_cases_track_representative_methods() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("fixtures/pybaselines_1d_reference.json")).unwrap();
    let whittaker = WhittakerParams {
        lambda: 1e5,
        max_iter: 50,
        tol: 1e-3,
    };
    let morphology = MorphologyParams { window_size: 17 };

    for (case_name, case) in &fixture.cases {
        if case_name == "reference" {
            continue;
        }
        assert_case_close(
            case_name,
            "asls",
            &case.baselines,
            asls(&case.signal, AslsParams { whittaker, p: 0.01 })
                .unwrap()
                .baseline,
            1e-8,
        );
        assert_case_close(
            case_name,
            "arpls",
            &case.baselines,
            arpls(&case.signal, ArPlsParams { whittaker })
                .unwrap()
                .baseline,
            2e-3,
        );
        assert_case_close(
            case_name,
            "rolling_ball",
            &case.baselines,
            rolling_ball(&case.signal, morphology).unwrap().baseline,
            1e-12,
        );
        assert_case_close(
            case_name,
            "pspline_asls",
            &case.baselines,
            pspline_asls(&case.signal, AslsParams { whittaker, p: 0.01 })
                .unwrap()
                .baseline,
            5e-4,
        );
        if case.baselines.contains_key("cwt_br") {
            assert_case_close(
                case_name,
                "cwt_br",
                &case.baselines,
                cwt_br(
                    &case.signal,
                    CwtBrParams {
                        poly_order: 2,
                        scales: Some(vec![8]),
                        num_std: 1.0,
                        min_length: 2,
                        max_iter: 50,
                        tol: 1e-3,
                        symmetric: false,
                    },
                )
                .unwrap()
                .baseline,
                3e-2,
            );
        }
        assert_case_close(
            case_name,
            "custom_bc",
            &case.baselines,
            custom_bc(
                &case.signal,
                CustomBcParams {
                    sampling: 4,
                    asls: AslsParams { whittaker, p: 0.01 },
                    ..CustomBcParams::default()
                },
            )
            .unwrap()
            .baseline,
            1e-8,
        );
        assert_case_close(
            case_name,
            "rubberband",
            &case.baselines,
            rubberband(&case.signal).unwrap().baseline,
            1e-12,
        );
        assert_case_close(
            case_name,
            "beads",
            &case.baselines,
            beads(&case.signal, BeadsParams::default())
                .unwrap()
                .baseline,
            1e-8,
        );
    }
}

fn collab_signal(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let x = index as f64 / (values.len() - 1) as f64;
            value + 0.03 * x + 0.15 * (-((x - 0.55).powi(2)) / 0.002).exp()
        })
        .collect()
}

fn assert_close(name: &str, fixture: &Fixture, actual: Vec<f64>, tolerance: f64) {
    assert_baseline_close(name, &fixture.baselines, actual, tolerance);
}

fn assert_case_close(
    case_name: &str,
    method_name: &str,
    baselines: &BTreeMap<String, Vec<f64>>,
    actual: Vec<f64>,
    tolerance: f64,
) {
    let label = format!("{case_name}.{method_name}");
    assert_baseline_close(&label, baselines, actual, tolerance);
}

fn assert_baseline_close(
    name: &str,
    baselines: &BTreeMap<String, Vec<f64>>,
    actual: Vec<f64>,
    tolerance: f64,
) {
    let expected = baselines
        .get(name.rsplit_once('.').map_or(name, |(_, method)| method))
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
