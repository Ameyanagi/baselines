use baselines::MatrixView;
use baselines::backend::cpu::snip_batch_into;
use baselines::classification as class;
use baselines::misc;
use baselines::morphology as morph;
use baselines::optimizers as opt;
use baselines::polynomial as poly;
use baselines::smoothing;
use baselines::spline;
use baselines::two_d::morphology as morph2d;
use baselines::two_d::optimizers as opt2d;
use baselines::two_d::polynomial as poly2d;
use baselines::two_d::spline as spline2d;
use baselines::two_d::whittaker as whittaker2d;
use baselines::whittaker as whit;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| {
            let x = index as f64 / (n - 1) as f64;
            let baseline = 0.8 + 0.2 * x + 0.05 * (2.0 * std::f64::consts::PI * x).sin();
            let peak_a = (-((x - 0.35).powi(2)) / 0.0015).exp();
            let peak_b = 0.5 * (-((x - 0.72).powi(2)) / 0.003).exp();
            baseline + peak_a + peak_b
        })
        .collect()
}

fn surface(rows: usize, cols: usize) -> Vec<f64> {
    let row_scale = (rows - 1).max(1) as f64;
    let col_scale = (cols - 1).max(1) as f64;
    let mut data = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        let y = row as f64 / row_scale;
        for col in 0..cols {
            let x = col as f64 / col_scale;
            let baseline = 1.0 + 0.2 * x + 0.15 * y + 0.05 * (2.0 * std::f64::consts::PI * x).sin();
            let peak = (-(((x - 0.35).powi(2) + (y - 0.55).powi(2)) / 0.01)).exp();
            data.push(baseline + peak);
        }
    }
    data
}

fn matrix(data: &[f64], rows: usize, cols: usize) -> MatrixView<'_> {
    MatrixView::row_major(data, rows, cols).unwrap()
}

fn bench_whittaker_1d(c: &mut Criterion) {
    let y = signal(256);
    let mut group = c.benchmark_group("whittaker_1d");
    group.bench_function("asls_256", |bench| {
        bench.iter(|| whit::asls(black_box(y.as_slice()), whit::AslsParams::default()).unwrap())
    });
    group.bench_function("airpls_256", |bench| {
        bench.iter(|| whit::airpls(black_box(y.as_slice()), whit::AirPlsParams::default()).unwrap())
    });
    group.bench_function("arpls_256", |bench| {
        bench.iter(|| whit::arpls(black_box(y.as_slice()), whit::ArPlsParams::default()).unwrap())
    });
    group.bench_function("iasls_256", |bench| {
        bench.iter(|| whit::iasls(black_box(y.as_slice()), whit::IaslsParams::default()).unwrap())
    });
    group.bench_function("drpls_256", |bench| {
        bench.iter(|| whit::drpls(black_box(y.as_slice()), whit::DrPlsParams::default()).unwrap())
    });
    group.bench_function("iarpls_256", |bench| {
        bench.iter(|| whit::iarpls(black_box(y.as_slice()), whit::IarPlsParams::default()).unwrap())
    });
    group.bench_function("aspls_256", |bench| {
        bench.iter(|| whit::aspls(black_box(y.as_slice()), whit::AsPlsParams::default()).unwrap())
    });
    group.bench_function("psalsa_256", |bench| {
        bench.iter(|| whit::psalsa(black_box(y.as_slice()), whit::PsalsaParams::default()).unwrap())
    });
    group.bench_function("derpsalsa_256", |bench| {
        bench.iter(|| {
            whit::derpsalsa(black_box(y.as_slice()), whit::DerPsalsaParams::default()).unwrap()
        })
    });
    group.bench_function("brpls_256", |bench| {
        bench.iter(|| whit::brpls(black_box(y.as_slice()), whit::BrPlsParams::default()).unwrap())
    });
    group.bench_function("lsrpls_256", |bench| {
        bench.iter(|| whit::lsrpls(black_box(y.as_slice()), whit::LsrPlsParams::default()).unwrap())
    });
    group.finish();
}

fn bench_polynomial_1d(c: &mut Criterion) {
    let y = signal(256);
    let mut group = c.benchmark_group("polynomial_1d");
    group.bench_function("poly_256", |bench| {
        bench.iter(|| poly::poly(black_box(y.as_slice()), poly::PolyParams::default()).unwrap())
    });
    group.bench_function("modpoly_256", |bench| {
        bench.iter(|| {
            poly::modpoly(black_box(y.as_slice()), poly::ModPolyParams::default()).unwrap()
        })
    });
    group.bench_function("imodpoly_256", |bench| {
        bench.iter(|| {
            poly::imodpoly(black_box(y.as_slice()), poly::ImodPolyParams::default()).unwrap()
        })
    });
    group.bench_function("penalized_poly_256", |bench| {
        bench.iter(|| {
            poly::penalized_poly(
                black_box(y.as_slice()),
                poly::PenalizedPolyParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("loess_256", |bench| {
        bench.iter(|| poly::loess(black_box(y.as_slice()), poly::LoessParams::default()).unwrap())
    });
    group.bench_function("quant_reg_256", |bench| {
        bench.iter(|| {
            poly::quant_reg(black_box(y.as_slice()), poly::QuantRegParams::default()).unwrap()
        })
    });
    group.bench_function("goldindec_256", |bench| {
        bench.iter(|| {
            poly::goldindec(black_box(y.as_slice()), poly::GoldindecParams::default()).unwrap()
        })
    });
    group.finish();
}

fn bench_morphology_1d(c: &mut Criterion) {
    let y = signal(256);
    let mut group = c.benchmark_group("morphology_1d");
    group.bench_function("rolling_ball_256", |bench| {
        bench.iter(|| {
            morph::rolling_ball(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("tophat_256", |bench| {
        bench.iter(|| {
            morph::tophat(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("mwmv_256", |bench| {
        bench.iter(|| {
            morph::mwmv(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("mor_256", |bench| {
        bench.iter(|| {
            morph::mor(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("mpls_256", |bench| {
        bench.iter(|| {
            morph::mpls(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("imor_256", |bench| {
        bench.iter(|| {
            morph::imor(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("mormol_256", |bench| {
        bench.iter(|| {
            morph::mormol(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("amormol_256", |bench| {
        bench.iter(|| {
            morph::amormol(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("mpspline_256", |bench| {
        bench.iter(|| {
            morph::mpspline(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("jbcd_256", |bench| {
        bench.iter(|| {
            morph::jbcd(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("snip_256", |bench| {
        bench.iter(|| morph::snip(black_box(y.as_slice()), morph::SnipParams::default()).unwrap())
    });
    group.finish();
}

fn bench_smoothing_1d(c: &mut Criterion) {
    let y = signal(256);
    let mut group = c.benchmark_group("smoothing_1d");
    group.bench_function("noise_median_256", |bench| {
        bench.iter(|| {
            smoothing::noise_median(
                black_box(y.as_slice()),
                smoothing::SmoothingParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("snip_256", |bench| {
        bench.iter(|| {
            smoothing::snip(black_box(y.as_slice()), morph::SnipParams::default()).unwrap()
        })
    });
    group.bench_function("swima_256", |bench| {
        bench.iter(|| {
            smoothing::swima(
                black_box(y.as_slice()),
                smoothing::SmoothingParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("ipsa_256", |bench| {
        bench.iter(|| {
            smoothing::ipsa(
                black_box(y.as_slice()),
                smoothing::SmoothingParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("ria_256", |bench| {
        bench.iter(|| {
            smoothing::ria(
                black_box(y.as_slice()),
                smoothing::SmoothingParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("peak_filling_256", |bench| {
        bench.iter(|| {
            smoothing::peak_filling(
                black_box(y.as_slice()),
                smoothing::SmoothingParams::default(),
            )
            .unwrap()
        })
    });
    group.finish();
}

fn bench_spline_1d(c: &mut Criterion) {
    let y = signal(256);
    let mut group = c.benchmark_group("spline_1d");
    group.bench_function("irsqr_256", |bench| {
        bench.iter(|| {
            spline::irsqr(black_box(y.as_slice()), spline::IrsqrParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_asls_256", |bench| {
        bench.iter(|| {
            spline::pspline_asls(black_box(y.as_slice()), whit::AslsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_iasls_256", |bench| {
        bench.iter(|| {
            spline::pspline_iasls(black_box(y.as_slice()), whit::IaslsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_airpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_airpls(black_box(y.as_slice()), whit::AirPlsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_arpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_arpls(black_box(y.as_slice()), whit::ArPlsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_drpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_drpls(black_box(y.as_slice()), whit::DrPlsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_iarpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_iarpls(black_box(y.as_slice()), whit::IarPlsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_aspls_256", |bench| {
        bench.iter(|| {
            spline::pspline_aspls(black_box(y.as_slice()), whit::AsPlsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_psalsa_256", |bench| {
        bench.iter(|| {
            spline::pspline_psalsa(black_box(y.as_slice()), whit::PsalsaParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_derpsalsa_256", |bench| {
        bench.iter(|| {
            spline::pspline_derpsalsa(black_box(y.as_slice()), whit::DerPsalsaParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_mpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_mpls(
                black_box(y.as_slice()),
                morph::MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    group.bench_function("pspline_brpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_brpls(black_box(y.as_slice()), whit::BrPlsParams::default()).unwrap()
        })
    });
    group.bench_function("pspline_lsrpls_256", |bench| {
        bench.iter(|| {
            spline::pspline_lsrpls(black_box(y.as_slice()), whit::LsrPlsParams::default()).unwrap()
        })
    });
    group.bench_function("mixture_model_256", |bench| {
        bench.iter(|| {
            spline::mixture_model(
                black_box(y.as_slice()),
                spline::MixtureModelParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("corner_cutting_256", |bench| {
        bench.iter(|| {
            spline::corner_cutting(
                black_box(y.as_slice()),
                spline::CornerCuttingParams::default(),
            )
            .unwrap()
        })
    });
    group.finish();
}

fn bench_classification_1d(c: &mut Criterion) {
    let y = signal(256);
    let cwt_params = class::CwtBrParams {
        poly_order: 2,
        scales: Some(vec![8]),
        ..class::CwtBrParams::default()
    };
    let mut group = c.benchmark_group("classification_1d");
    group.bench_function("dietrich_256", |bench| {
        bench.iter(|| {
            class::dietrich(black_box(y.as_slice()), class::DietrichParams::default()).unwrap()
        })
    });
    group.bench_function("golotvin_256", |bench| {
        bench.iter(|| {
            class::golotvin(black_box(y.as_slice()), class::GolotvinParams::default()).unwrap()
        })
    });
    group.bench_function("std_distribution_256", |bench| {
        bench.iter(|| {
            class::std_distribution(
                black_box(y.as_slice()),
                class::StdDistributionParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("std_distribution_with_mask_256", |bench| {
        bench.iter(|| {
            class::std_distribution_with_mask(
                black_box(y.as_slice()),
                class::StdDistributionParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("fastchrom_256", |bench| {
        bench.iter(|| {
            class::fastchrom(black_box(y.as_slice()), class::FastChromParams::default()).unwrap()
        })
    });
    group.bench_function("fastchrom_with_mask_256", |bench| {
        bench.iter(|| {
            class::fastchrom_with_mask(black_box(y.as_slice()), class::FastChromParams::default())
                .unwrap()
        })
    });
    group.bench_function("cwt_br_256", |bench| {
        bench.iter(|| class::cwt_br(black_box(y.as_slice()), cwt_params.clone()).unwrap())
    });
    group.bench_function("fabc_256", |bench| {
        bench.iter(|| class::fabc(black_box(y.as_slice()), class::FabcParams::default()).unwrap())
    });
    group.bench_function("rubberband_256", |bench| {
        bench.iter(|| class::rubberband(black_box(y.as_slice())).unwrap())
    });
    group.finish();
}

fn bench_optimizers_and_misc_1d(c: &mut Criterion) {
    let y = signal(256);
    let points = vec![
        (0, y[0]),
        (y.len() / 2, y[y.len() / 2]),
        (y.len() - 1, y[y.len() - 1]),
    ];
    let mut spectra = Vec::new();
    for index in 0..3 {
        let shift = index as f64 * 0.001;
        spectra.push(y.iter().map(|value| value + shift).collect::<Vec<_>>());
    }
    let lambda_search = opt::LambdaSearchParams {
        start_exp: 2.0,
        stop_exp: 5.0,
        steps: 4,
    };
    let custom_bc = opt::CustomBcParams::default();
    let mut group = c.benchmark_group("optimizers_misc_1d");
    group.bench_function("optimize_extended_range_256", |bench| {
        bench.iter(|| opt::optimize_extended_range(black_box(y.as_slice()), lambda_search).unwrap())
    });
    group.bench_function("custom_bc_256", |bench| {
        bench.iter(|| opt::custom_bc(black_box(y.as_slice()), custom_bc.clone()).unwrap())
    });
    group.bench_function("adaptive_minmax_256", |bench| {
        bench.iter(|| {
            opt::adaptive_minmax(
                black_box(y.as_slice()),
                opt::AdaptiveMinmaxParams::default(),
            )
            .unwrap()
        })
    });
    group.bench_function("collab_pls_3x256", |bench| {
        bench.iter(|| {
            opt::collab_pls(black_box(spectra.as_slice()), whit::AslsParams::default()).unwrap()
        })
    });
    group.bench_function("beads_256", |bench| {
        bench.iter(|| misc::beads(black_box(y.as_slice()), misc::BeadsParams::default()).unwrap())
    });
    group.bench_function("interp_pts_256", |bench| {
        bench.iter(|| {
            misc::interp_pts(black_box(y.as_slice()), black_box(points.as_slice())).unwrap()
        })
    });
    group.finish();
}

fn bench_whittaker_2d(c: &mut Criterion) {
    let rows = 16;
    let cols = 16;
    let data = surface(rows, cols);
    let input = matrix(&data, rows, cols);
    let mut group = c.benchmark_group("whittaker_2d");
    group.bench_function("asls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::asls(black_box(input), whittaker2d::Asls2DParams::default()).unwrap()
        })
    });
    group.bench_function("iasls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::iasls(black_box(input), whittaker2d::Iasls2DParams::default()).unwrap()
        })
    });
    group.bench_function("airpls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::airpls(black_box(input), whittaker2d::AirPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("arpls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::arpls(black_box(input), whittaker2d::ArPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("drpls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::drpls(black_box(input), whittaker2d::DrPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("iarpls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::iarpls(black_box(input), whittaker2d::IarPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("aspls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::aspls(black_box(input), whittaker2d::AsPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("psalsa_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::psalsa(black_box(input), whittaker2d::Psalsa2DParams::default()).unwrap()
        })
    });
    group.bench_function("brpls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::brpls(black_box(input), whittaker2d::BrPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("lsrpls_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::lsrpls(black_box(input), whittaker2d::LsrPls2DParams::default()).unwrap()
        })
    });
    group.bench_function("arpls_eigen_16x16", |bench| {
        bench.iter(|| {
            whittaker2d::arpls_eigen(
                black_box(input),
                whittaker2d::ArPls2DEigenParams {
                    whittaker: whittaker2d::Whittaker2DEigenParams {
                        num_eigens: (6, 6),
                        ..whittaker2d::Whittaker2DEigenParams::default()
                    },
                },
            )
            .unwrap()
        })
    });
    group.finish();
}

fn bench_polynomial_2d(c: &mut Criterion) {
    let rows = 16;
    let cols = 16;
    let data = surface(rows, cols);
    let input = matrix(&data, rows, cols);
    let mut group = c.benchmark_group("polynomial_2d");
    group.bench_function("poly_16x16", |bench| {
        bench.iter(|| poly2d::poly(black_box(input), poly2d::Poly2DParams::default()).unwrap())
    });
    group.bench_function("modpoly_16x16", |bench| {
        bench
            .iter(|| poly2d::modpoly(black_box(input), poly2d::ModPoly2DParams::default()).unwrap())
    });
    group.bench_function("imodpoly_16x16", |bench| {
        bench.iter(|| {
            poly2d::imodpoly(black_box(input), poly2d::ImodPoly2DParams::default()).unwrap()
        })
    });
    group.bench_function("penalized_poly_16x16", |bench| {
        bench.iter(|| {
            poly2d::penalized_poly(black_box(input), poly2d::PenalizedPoly2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("quant_reg_16x16", |bench| {
        bench.iter(|| {
            poly2d::quant_reg(black_box(input), poly2d::QuantReg2DParams::default()).unwrap()
        })
    });
    group.finish();
}

fn bench_morphology_2d(c: &mut Criterion) {
    let rows = 16;
    let cols = 16;
    let data = surface(rows, cols);
    let input = matrix(&data, rows, cols);
    let params = morph2d::Morphology2DParams {
        window_rows: 5,
        window_cols: 5,
    };
    let mut group = c.benchmark_group("morphology_2d");
    group.bench_function("rolling_ball_16x16", |bench| {
        bench.iter(|| morph2d::rolling_ball(black_box(input), params).unwrap())
    });
    group.bench_function("tophat_16x16", |bench| {
        bench.iter(|| morph2d::tophat(black_box(input), params).unwrap())
    });
    group.bench_function("mor_16x16", |bench| {
        bench.iter(|| morph2d::mor(black_box(input), params).unwrap())
    });
    group.bench_function("imor_16x16", |bench| {
        bench.iter(|| morph2d::imor(black_box(input), morph2d::Imor2DParams::default()).unwrap())
    });
    group.bench_function("noise_median_16x16", |bench| {
        bench.iter(|| morph2d::noise_median(black_box(input), params).unwrap())
    });
    group.finish();
}

fn bench_spline_2d(c: &mut Criterion) {
    let rows = 16;
    let cols = 16;
    let data = surface(rows, cols);
    let input = matrix(&data, rows, cols);
    let mut group = c.benchmark_group("spline_2d");
    group.bench_function("pspline_asls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_asls(black_box(input), spline2d::PsplineAsls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_iasls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_iasls(black_box(input), spline2d::PsplineIasls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_airpls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_airpls(black_box(input), spline2d::PsplineAirPls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_arpls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_arpls(black_box(input), spline2d::PsplineArPls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_iarpls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_iarpls(black_box(input), spline2d::PsplineIarPls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_psalsa_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_psalsa(black_box(input), spline2d::PsplinePsalsa2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_brpls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_brpls(black_box(input), spline2d::PsplineBrPls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("pspline_lsrpls_16x16", |bench| {
        bench.iter(|| {
            spline2d::pspline_lsrpls(black_box(input), spline2d::PsplineLsrPls2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("irsqr_16x16", |bench| {
        bench
            .iter(|| spline2d::irsqr(black_box(input), spline2d::Irsqr2DParams::default()).unwrap())
    });
    group.bench_function("mixture_model_16x16", |bench| {
        bench.iter(|| {
            spline2d::mixture_model(black_box(input), spline2d::MixtureModel2DParams::default())
                .unwrap()
        })
    });
    group.finish();
}

fn bench_optimizers_2d(c: &mut Criterion) {
    let rows = 16;
    let cols = 16;
    let data = surface(rows, cols);
    let data_b: Vec<f64> = data.iter().map(|value| value + 0.001).collect();
    let input = matrix(&data, rows, cols);
    let input_b = matrix(&data_b, rows, cols);
    let surfaces = vec![input, input_b];
    let mut group = c.benchmark_group("optimizers_2d");
    group.bench_function("adaptive_minmax_16x16", |bench| {
        bench.iter(|| {
            opt2d::adaptive_minmax(black_box(input), opt2d::AdaptiveMinmax2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("individual_axes_16x16", |bench| {
        bench.iter(|| {
            opt2d::individual_axes(black_box(input), opt2d::IndividualAxes2DParams::default())
                .unwrap()
        })
    });
    group.bench_function("collab_pls_2x16x16", |bench| {
        bench.iter(|| {
            opt2d::collab_pls(
                black_box(surfaces.as_slice()),
                opt2d::CollabPls2DParams::default(),
            )
            .unwrap()
        })
    });
    group.finish();
}

fn bench_batch_cpu(c: &mut Criterion) {
    let one = signal(256);
    let n_spectra = 16;
    let n_points = one.len();
    let mut batch = Vec::with_capacity(n_spectra * n_points);
    for spectrum in 0..n_spectra {
        let shift = spectrum as f64 * 0.001;
        batch.extend(one.iter().map(|value| value + shift));
    }
    let mut output = vec![0.0; batch.len()];
    let mut group = c.benchmark_group("batch_cpu");
    group.bench_function("snip_batch_cpu_16x256", |bench| {
        bench.iter(|| {
            snip_batch_into(
                black_box(batch.as_slice()),
                n_spectra,
                n_points,
                morph::SnipParams { max_half_window: 8 },
                black_box(output.as_mut_slice()),
            )
            .unwrap()
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1));
    targets =
        bench_whittaker_1d,
        bench_polynomial_1d,
        bench_morphology_1d,
        bench_smoothing_1d,
        bench_spline_1d,
        bench_classification_1d,
        bench_optimizers_and_misc_1d,
        bench_whittaker_2d,
        bench_polynomial_2d,
        bench_morphology_2d,
        bench_spline_2d,
        bench_optimizers_2d,
        bench_batch_cpu
}
criterion_main!(benches);
