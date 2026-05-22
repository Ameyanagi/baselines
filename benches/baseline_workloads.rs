use baselines::backend::cpu::snip_batch_into;
use baselines::classification::{CwtBrParams, cwt_br};
use baselines::misc::{BeadsParams, beads};
use baselines::morphology::{MorphologyParams, SnipParams, rolling_ball};
use baselines::spline::pspline_asls;
use baselines::whittaker::{ArPlsParams, AslsParams, arpls, asls};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

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

fn bench_whittaker(c: &mut Criterion) {
    let y = signal(512);
    c.bench_function("asls_512", |bench| {
        bench.iter(|| asls(black_box(y.as_slice()), AslsParams::default()).unwrap())
    });
    c.bench_function("arpls_512", |bench| {
        bench.iter(|| arpls(black_box(y.as_slice()), ArPlsParams::default()).unwrap())
    });
}

fn bench_morphology_and_spline(c: &mut Criterion) {
    let y = signal(512);
    c.bench_function("rolling_ball_512", |bench| {
        bench.iter(|| {
            rolling_ball(
                black_box(y.as_slice()),
                MorphologyParams { window_size: 17 },
            )
            .unwrap()
        })
    });
    c.bench_function("pspline_asls_512", |bench| {
        bench.iter(|| pspline_asls(black_box(y.as_slice()), AslsParams::default()).unwrap())
    });
}

fn bench_classification_and_misc(c: &mut Criterion) {
    let y = signal(256);
    c.bench_function("cwt_br_256", |bench| {
        bench.iter(|| {
            cwt_br(
                black_box(y.as_slice()),
                CwtBrParams {
                    poly_order: 2,
                    scales: Some(vec![8]),
                    ..CwtBrParams::default()
                },
            )
            .unwrap()
        })
    });
    c.bench_function("beads_256", |bench| {
        bench.iter(|| beads(black_box(y.as_slice()), BeadsParams::default()).unwrap())
    });
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
    c.bench_function("snip_batch_cpu_16x256", |bench| {
        bench.iter(|| {
            snip_batch_into(
                black_box(batch.as_slice()),
                n_spectra,
                n_points,
                SnipParams { max_half_window: 8 },
                black_box(output.as_mut_slice()),
            )
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_whittaker,
    bench_morphology_and_spline,
    bench_classification_and_misc,
    bench_batch_cpu
);
criterion_main!(benches);
