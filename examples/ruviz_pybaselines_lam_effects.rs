use baselines::whittaker::{ArPlsParams, WhittakerParams, arpls};
use ruviz::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};

const N: usize = 1000;
const OUTPUT_DIR: &str = "target/baselines-ruviz";
const LAMBDAS: [f64; 4] = [1.0, 1.0e3, 1.0e6, 1.0e10];

fn main() -> std::result::Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(OUTPUT_DIR)?;

    let (x, y, true_baseline) = pybaselines_lam_effects_signal();
    let fits: Vec<(f64, Vec<f64>)> = LAMBDAS
        .iter()
        .map(|&lambda| {
            let fit = arpls(
                &y,
                ArPlsParams {
                    whittaker: WhittakerParams {
                        lambda,
                        ..WhittakerParams::default()
                    },
                },
            )?;
            Ok((lambda, fit.baseline))
        })
        .collect::<baselines::Result<_>>()?;

    let combined_path = output_path("pybaselines_lam_effects.png");
    Plot::new()
        .title("pybaselines lam effects")
        .xlabel("x")
        .ylabel("intensity")
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best)
        .line(&x, &y)
        .label("data")
        .color(Color::new(43, 70, 104))
        .line(&x, &true_baseline)
        .label("true baseline")
        .color(Color::new(80, 145, 110))
        .line(&x, &fits[0].1)
        .label(lambda_label(fits[0].0))
        .color(Color::new(218, 111, 76))
        .line(&x, &fits[1].1)
        .label(lambda_label(fits[1].0))
        .color(Color::new(232, 168, 72))
        .line(&x, &fits[2].1)
        .label(lambda_label(fits[2].0))
        .color(Color::new(84, 151, 160))
        .line(&x, &fits[3].1)
        .label(lambda_label(fits[3].0))
        .color(Color::new(118, 85, 148))
        .save(&combined_path)?;

    print_output(&combined_path);
    for (lambda, baseline) in &fits {
        let path = output_path(&format!(
            "pybaselines_lam_effects_1e{:.0}.png",
            lambda.log10()
        ));
        Plot::new()
            .title(format!(
                "pybaselines lam effects: {}",
                lambda_label(*lambda)
            ))
            .xlabel("x")
            .ylabel("intensity")
            .max_resolution(1800, 1200)
            .legend_position(LegendPosition::Best)
            .line(&x, &y)
            .label("data")
            .color(Color::new(43, 70, 104))
            .line(&x, baseline)
            .label(lambda_label(*lambda))
            .color(Color::new(218, 111, 76))
            .save(&path)?;
        print_output(&path);
    }

    Ok(())
}

fn pybaselines_lam_effects_signal() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = linspace(0.0, 1000.0, N);
    let signal: Vec<f64> = x
        .iter()
        .map(|&value| {
            gaussian(value, 9.0, 100.0, 12.0)
                + gaussian(value, 6.0, 180.0, 5.0)
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
        .map(|&value| 5.0 + 10.0 * (-value / 800.0).exp())
        .collect();
    let mut noise = NormalNoise::new(0);
    let y: Vec<f64> = signal
        .iter()
        .zip(&baseline)
        .map(|(peak, baseline)| peak + baseline + noise.sample(0.2))
        .collect();

    (x, y, baseline)
}

fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let step = (stop - start) / (count - 1) as f64;
    (0..count)
        .map(|index| start + step * index as f64)
        .collect()
}

fn gaussian(x: f64, height: f64, center: f64, sigma: f64) -> f64 {
    height * (-0.5 * ((x - center) / sigma).powi(2)).exp()
}

fn lambda_label(lambda: f64) -> String {
    format!("lam=1e{:.0}", lambda.log10())
}

fn output_path(name: &str) -> PathBuf {
    Path::new(OUTPUT_DIR).join(name)
}

fn print_output(path: &Path) {
    println!("wrote {}", path.display());
}

struct NormalNoise {
    state: u64,
    spare: Option<f64>,
}

impl NormalNoise {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn sample(&mut self, sigma: f64) -> f64 {
        if let Some(value) = self.spare.take() {
            return sigma * value;
        }

        let u1 = self.next_unit_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_unit_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(radius * theta.sin());
        sigma * radius * theta.cos()
    }

    fn next_unit_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        value as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}
