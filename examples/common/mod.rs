use ruviz::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};

pub const OUTPUT_DIR: &str = "target/baselines-ruviz";

pub fn ensure_output_dir() -> std::io::Result<()> {
    std::fs::create_dir_all(OUTPUT_DIR)
}

pub fn output_path(name: &str) -> PathBuf {
    Path::new(OUTPUT_DIR).join(name)
}

pub fn print_output(path: &Path) {
    println!("wrote {}", path.display());
}

pub fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let step = (stop - start) / (count - 1) as f64;
    (0..count)
        .map(|index| start + step * index as f64)
        .collect()
}

pub fn gaussian(x: f64, height: f64, center: f64, sigma: f64) -> f64 {
    height * (-0.5 * ((x - center) / sigma).powi(2)).exp()
}

pub fn standard_signal(x: &[f64]) -> Vec<f64> {
    x.iter()
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
        .collect()
}

pub fn add3(left: &[f64], middle: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(middle)
        .zip(right)
        .map(|((a, b), c)| a + b + c)
        .collect()
}

pub fn uniform_filter_reflect(values: &[f64], window_size: usize) -> Vec<f64> {
    let radius = window_size / 2;
    (0..values.len())
        .map(|index| {
            let mut total = 0.0;
            for offset in 0..window_size {
                let source = reflect_index(
                    index as isize + offset as isize - radius as isize,
                    values.len(),
                );
                total += values[source];
            }
            total / window_size as f64
        })
        .collect()
}

pub fn rolling_std_reflect(values: &[f64], half_window: usize) -> Vec<f64> {
    let window_size = 2 * half_window + 1;
    (0..values.len())
        .map(|index| {
            let mut window = Vec::with_capacity(window_size);
            for offset in 0..window_size {
                let source = reflect_index(
                    index as isize + offset as isize - half_window as isize,
                    values.len(),
                );
                window.push(values[source]);
            }
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window
                .iter()
                .map(|value| {
                    let centered = value - mean;
                    centered * centered
                })
                .sum::<f64>()
                / (window.len() - 1) as f64;
            variance.sqrt()
        })
        .collect()
}

pub fn percentile(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[index]
}

pub fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

pub fn linear_interpolate_masked(x: &[f64], y: &[f64], keep: &[bool]) -> Vec<f64> {
    let mut kept = x
        .iter()
        .copied()
        .zip(y.iter().copied())
        .zip(keep.iter().copied());
    let mut anchors: Vec<(f64, f64)> = kept
        .by_ref()
        .filter_map(|((x_value, y_value), keep_value)| keep_value.then_some((x_value, y_value)))
        .collect();
    if anchors.is_empty() {
        return y.to_vec();
    }
    anchors.sort_by(|left, right| left.0.total_cmp(&right.0));

    x.iter()
        .map(|&x_value| {
            if x_value <= anchors[0].0 {
                return anchors[0].1;
            }
            if x_value >= anchors[anchors.len() - 1].0 {
                return anchors[anchors.len() - 1].1;
            }
            let upper = anchors.partition_point(|(anchor_x, _)| *anchor_x < x_value);
            let (x0, y0) = anchors[upper - 1];
            let (x1, y1) = anchors[upper];
            y0 + (y1 - y0) * (x_value - x0) / (x1 - x0)
        })
        .collect()
}

pub fn pad_edges(
    values: &[f64],
    pad_len: usize,
    mode: PadMode,
    extrapolate_window: (usize, usize),
) -> Vec<f64> {
    let mut output = Vec::with_capacity(values.len() + 2 * pad_len);
    match mode {
        PadMode::Reflect => {
            for index in (0..pad_len).rev() {
                output.push(values[reflect_index(index as isize, values.len())]);
            }
            output.extend_from_slice(values);
            for index in values.len()..values.len() + pad_len {
                output.push(values[reflect_index(index as isize, values.len())]);
            }
        }
        PadMode::Edge => {
            output.extend(std::iter::repeat_n(values[0], pad_len));
            output.extend_from_slice(values);
            output.extend(std::iter::repeat_n(values[values.len() - 1], pad_len));
        }
        PadMode::Extrapolate => {
            let (left_slope, left_intercept) = linear_edge_fit(values, extrapolate_window.0, true);
            let (right_slope, right_intercept) =
                linear_edge_fit(values, extrapolate_window.1, false);
            for index in 0..pad_len {
                let x = -((pad_len - index) as f64);
                output.push(left_slope * x + left_intercept);
            }
            output.extend_from_slice(values);
            for index in 0..pad_len {
                let x = values.len() as f64 + index as f64;
                output.push(right_slope * x + right_intercept);
            }
        }
    }
    output
}

pub fn save_lines(
    title: &str,
    x_label: &str,
    y_label: &str,
    x: &[f64],
    series: &[LineSeries<'_>],
    path: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let mut plot = Plot::new()
        .title(title)
        .xlabel(x_label)
        .ylabel(y_label)
        .max_resolution(1800, 1200)
        .legend_position(LegendPosition::Best);
    for item in series {
        plot = plot
            .line(&x, &item.y)
            .label(item.label)
            .color(item.color)
            .into();
    }
    plot.save(path)?;
    Ok(())
}

pub fn save_heatmap(
    flat: &[f64],
    rows: usize,
    cols: usize,
    title: &str,
    label: &str,
    path: &Path,
) -> std::result::Result<(), Box<dyn Error>> {
    let matrix: Vec<Vec<f64>> = flat.chunks(cols).take(rows).map(<[f64]>::to_vec).collect();
    let config = HeatmapConfig::new()
        .colormap(ColorMap::coolwarm())
        .interpolation(Interpolation::Nearest)
        .colorbar_label(label);
    Plot::new()
        .title(title)
        .xlabel("column")
        .ylabel("row")
        .max_resolution(1800, 1200)
        .heatmap(&matrix, Some(config))
        .save(path)?;
    Ok(())
}

pub struct LineSeries<'a> {
    pub label: &'a str,
    pub y: &'a [f64],
    pub color: Color,
}

#[derive(Debug, Clone, Copy)]
pub enum PadMode {
    Reflect,
    Edge,
    Extrapolate,
}

pub struct NormalNoise {
    state: u64,
    spare: Option<f64>,
}

impl NormalNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub fn sample(&mut self, sigma: f64) -> f64 {
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

fn reflect_index(index: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * len as isize - 2;
    let wrapped = index.rem_euclid(period);
    if wrapped < len as isize {
        wrapped as usize
    } else {
        (period - wrapped) as usize
    }
}

fn linear_edge_fit(values: &[f64], window: usize, left: bool) -> (f64, f64) {
    let count = window.clamp(2, values.len());
    let start = if left { 0 } else { values.len() - count };
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    for (index, &y) in values.iter().enumerate().skip(start).take(count) {
        let x = index as f64;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    let count = count as f64;
    let denom = count * sum_xx - sum_x * sum_x;
    let slope = if denom.abs() < f64::EPSILON {
        0.0
    } else {
        (count * sum_xy - sum_x * sum_y) / denom
    };
    let intercept = (sum_y - slope * sum_x) / count;
    (slope, intercept)
}
