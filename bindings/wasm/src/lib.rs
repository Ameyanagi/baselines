//! WebAssembly bindings for the simple `baselines` API.

use baselines::{Method, Method2D};
use wasm_bindgen::prelude::*;

fn js_error(error: baselines::BaselineError) -> JsError {
    JsError::new(&error.to_string())
}

/// Estimates an AsLS baseline with default parameters.
#[wasm_bindgen]
pub fn baseline(data: &[f64]) -> Result<Vec<f64>, JsError> {
    baselines::baseline(data).map_err(js_error)
}

/// Estimates a baseline with a selected method and its default parameters.
#[wasm_bindgen(js_name = baselineWith)]
pub fn baseline_with(data: &[f64], method: &str) -> Result<Vec<f64>, JsError> {
    let method = method.parse::<Method>().map_err(js_error)?;
    baselines::baseline_with(data, method).map_err(js_error)
}

/// Corrects a signal with AsLS and default parameters.
#[wasm_bindgen]
pub fn correct(data: &[f64]) -> Result<Vec<f64>, JsError> {
    baselines::correct(data).map_err(js_error)
}

/// Corrects a signal with a selected method and its default parameters.
#[wasm_bindgen(js_name = correctWith)]
pub fn correct_with(data: &[f64], method: &str) -> Result<Vec<f64>, JsError> {
    let method = method.parse::<Method>().map_err(js_error)?;
    baselines::correct_with(data, method).map_err(js_error)
}

/// Estimates a row-major 2D baseline with a selected method and its defaults.
#[wasm_bindgen(js_name = baseline2d)]
pub fn baseline_2d(
    data: &[f64],
    rows: usize,
    cols: usize,
    method: &str,
) -> Result<Vec<f64>, JsError> {
    let method = method.parse::<Method2D>().map_err(js_error)?;
    baselines::baseline_2d_with(data, rows, cols, method).map_err(js_error)
}

/// Corrects row-major 2D data with a selected method and its defaults.
#[wasm_bindgen(js_name = correct2d)]
pub fn correct_2d(
    data: &[f64],
    rows: usize,
    cols: usize,
    method: &str,
) -> Result<Vec<f64>, JsError> {
    let method = method.parse::<Method2D>().map_err(js_error)?;
    baselines::correct_2d_with(data, rows, cols, method).map_err(js_error)
}

/// Returns the comma-separated one-dimensional method names.
#[wasm_bindgen(js_name = availableMethods)]
pub fn available_methods() -> String {
    Method::ALL
        .into_iter()
        .map(Method::as_str)
        .collect::<Vec<_>>()
        .join(",")
}

/// Returns the comma-separated two-dimensional method names.
#[wasm_bindgen(js_name = availableMethods2d)]
pub fn available_methods_2d() -> String {
    Method2D::ALL
        .into_iter()
        .map(Method2D::as_str)
        .collect::<Vec<_>>()
        .join(",")
}
