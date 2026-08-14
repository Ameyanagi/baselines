//! WebAssembly bindings for the simple `baselines` API.

use baselines::{BaselineOptions, BaselineOptions2D, Fit1D, Fit2D, FitReport, Method, Method2D};
use wasm_bindgen::prelude::*;

fn js_error(error: baselines::BaselineError) -> JsError {
    JsError::new(&error.to_string())
}

/// One-dimensional fit output exposed to the JavaScript facade.
#[wasm_bindgen]
pub struct WasmFitResult {
    baseline: Vec<f64>,
    corrected: Vec<f64>,
    report: FitReport,
}

impl WasmFitResult {
    fn new(input: &[f64], fit: Fit1D) -> Result<Self, JsError> {
        let corrected = fit.corrected(input).map_err(js_error)?;
        Ok(Self {
            baseline: fit.baseline,
            corrected,
            report: fit.report,
        })
    }
}

#[wasm_bindgen]
impl WasmFitResult {
    /// Estimated baseline.
    #[wasm_bindgen(getter)]
    pub fn baseline(&self) -> Vec<f64> {
        self.baseline.clone()
    }

    /// Corrected signal.
    #[wasm_bindgen(getter)]
    pub fn corrected(&self) -> Vec<f64> {
        self.corrected.clone()
    }

    /// Number of iterations performed.
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.report.iterations
    }

    /// Whether the algorithm met its convergence tolerance.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.report.converged
    }

    /// Final convergence metric.
    #[wasm_bindgen(getter)]
    pub fn tolerance(&self) -> f64 {
        self.report.tolerance
    }
}

/// Two-dimensional fit output exposed to the JavaScript facade.
#[wasm_bindgen]
pub struct WasmFitResult2D {
    baseline: Vec<f64>,
    corrected: Vec<f64>,
    rows: usize,
    cols: usize,
    report: FitReport,
}

impl WasmFitResult2D {
    fn new(input: &[f64], fit: Fit2D) -> Result<Self, JsError> {
        let corrected = fit.corrected(input).map_err(js_error)?;
        Ok(Self {
            baseline: fit.baseline,
            corrected,
            rows: fit.rows,
            cols: fit.cols,
            report: fit.report,
        })
    }
}

#[wasm_bindgen]
impl WasmFitResult2D {
    /// Estimated row-major baseline.
    #[wasm_bindgen(getter)]
    pub fn baseline(&self) -> Vec<f64> {
        self.baseline.clone()
    }

    /// Corrected row-major data.
    #[wasm_bindgen(getter)]
    pub fn corrected(&self) -> Vec<f64> {
        self.corrected.clone()
    }

    /// Number of rows.
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[wasm_bindgen(getter)]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of iterations performed.
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> usize {
        self.report.iterations
    }

    /// Whether the algorithm met its convergence tolerance.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.report.converged
    }

    /// Final convergence metric.
    #[wasm_bindgen(getter)]
    pub fn tolerance(&self) -> f64 {
        self.report.tolerance
    }
}

/// Fits a configurable one-dimensional baseline for the JavaScript facade.
#[wasm_bindgen(js_name = fitConfigured)]
#[allow(clippy::too_many_arguments)]
pub fn fit_configured(
    data: &[f64],
    method: &str,
    lambda: Option<f64>,
    p: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    window_size: Option<usize>,
    order: Option<usize>,
) -> Result<WasmFitResult, JsError> {
    let method = method.parse::<Method>().map_err(js_error)?;
    let options = BaselineOptions {
        method,
        lambda,
        p,
        max_iter,
        tol,
        window_size,
        order,
    };
    let fit = baselines::fit_with_options(data, options).map_err(js_error)?;
    WasmFitResult::new(data, fit)
}

/// Fits a configurable row-major two-dimensional baseline for the JavaScript facade.
#[wasm_bindgen(js_name = fit2dConfigured)]
#[allow(clippy::too_many_arguments)]
pub fn fit_2d_configured(
    data: &[f64],
    rows: usize,
    cols: usize,
    method: &str,
    lambda: Option<f64>,
    lambda_rows: Option<f64>,
    lambda_cols: Option<f64>,
    p: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
    cg_max_iter: Option<usize>,
    cg_tol: Option<f64>,
    window_rows: Option<usize>,
    window_cols: Option<usize>,
    order: Option<usize>,
) -> Result<WasmFitResult2D, JsError> {
    let method = method.parse::<Method2D>().map_err(js_error)?;
    let options = BaselineOptions2D {
        method,
        lambda,
        lambda_rows,
        lambda_cols,
        p,
        max_iter,
        tol,
        cg_max_iter,
        cg_tol,
        window_rows,
        window_cols,
        order,
    };
    let fit = baselines::fit_2d_with_options(data, rows, cols, options).map_err(js_error)?;
    WasmFitResult2D::new(data, fit)
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
