//! Python bindings for the configurable `baselines` API.

use baselines::{
    BaselineError, BaselineOptions, BaselineOptions2D, Fit1D, Fit2D, FitReport, Method, Method2D,
};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const OPTIONS_1D: &[&str] = &[
    "lambda",
    "lam",
    "p",
    "max_iter",
    "tol",
    "window_size",
    "order",
];
const OPTIONS_2D: &[&str] = &[
    "lambda",
    "lam",
    "lambda_rows",
    "lambda_cols",
    "p",
    "max_iter",
    "tol",
    "cg_max_iter",
    "cg_tol",
    "window_rows",
    "window_cols",
    "order",
];

fn python_error(error: BaselineError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn validate_option_keys(options: Option<&Bound<'_, PyDict>>, allowed: &[&str]) -> PyResult<()> {
    let Some(options) = options else {
        return Ok(());
    };
    for (key, _) in options.iter() {
        let key = key.extract::<String>()?;
        if !allowed.contains(&key.as_str()) {
            return Err(PyValueError::new_err(format!(
                "unknown option '{key}'; expected one of {}",
                allowed.join(", ")
            )));
        }
    }
    Ok(())
}

fn get_f64(options: Option<&Bound<'_, PyDict>>, name: &str) -> PyResult<Option<f64>> {
    match options {
        Some(options) => match options.get_item(name)? {
            Some(value) => value.extract().map(Some),
            None => Ok(None),
        },
        None => Ok(None),
    }
}

fn get_usize(options: Option<&Bound<'_, PyDict>>, name: &str) -> PyResult<Option<usize>> {
    match options {
        Some(options) => match options.get_item(name)? {
            Some(value) => value.extract().map(Some),
            None => Ok(None),
        },
        None => Ok(None),
    }
}

fn get_lambda(options: Option<&Bound<'_, PyDict>>) -> PyResult<Option<f64>> {
    let lambda = get_f64(options, "lambda")?;
    let lam = get_f64(options, "lam")?;
    if lambda.is_some() && lam.is_some() {
        return Err(PyValueError::new_err(
            "options cannot contain both 'lambda' and its 'lam' alias",
        ));
    }
    Ok(lambda.or(lam))
}

fn options_1d(method: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<BaselineOptions> {
    validate_option_keys(options, OPTIONS_1D)?;
    Ok(BaselineOptions {
        method: method.parse::<Method>().map_err(python_error)?,
        lambda: get_lambda(options)?,
        p: get_f64(options, "p")?,
        max_iter: get_usize(options, "max_iter")?,
        tol: get_f64(options, "tol")?,
        window_size: get_usize(options, "window_size")?,
        order: get_usize(options, "order")?,
    })
}

fn options_2d(method: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<BaselineOptions2D> {
    validate_option_keys(options, OPTIONS_2D)?;
    Ok(BaselineOptions2D {
        method: method.parse::<Method2D>().map_err(python_error)?,
        lambda: get_lambda(options)?,
        lambda_rows: get_f64(options, "lambda_rows")?,
        lambda_cols: get_f64(options, "lambda_cols")?,
        p: get_f64(options, "p")?,
        max_iter: get_usize(options, "max_iter")?,
        tol: get_f64(options, "tol")?,
        cg_max_iter: get_usize(options, "cg_max_iter")?,
        cg_tol: get_f64(options, "cg_tol")?,
        window_rows: get_usize(options, "window_rows")?,
        window_cols: get_usize(options, "window_cols")?,
        order: get_usize(options, "order")?,
    })
}

fn report_dict<'py>(py: Python<'py>, report: FitReport) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("iterations", report.iterations)?;
    output.set_item("converged", report.converged)?;
    output.set_item("tolerance", report.tolerance)?;
    Ok(output)
}

fn array_2d<'py>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let output = Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(output.into_pyarray(py))
}

#[pyfunction(signature = (y, method = "asls", options = None))]
fn baseline<'py>(
    py: Python<'py>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    method: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let options = options_1d(method, options)?;
    let input: Vec<f64> = y.as_array().iter().copied().collect();
    let output = py
        .detach(move || baselines::baseline_with_options(&input, options))
        .map_err(python_error)?;
    Ok(output.into_pyarray(py))
}

#[pyfunction(signature = (y, method = "asls", options = None))]
fn correct<'py>(
    py: Python<'py>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    method: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let options = options_1d(method, options)?;
    let input: Vec<f64> = y.as_array().iter().copied().collect();
    let output = py
        .detach(move || baselines::correct_with_options(&input, options))
        .map_err(python_error)?;
    Ok(output.into_pyarray(py))
}

#[pyfunction(signature = (y, method = "asls", options = None))]
fn fit<'py>(
    py: Python<'py>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    method: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = options_1d(method, options)?;
    let input: Vec<f64> = y.as_array().iter().copied().collect();
    let (fit, corrected) = py
        .detach(move || {
            let fit = baselines::fit_with_options(&input, options)?;
            let corrected = fit.corrected(&input)?;
            Ok::<(Fit1D, Vec<f64>), BaselineError>((fit, corrected))
        })
        .map_err(python_error)?;

    let result = PyDict::new(py);
    result.set_item("baseline", fit.baseline.into_pyarray(py))?;
    result.set_item("corrected", corrected.into_pyarray(py))?;
    result.set_item("report", report_dict(py, fit.report)?)?;
    Ok(result)
}

#[pyfunction(signature = (data, method = "asls", options = None))]
fn baseline_2d<'py>(
    py: Python<'py>,
    data: PyArrayLike2<'py, f64, AllowTypeChange>,
    method: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let options = options_2d(method, options)?;
    let view = data.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let input: Vec<f64> = view.iter().copied().collect();
    let output = py
        .detach(move || baselines::baseline_2d_with_options(&input, rows, cols, options))
        .map_err(python_error)?;
    array_2d(py, rows, cols, output)
}

#[pyfunction(signature = (data, method = "asls", options = None))]
fn correct_2d<'py>(
    py: Python<'py>,
    data: PyArrayLike2<'py, f64, AllowTypeChange>,
    method: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let options = options_2d(method, options)?;
    let view = data.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let input: Vec<f64> = view.iter().copied().collect();
    let output = py
        .detach(move || baselines::correct_2d_with_options(&input, rows, cols, options))
        .map_err(python_error)?;
    array_2d(py, rows, cols, output)
}

#[pyfunction(signature = (data, method = "asls", options = None))]
fn fit_2d<'py>(
    py: Python<'py>,
    data: PyArrayLike2<'py, f64, AllowTypeChange>,
    method: &str,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let options = options_2d(method, options)?;
    let view = data.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let input: Vec<f64> = view.iter().copied().collect();
    let (fit, corrected) = py
        .detach(move || {
            let fit = baselines::fit_2d_with_options(&input, rows, cols, options)?;
            let corrected = fit.corrected(&input)?;
            Ok::<(Fit2D, Vec<f64>), BaselineError>((fit, corrected))
        })
        .map_err(python_error)?;

    let result = PyDict::new(py);
    result.set_item("baseline", array_2d(py, fit.rows, fit.cols, fit.baseline)?)?;
    result.set_item("corrected", array_2d(py, rows, cols, corrected)?)?;
    result.set_item("report", report_dict(py, fit.report)?)?;
    Ok(result)
}

#[pyfunction]
fn methods() -> Vec<&'static str> {
    Method::ALL.into_iter().map(Method::as_str).collect()
}

#[pyfunction]
fn methods_2d() -> Vec<&'static str> {
    Method2D::ALL.into_iter().map(Method2D::as_str).collect()
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(baseline, module)?)?;
    module.add_function(wrap_pyfunction!(correct, module)?)?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    module.add_function(wrap_pyfunction!(baseline_2d, module)?)?;
    module.add_function(wrap_pyfunction!(correct_2d, module)?)?;
    module.add_function(wrap_pyfunction!(fit_2d, module)?)?;
    module.add_function(wrap_pyfunction!(methods, module)?)?;
    module.add_function(wrap_pyfunction!(methods_2d, module)?)?;
    Ok(())
}
