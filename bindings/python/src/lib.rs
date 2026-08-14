//! Python bindings for the simple `baselines` API.

use baselines::{BaselineError, Method, Method2D};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Transform2D = fn(&[f64], usize, usize, Method2D) -> baselines::Result<Vec<f64>>;

fn python_error(error: BaselineError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction(signature = (y, method = "asls"))]
fn baseline<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let method = method.parse::<Method>().map_err(python_error)?;
    let input: Vec<f64> = y.as_array().iter().copied().collect();
    let output = py
        .detach(move || baselines::baseline_with(&input, method))
        .map_err(python_error)?;
    Ok(output.into_pyarray(py))
}

#[pyfunction(signature = (y, method = "asls"))]
fn correct<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let method = method.parse::<Method>().map_err(python_error)?;
    let input: Vec<f64> = y.as_array().iter().copied().collect();
    let output = py
        .detach(move || baselines::correct_with(&input, method))
        .map_err(python_error)?;
    Ok(output.into_pyarray(py))
}

#[pyfunction(signature = (data, method = "asls"))]
fn baseline_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    transform_2d(py, data, method, baselines::baseline_2d_with)
}

#[pyfunction(signature = (data, method = "asls"))]
fn correct_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    transform_2d(py, data, method, baselines::correct_2d_with)
}

fn transform_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    method: &str,
    operation: Transform2D,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let method = method.parse::<Method2D>().map_err(python_error)?;
    let view = data.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let input: Vec<f64> = view.iter().copied().collect();
    let output = py
        .detach(move || operation(&input, rows, cols, method))
        .map_err(python_error)?;
    let output = Array2::from_shape_vec((rows, cols), output)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(output.into_pyarray(py))
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
    module.add_function(wrap_pyfunction!(baseline_2d, module)?)?;
    module.add_function(wrap_pyfunction!(correct_2d, module)?)?;
    module.add_function(wrap_pyfunction!(methods, module)?)?;
    module.add_function(wrap_pyfunction!(methods_2d, module)?)?;
    Ok(())
}
