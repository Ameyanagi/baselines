#![deny(unsafe_code)]
#![warn(missing_docs)]

//! Baseline correction algorithms for one-dimensional signals and spectra.
//!
//! This crate is an independent Rust implementation inspired by the baseline
//! correction literature. The Python project `pybaselines` is used as an API
//! and behavioral reference, not as copied implementation code.

pub mod backend;
pub mod classification;
pub mod data;
pub mod error;
pub mod fit;
pub mod linalg;
pub mod misc;
pub mod morphology;
pub mod optimizers;
pub mod polynomial;
pub mod smoothing;
pub mod spline;
pub mod whittaker;
pub mod workspace;

pub use data::{MatrixLayout, MatrixShape, MatrixView, MatrixViewMut};
pub use error::{BaselineError, Result};
pub use fit::{Fit, Fit1D, Fit2D, FitReport};
