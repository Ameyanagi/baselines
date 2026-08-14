#![deny(unsafe_code)]
#![warn(missing_docs)]

//! Baseline correction algorithms for one-dimensional signals and spectra.
//!
//! This crate is an independent Rust implementation inspired by the baseline
//! correction literature. The Python project `pybaselines` is used as an API
//! and behavioral reference, not as copied implementation code.
//!
//! The simplest Rust API estimates or corrects a signal with one function:
//!
//! ```
//! use baselines::prelude::*;
//!
//! let y = vec![1.0, 1.1, 4.2, 1.2, 1.0];
//! let corrected = correct(&y)?;
//! # Ok::<(), baselines::BaselineError>(())
//! ```
//!
//! Use [`Baseline`] and [`Baseline2D`] when you need to tune parameters. The
//! family modules remain public for explicit parameter structs, workspace
//! reuse, and direct behavioral comparisons against published examples.

pub mod api;
pub mod backend;
pub mod classification;
pub mod data;
pub mod error;
pub mod fit;
#[doc = include_str!("../docs/GALLERY.md")]
pub mod gallery {}
pub mod linalg;
pub mod misc;
pub mod morphology;
pub mod optimizers;
pub mod polynomial;
pub mod simple;
pub mod smoothing;
pub mod spline;
pub mod two_d;
pub mod whittaker;
pub mod workspace;
#[doc = include_str!("../docs/PYBASELINES_EXAMPLES.md")]
pub mod reference_examples {}

pub use api::{Baseline, Baseline2D, BaselineXY};
pub use classification::ClassificationFit;
pub use data::{MatrixLayout, MatrixShape, MatrixView, MatrixViewMut};
pub use error::{BaselineError, Result};
pub use fit::{Fit, Fit1D, Fit2D, FitHistory, FitReport};
pub use simple::{
    Method, Method2D, baseline, baseline_2d, baseline_2d_with, baseline_with, correct, correct_2d,
    correct_2d_with, correct_with,
};

/// Common imports for the simple and method-chain APIs.
pub mod prelude {
    pub use crate::{
        Baseline, Baseline2D, BaselineError, BaselineXY, ClassificationFit, Fit, Fit1D, Fit2D,
        FitHistory, FitReport, MatrixView, MatrixViewMut, Method, Method2D, Result, baseline,
        baseline_2d, baseline_2d_with, baseline_with, correct, correct_2d, correct_2d_with,
        correct_with,
    };
}
