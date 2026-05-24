#![deny(unsafe_code)]
#![warn(missing_docs)]

//! Baseline correction algorithms for one-dimensional signals and spectra.
//!
//! This crate is an independent Rust implementation inspired by the baseline
//! correction literature. The Python project `pybaselines` is used as an API
//! and behavioral reference, not as copied implementation code.
//!
//! The recommended Rust API starts from [`Baseline`] for one-dimensional data:
//!
//! ```
//! use baselines::prelude::*;
//!
//! let y = vec![1.0, 1.1, 4.2, 1.2, 1.0];
//! let fit = Baseline::new(&y).asls().lambda(1.0e6).p(0.01).fit()?;
//! let corrected = fit.corrected(&y)?;
//! # Ok::<(), baselines::BaselineError>(())
//! ```
//!
//! Use [`Baseline2D`] for row-major two-dimensional data. The family modules
//! remain public for explicit parameter structs, workspace reuse, and direct
//! behavioral comparisons against published examples.

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
pub mod smoothing;
pub mod spline;
pub mod two_d;
pub mod whittaker;
pub mod workspace;
#[doc = include_str!("../docs/PYBASELINES_EXAMPLES.md")]
pub mod pybaselines_examples {}

pub use api::{Baseline, Baseline2D};
pub use classification::ClassificationFit;
pub use data::{MatrixLayout, MatrixShape, MatrixView, MatrixViewMut};
pub use error::{BaselineError, Result};
pub use fit::{Fit, Fit1D, Fit2D, FitHistory, FitReport};

/// Common imports for the method-chain API.
pub mod prelude {
    pub use crate::{
        Baseline, Baseline2D, BaselineError, ClassificationFit, Fit, Fit1D, Fit2D, FitHistory,
        FitReport, MatrixView, MatrixViewMut, Result,
    };
}
