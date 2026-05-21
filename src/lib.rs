#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Baseline correction algorithms for one-dimensional signals and spectra.
//!
//! This crate is an independent Rust implementation inspired by the baseline
//! correction literature. The Python project `pybaselines` is used as an API
//! and behavioral reference, not as copied implementation code.

pub mod backend;
pub mod error;
pub mod fit;
pub mod linalg;
pub mod morphology;
pub mod polynomial;
pub mod whittaker;
pub mod workspace;

pub use error::{BaselineError, Result};
pub use fit::{Fit, FitReport};
pub use morphology::{MorphologyParams, SnipParams, mor, mwmv, rolling_ball, snip, tophat};
pub use polynomial::{ImodPolyParams, ModPolyParams, PolyParams, imodpoly, modpoly, poly};
pub use whittaker::{AirPlsParams, ArPlsParams, AslsParams, WhittakerParams, airpls, arpls, asls};
