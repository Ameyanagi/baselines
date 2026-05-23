//! Whittaker-style penalized least-squares baseline algorithms.
//!
//! # References
//!
//! - P. H. C. Eilers, "A Perfect Smoother", *Analytical Chemistry*, 2003.
//! - P. H. C. Eilers and H. F. M. Boelens, "Baseline Correction with
//!   Asymmetric Least Squares Smoothing", 2005.
//! - Z.-M. Zhang, S. Chen, and Y.-Z. Liang, "Baseline correction using
//!   adaptive iteratively reweighted penalized least squares", *Analyst*, 2010.
//! - J. Baek et al., "Baseline correction using asymmetrically reweighted
//!   penalized least squares smoothing", *Analyst*, 2015.
//! - `pybaselines` is used as a behavioral reference.

mod airpls;
mod arpls;
mod asls;
mod engine;
mod variants;

pub use airpls::{AirPlsParams, airpls, airpls_into};
pub use arpls::{ArPlsParams, arpls, arpls_into};
pub use asls::{AslsParams, asls, asls_into, asls_into_with_history, asls_with_history};
pub use engine::{WhittakerParams, WhittakerWorkspace};
pub use variants::{
    AsPlsParams, BrPlsParams, DerPsalsaParams, DrPlsParams, IarPlsParams, IaslsParams,
    LsrPlsParams, PsalsaParams, aspls, aspls_into_with_history, aspls_with_history, brpls,
    derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};
