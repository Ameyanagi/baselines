//! Linear algebra helpers.

pub(crate) mod banded;
pub(crate) mod dense;
pub mod pentadiagonal;
pub(crate) mod pspline;

#[cfg(feature = "faer")]
pub mod faer_dense;
