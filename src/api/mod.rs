//! Ergonomic method-chain API.
//!
//! The builders in this module are thin wrappers over the family modules. See
//! the underlying algorithm functions for detailed references and numerical
//! notes.

#![allow(missing_docs)]
#![allow(clippy::double_must_use)]

mod one_d;
mod two_d;

pub use one_d::{Baseline, CollabPlsBuilder};
pub use two_d::{Baseline2D, CollabPls2DBuilder};
