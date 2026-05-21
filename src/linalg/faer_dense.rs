//! Dense linear algebra helpers backed by `faer`.

/// Marker for the optional faer backend.
///
/// The current public algorithms use small in-crate solvers for portability;
/// this module keeps the feature boundary explicit for later reference solves
/// and larger dense systems.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaerDenseBackend;
