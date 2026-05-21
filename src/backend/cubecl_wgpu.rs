//! Experimental CubeCL WGPU/Metal backend boundary.
//!
//! This module is intentionally narrow while the CPU behavior is stabilized.
//! It compiles only with the `gpu-wgpu` feature and keeps GPU-specific types
//! out of the default public API.

use crate::{BaselineError, Result};

/// Marker type for the experimental CubeCL WGPU backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeClWgpuBackend;

impl CubeClWgpuBackend {
    /// Returns whether the backend was compiled into this build.
    #[must_use]
    pub fn is_compiled() -> bool {
        true
    }

    /// Placeholder for future batch morphology kernels.
    pub fn morphology_kernels() -> Result<()> {
        let _ = core::any::type_name::<cubecl::prelude::CubeDim>();
        Err(BaselineError::Unsupported {
            feature: "gpu-wgpu",
            reason: "CubeCL morphology kernels are planned after CPU parity fixtures are stable",
        })
    }
}
