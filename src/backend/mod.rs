//! Execution backends.

pub mod cpu;

#[cfg(feature = "gpu-wgpu")]
pub mod cubecl_wgpu;
