//! Experimental CubeCL WGPU/Metal backend boundary.
//!
//! This module is intentionally narrow while the CPU behavior is stabilized.
//! It compiles only with the `gpu-wgpu` feature and keeps GPU-specific types
//! out of the default public API.

#![allow(unsafe_code)]

use core::mem::size_of_val;

use crate::workspace::validate_output;
use crate::{BaselineError, Result};
use cubecl::prelude::*;
use cubecl::{self as cubecl};

#[cube(launch)]
fn moving_min_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] n_points: usize,
    #[comptime] radius: usize,
) {
    if ABSOLUTE_POS < output.len() {
        let idx = ABSOLUTE_POS;
        let point = idx % n_points;
        let spectrum_start = idx - point;
        let mut start = 0usize;
        if point > radius {
            start = point - radius;
        }
        let mut end = point + radius + 1;
        if end > n_points {
            end = n_points;
        }

        let mut value = input[idx];
        for offset in 0..comptime!(2 * radius + 1) {
            let window_index = start + offset;
            if window_index < end {
                let candidate = input[spectrum_start + window_index];
                if candidate < value {
                    value = candidate;
                }
            }
        }
        output[idx] = value;
    }
}

/// Marker type for the experimental CubeCL WGPU backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeClWgpuBackend;

impl CubeClWgpuBackend {
    /// Returns whether the backend was compiled into this build.
    #[must_use]
    pub fn is_compiled() -> bool {
        true
    }

    /// Returns metadata for the available morphology kernels.
    ///
    /// This proves the CubeCL feature is compiled without initializing a device.
    pub fn morphology_kernels() -> Result<()> {
        let _ = core::any::type_name::<cubecl::prelude::CubeDim>();
        Ok(())
    }

    /// Runs a moving-minimum morphology kernel on the default WGPU device.
    ///
    /// The input layout is spectrum-major: `spectrum0[0..n_points]`,
    /// `spectrum1[0..n_points]`, and so on.
    pub fn moving_min_batch_f32(
        input: &[f32],
        n_spectra: usize,
        n_points: usize,
        window_size: usize,
    ) -> Result<Vec<f32>> {
        validate_batch(input, n_spectra, n_points)?;
        validate_window_size(window_size)?;
        let client = cubecl::wgpu::WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::DefaultDevice);
        Self::moving_min_batch_f32_on_client(&client, input, n_spectra, n_points, window_size)
    }

    /// Runs a moving-minimum morphology kernel on a caller-provided CubeCL client.
    ///
    /// Supplying the client lets applications control device selection and reuse
    /// GPU resources across many batch operations.
    pub fn moving_min_batch_f32_on_client<R: Runtime>(
        client: &ComputeClient<R>,
        input: &[f32],
        n_spectra: usize,
        n_points: usize,
        window_size: usize,
    ) -> Result<Vec<f32>> {
        validate_batch(input, n_spectra, n_points)?;
        validate_window_size(window_size)?;

        let total = input.len();
        let radius = window_size / 2;
        let input_handle = client.create_from_slice(f32::as_bytes(input));
        let output_handle = client.empty(size_of_val(input));
        let cube_dim = CubeDim::new_1d(128);
        let cube_count = cubecl::calculate_cube_count_elemwise::<R>(client, total, cube_dim);

        moving_min_kernel::launch::<R>(
            client,
            cube_count,
            cube_dim,
            // SAFETY: `input_handle` was created from `input`, so it owns exactly
            // `total` contiguous `f32` elements for the duration of the launch.
            unsafe { ArrayArg::from_raw_parts(input_handle, total) },
            // SAFETY: `output_handle` was allocated with `total * size_of::<f32>()`
            // bytes immediately above, matching exactly `total` `f32` elements.
            unsafe { ArrayArg::from_raw_parts(output_handle.clone(), total) },
            n_points,
            radius,
        );

        let bytes = client
            .read_one(output_handle)
            .map_err(|_| BaselineError::Unsupported {
                feature: "gpu-wgpu",
                reason: "failed to read CubeCL WGPU output buffer",
            })?;
        Ok(f32::from_bytes(&bytes).to_vec())
    }
}

fn validate_batch(input: &[f32], n_spectra: usize, n_points: usize) -> Result<()> {
    if input.is_empty() {
        return Err(BaselineError::EmptyInput);
    }
    for (index, value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(BaselineError::NonFiniteInput { index });
        }
    }
    if n_spectra == 0 || n_points == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "batch_shape",
            reason: "n_spectra and n_points must be greater than zero",
        });
    }
    let expected = n_spectra
        .checked_mul(n_points)
        .ok_or(BaselineError::InvalidParameter {
            name: "batch_shape",
            reason: "n_spectra * n_points overflowed",
        })?;
    validate_output("input", expected, input.len())
}

fn validate_window_size(window_size: usize) -> Result<()> {
    if window_size == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "window_size",
            reason: "must be greater than zero",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_metadata_is_available() {
        CubeClWgpuBackend::morphology_kernels().unwrap();
    }

    #[test]
    fn moving_min_rejects_invalid_shape_before_launch() {
        let error = CubeClWgpuBackend::moving_min_batch_f32(&[1.0, 2.0], 1, 3, 3).unwrap_err();
        assert!(matches!(error, BaselineError::LengthMismatch { .. }));
    }

    #[test]
    #[ignore = "requires a working WGPU device; run with `cargo test --features gpu-wgpu -- --ignored`"]
    fn moving_min_matches_cpu_reference() {
        let input = vec![3.0, 2.0, 4.0, 1.0, 5.0, 8.0, 6.0, 7.0];
        let actual = CubeClWgpuBackend::moving_min_batch_f32(&input, 2, 4, 3).unwrap();
        let expected = vec![2.0, 2.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0];
        assert_eq!(actual, expected);
    }
}
