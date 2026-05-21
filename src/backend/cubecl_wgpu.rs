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
use cubecl::server::Handle;
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

#[cube(launch)]
fn moving_max_kernel(
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
                if candidate > value {
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
            cube_count.clone(),
            cube_dim,
            // SAFETY: `input_handle` was created from `input`, so it owns exactly
            // `total` contiguous `f32` elements for the duration of the launch.
            unsafe { ArrayArg::from_raw_parts(input_handle, total) },
            // SAFETY: `output_handle` was allocated with `size_of_val(input)`
            // bytes immediately above, matching exactly `total` `f32` elements.
            unsafe { ArrayArg::from_raw_parts(output_handle.clone(), total) },
            n_points,
            radius,
        );

        read_f32_output(client, output_handle)
    }

    /// Runs a moving-maximum morphology kernel on the default WGPU device.
    ///
    /// The input layout is spectrum-major.
    pub fn moving_max_batch_f32(
        input: &[f32],
        n_spectra: usize,
        n_points: usize,
        window_size: usize,
    ) -> Result<Vec<f32>> {
        validate_batch(input, n_spectra, n_points)?;
        validate_window_size(window_size)?;
        let client = cubecl::wgpu::WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::DefaultDevice);
        Self::moving_max_batch_f32_on_client(&client, input, n_spectra, n_points, window_size)
    }

    /// Runs a moving-maximum morphology kernel on a caller-provided CubeCL client.
    pub fn moving_max_batch_f32_on_client<R: Runtime>(
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

        moving_max_kernel::launch::<R>(
            client,
            cube_count,
            cube_dim,
            // SAFETY: `input_handle` was created from `input`, so it owns exactly
            // `total` contiguous `f32` elements for the duration of the launch.
            unsafe { ArrayArg::from_raw_parts(input_handle, total) },
            // SAFETY: `output_handle` was allocated with `size_of_val(input)`
            // bytes immediately above, matching exactly `total` `f32` elements.
            unsafe { ArrayArg::from_raw_parts(output_handle.clone(), total) },
            n_points,
            radius,
        );

        read_f32_output(client, output_handle)
    }

    /// Runs morphological opening, moving-minimum followed by moving-maximum,
    /// on the default WGPU device.
    ///
    /// The temporary erosion buffer stays on the GPU between kernels.
    pub fn opening_batch_f32(
        input: &[f32],
        n_spectra: usize,
        n_points: usize,
        window_size: usize,
    ) -> Result<Vec<f32>> {
        validate_batch(input, n_spectra, n_points)?;
        validate_window_size(window_size)?;
        let client = cubecl::wgpu::WgpuRuntime::client(&cubecl::wgpu::WgpuDevice::DefaultDevice);
        Self::opening_batch_f32_on_client(&client, input, n_spectra, n_points, window_size)
    }

    /// Runs morphological opening on a caller-provided CubeCL client.
    pub fn opening_batch_f32_on_client<R: Runtime>(
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
        let temp_handle = client.empty(size_of_val(input));
        let output_handle = client.empty(size_of_val(input));
        let cube_dim = CubeDim::new_1d(128);
        let cube_count = cubecl::calculate_cube_count_elemwise::<R>(client, total, cube_dim);

        moving_min_kernel::launch::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            // SAFETY: `input_handle` was created from `input`, so it owns exactly
            // `total` contiguous `f32` elements for the duration of the launch.
            unsafe { ArrayArg::from_raw_parts(input_handle, total) },
            // SAFETY: `temp_handle` was allocated with `size_of_val(input)`
            // bytes, matching exactly `total` `f32` elements.
            unsafe { ArrayArg::from_raw_parts(temp_handle.clone(), total) },
            n_points,
            radius,
        );
        moving_max_kernel::launch::<R>(
            client,
            cube_count,
            cube_dim,
            // SAFETY: `temp_handle` was produced by the previous kernel and has
            // exactly `total` `f32` elements.
            unsafe { ArrayArg::from_raw_parts(temp_handle, total) },
            // SAFETY: `output_handle` was allocated with `size_of_val(input)`
            // bytes, matching exactly `total` `f32` elements.
            unsafe { ArrayArg::from_raw_parts(output_handle.clone(), total) },
            n_points,
            radius,
        );

        read_f32_output(client, output_handle)
    }

    /// Runs the top-hat baseline primitive used by this crate, equivalent to
    /// morphological opening, on the default WGPU device.
    pub fn tophat_baseline_batch_f32(
        input: &[f32],
        n_spectra: usize,
        n_points: usize,
        window_size: usize,
    ) -> Result<Vec<f32>> {
        Self::opening_batch_f32(input, n_spectra, n_points, window_size)
    }

    /// Runs the top-hat baseline primitive on a caller-provided CubeCL client.
    pub fn tophat_baseline_batch_f32_on_client<R: Runtime>(
        client: &ComputeClient<R>,
        input: &[f32],
        n_spectra: usize,
        n_points: usize,
        window_size: usize,
    ) -> Result<Vec<f32>> {
        Self::opening_batch_f32_on_client(client, input, n_spectra, n_points, window_size)
    }
}

fn read_f32_output<R: Runtime>(
    client: &ComputeClient<R>,
    output_handle: Handle,
) -> Result<Vec<f32>> {
    let bytes = client
        .read_one(output_handle)
        .map_err(|_| BaselineError::Unsupported {
            feature: "gpu-wgpu",
            reason: "failed to read CubeCL WGPU output buffer",
        })?;
    Ok(f32::from_bytes(&bytes).to_vec())
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
    fn morphology_kernels_match_cpu_reference() {
        let input = vec![3.0, 2.0, 4.0, 1.0, 5.0, 8.0, 6.0, 7.0];
        let n_spectra = 2;
        let n_points = 4;
        let window_size = 3;
        let radius = window_size / 2;

        let moving_min =
            CubeClWgpuBackend::moving_min_batch_f32(&input, n_spectra, n_points, window_size)
                .unwrap();
        assert_eq!(moving_min, moving_min_reference(&input, n_points, radius));

        let moving_max =
            CubeClWgpuBackend::moving_max_batch_f32(&input, n_spectra, n_points, window_size)
                .unwrap();
        assert_eq!(moving_max, moving_max_reference(&input, n_points, radius));

        let opening =
            CubeClWgpuBackend::opening_batch_f32(&input, n_spectra, n_points, window_size).unwrap();
        let expected_opening = moving_max_reference(
            &moving_min_reference(&input, n_points, radius),
            n_points,
            radius,
        );
        assert_eq!(opening, expected_opening);

        let top_hat =
            CubeClWgpuBackend::tophat_baseline_batch_f32(&input, n_spectra, n_points, window_size)
                .unwrap();
        assert_eq!(top_hat, expected_opening);
    }

    fn moving_min_reference(input: &[f32], n_points: usize, radius: usize) -> Vec<f32> {
        moving_window_reference(input, n_points, radius, f32::INFINITY, f32::min)
    }

    fn moving_max_reference(input: &[f32], n_points: usize, radius: usize) -> Vec<f32> {
        moving_window_reference(input, n_points, radius, f32::NEG_INFINITY, f32::max)
    }

    fn moving_window_reference(
        input: &[f32],
        n_points: usize,
        radius: usize,
        initial: f32,
        reduce: impl Fn(f32, f32) -> f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for (index, target) in output.iter_mut().enumerate() {
            let point = index % n_points;
            let spectrum_start = index - point;
            let start = point.saturating_sub(radius);
            let end = (point + radius + 1).min(n_points);
            let mut value = initial;
            for candidate in &input[spectrum_start + start..spectrum_start + end] {
                value = reduce(value, *candidate);
            }
            *target = value;
        }
        output
    }
}
