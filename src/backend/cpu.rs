//! CPU backend helpers.

use crate::Result;
use crate::fit::FitReport;
use crate::morphology::{SnipParams, snip_into};
use crate::workspace::{validate_output, validate_signal};

/// Runs SNIP independently for each contiguous spectrum in `input`.
pub fn snip_batch_into(
    input: &[f64],
    n_spectra: usize,
    n_points: usize,
    params: SnipParams,
    output: &mut [f64],
) -> Result<Vec<FitReport>> {
    validate_batch(input, n_spectra, n_points, output)?;
    let mut reports = Vec::with_capacity(n_spectra);
    for spectrum_index in 0..n_spectra {
        let start = spectrum_index * n_points;
        let end = start + n_points;
        reports.push(snip_into(
            &input[start..end],
            params,
            &mut output[start..end],
        )?);
    }
    Ok(reports)
}

fn validate_batch(input: &[f64], n_spectra: usize, n_points: usize, output: &[f64]) -> Result<()> {
    if n_spectra == 0 || n_points == 0 {
        validate_signal(input)?;
    }
    let expected = n_spectra * n_points;
    validate_output("input", expected, input.len())?;
    validate_output("output", expected, output.len())
}
