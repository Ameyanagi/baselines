# One-Dimensional Parity

`baselines` targets the public one-dimensional `pybaselines.Baseline` method
surface from the pinned fixture version, currently `pybaselines` 1.2.1. The
crate is an independent Rust implementation; pybaselines is used for public
API comparison and behavioral fixtures only.

## Method Surface

| Family | Methods | Status |
| --- | --- | --- |
| Whittaker | `asls`, `airpls`, `arpls`, `drpls`, `iasls`, `iarpls`, `aspls`, `psalsa`, `derpsalsa`, `brpls`, `lsrpls` | Implemented and fixture-backed |
| Polynomial | `poly`, `modpoly`, `imodpoly`, `loess`, `penalized_poly`, `quant_reg`, `goldindec` | Implemented and fixture-backed |
| Morphology | `rolling_ball`, `mwmv`, `tophat`, `mor`, `mpls`, `imor`, `mormol`, `amormol`, `mpspline`, `jbcd` | Implemented and fixture-backed |
| Smoothing | `noise_median`, `snip`, `swima`, `ipsa`, `ria`, `peak_filling` | Implemented and fixture-backed |
| Classification | `rubberband`, `dietrich`, `golotvin`, `std_distribution`, `fastchrom`, `cwt_br`, `fabc` | Implemented and fixture-backed |
| Spline | `pspline_asls`, `pspline_iasls`, `pspline_airpls`, `pspline_arpls`, `pspline_drpls`, `pspline_iarpls`, `pspline_aspls`, `pspline_psalsa`, `pspline_derpsalsa`, `pspline_lsrpls`, `pspline_brpls`, `pspline_mpls`, `corner_cutting`, `irsqr`, `mixture_model` | Implemented and fixture-backed |
| Optimizer/meta | `adaptive_minmax`, `optimize_extended_range`, `custom_bc`, `collab_pls` | Implemented and fixture-backed |
| Misc | `interp_pts`, `beads` | Implemented and fixture-backed |

The generated fixture file records the pinned pybaselines method list. The Rust
fixture test fails if that method list drifts from the expected 62 one-dimensional
methods.

## Current Fixture Depth

- The reference fixture signal checks all 62 one-dimensional methods plus the
  collaborative outputs needed for `collab_pls`.
- Additional deterministic targeted cases cover short, noisy chromatogram-like,
  broad-baseline, and mixed positive/negative peak signals.
- Targeted cases currently exercise representative fragile or high-value paths:
  `asls`, `arpls`, `rolling_ball`, `pspline_asls`, `cwt_br`, `custom_bc`,
  `rubberband`, and `beads`.

## Two-Dimensional Fixture Status

- `tests/fixtures/pybaselines_2d_reference.json` records the pinned
  `pybaselines.Baseline2D` 1.2.1 method list and deterministic row-major
  reference outputs.
- The reference 2D surface covers all 33 public `Baseline2D` methods, with
  `collab_pls` represented by two collaborative output baselines.
- Targeted 2D cases currently cover tilted-plane, ridge/valley, and noisy
  image-like surfaces for representative methods.
- Native Rust 2D morphology/smoothing implementations currently cover
  `rolling_ball`, `tophat`, `mor`, `imor`, and `noise_median`.
- Native Rust 2D polynomial implementations currently cover `poly`, `modpoly`,
  `imodpoly`, `penalized_poly`, and `quant_reg`.
- The remaining native Rust 2D families are still pending; the fixture tests
  lock the upstream method surface and fixture integrity while those methods
  are implemented.

## Two-Dimensional Tolerance Ledger

The first native 2D morphology/smoothing methods are fixture-backed with a
`3e-1` tolerance while their padding, window, and iteration semantics are
tightened against pybaselines.

| Method | Family | Current fixture tolerance | Hardening direction |
| --- | --- | ---: | --- |
| `rolling_ball` | 2D morphology | `3e-1` | Align rolling-ball padding and smoothing behavior. |
| `tophat` | 2D morphology | `3e-1` | Match pybaselines' 2D morphological opening behavior. |
| `mor` | 2D morphology | `3e-1` | Tighten opening/closing envelope averaging. |
| `imor` | 2D morphology | `3e-1` | Align iterative morphology convergence and update policy. |
| `noise_median` | 2D smoothing | `3e-1` | Match median padding and smoothing behavior. |
| `poly` | 2D polynomial | `3e-1` | Align basis ordering and axis scaling with pybaselines. |
| `modpoly` | 2D polynomial | `3e-1` | Tighten clipped least-squares update semantics. |
| `imodpoly` | 2D polynomial | `3e-1` | Tighten improved clipping and weighting semantics. |
| `penalized_poly` | 2D polynomial | `3e-1` | Expand cost-function support beyond the default asymmetric truncated quadratic path. |
| `quant_reg` | 2D polynomial | `3e-1` | Align quantile IRLS weighting and convergence behavior. |

## Tolerance Ledger

The table below lists fixture tolerances above `1e-1`. These are the current
1D hardening priorities; each entry remains fixture-backed but should be
tightened before broader 2D work depends on the same primitive.

| Method | Family | Current fixture tolerance | Hardening direction |
| --- | --- | ---: | --- |
| `peak_filling` | Smoothing | `7e-1` | Match pybaselines' iterative peak-fill window behavior instead of the current conservative neighbor fill. |
| `loess` | Polynomial | `6e-1` | Replace the first-pass moving local constant estimate with weighted local regression semantics. |
| `ria` | Smoothing | `4e-1` | Align the range-independent averaging update and stopping behavior. |
| `swima` | Smoothing | `3.5e-1` | Align the moving-average window adaptation with pybaselines. |
| `ipsa` | Smoothing | `3.5e-1` | Align the iterative polynomial-style averaging limiter and iteration policy. |
| `amormol` | Morphology | `2e-1` | Tighten adaptive morphology weighting around peak regions. |
| `noise_median` | Smoothing | `1.3e-1` | Match median padding and optional smoothing behavior more closely. |

## Known Limits

- This parity document covers one-dimensional `pybaselines.Baseline` methods;
  two-dimensional pybaselines APIs are planned next and not yet implemented.
- Some Rust implementations intentionally use first-pass native solvers or
  approximations while retaining fixture-backed behavior for the tested
  parameter sets.
- `beads` currently supports `filter_type = 1`; unsupported filter types return
  `BaselineError::Unsupported`.
- `fabc` currently supports second-order Whittaker penalties.
- CubeCL WGPU support is experimental and limited to batched `f32` morphology
  primitives.
