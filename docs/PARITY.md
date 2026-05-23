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
- Native Rust 2D Whittaker implementations currently cover `asls`, `iasls`,
  `airpls`, `arpls`, `drpls`, `iarpls`, `aspls`, `psalsa`, `brpls`, and
  `lsrpls`.
- Native Rust 2D penalized-spline implementations currently cover
  `pspline_asls`, `pspline_iasls`, `pspline_airpls`, `pspline_arpls`,
  `pspline_iarpls`, `pspline_psalsa`, `pspline_brpls`, `pspline_lsrpls`,
  `irsqr`, and `mixture_model`.
- Native Rust 2D optimizer/meta implementations currently cover
  `adaptive_minmax`, `individual_axes`, and `collab_pls`.
- All 33 pinned `pybaselines.Baseline2D` 1.2.1 public methods now have native
  Rust entry points and fixture-backed first-pass behavior.

## Two-Dimensional Tolerance Ledger

The native 2D methods are fixture-backed with a first-pass `3e-1` tolerance
while their padding, basis, weighting, solver, and iteration semantics are
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
| `asls` | 2D Whittaker | `3e-1` | Tighten matrix-free solve settings and weight-update parity. |
| `iasls` | 2D Whittaker | `3e-1` | Add the first-derivative contribution used by pybaselines. |
| `airpls` | 2D Whittaker | `3e-1` | Align adaptive exponential weighting and stopping behavior. |
| `arpls` | 2D Whittaker | `3e-1` | Tighten negative-residual statistics and logistic weighting. |
| `drpls` | 2D Whittaker | `3e-1` | Add full doubly reweighted penalty behavior. |
| `iarpls` | 2D Whittaker | `3e-1` | Align improved arPLS update scaling. |
| `aspls` | 2D Whittaker | `3e-1` | Add adaptive smoothness behavior beyond the first-pass weight policy. |
| `psalsa` | 2D Whittaker | `3e-1` | Tighten exponential peak suppression and default `k` behavior. |
| `brpls` | 2D Whittaker | `3e-1` | Implement the full outer beta iteration semantics. |
| `lsrpls` | 2D Whittaker | `3e-1` | Align locally symmetric reweighting update behavior. |
| `pspline_asls` | 2D spline | `3e-1` | Replace separable first-pass smoothing with full tensor-product P-spline semantics. |
| `pspline_iasls` | 2D spline | `3e-1` | Tighten first-difference residual penalty behavior. |
| `pspline_airpls` | 2D spline | `3e-1` | Align adaptive exponential weighting and spline convergence. |
| `pspline_arpls` | 2D spline | `3e-1` | Tighten negative-residual statistics for spline weights. |
| `pspline_iarpls` | 2D spline | `3e-1` | Align improved arPLS spline update scaling. |
| `pspline_psalsa` | 2D spline | `3e-1` | Tighten exponential peak suppression for spline fits. |
| `pspline_brpls` | 2D spline | `3e-1` | Implement full outer beta iteration semantics for spline fits. |
| `pspline_lsrpls` | 2D spline | `3e-1` | Align locally symmetric spline reweighting behavior. |
| `irsqr` | 2D spline | `3e-1` | Tighten iterative quantile-regression spline weighting and coefficient convergence. |
| `mixture_model` | 2D spline | `3e-1` | Implement full mixture-model weighting instead of the first-pass asymmetric policy. |
| `adaptive_minmax` | 2D optimizer/meta | `3e-1` | Implement full adaptive candidate selection beyond the modified polynomial path. |
| `individual_axes` | 2D optimizer/meta | `3e-1` | Expand beyond row-then-column AsLS and expose method selection. |
| `collab_pls` | 2D optimizer/meta | `3e-1` | Tighten shared-weight collaborative fitting and multi-surface convergence. |

## Tolerance Ledger

No pinned 1D fixture tolerances are currently above `1e-1`.

## Known Limits

- Two-dimensional support covers all pinned `pybaselines.Baseline2D` 1.2.1
  methods with first-pass native Rust implementations.
- Some Rust implementations intentionally use first-pass native solvers or
  approximations while retaining fixture-backed behavior for the tested
  parameter sets.
- `beads` currently supports `filter_type = 1`; unsupported filter types return
  `BaselineError::Unsupported`.
- `fabc` currently supports second-order Whittaker penalties.
- CubeCL WGPU support is experimental and limited to batched `f32` morphology
  primitives.
