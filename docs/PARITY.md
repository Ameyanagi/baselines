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

No pinned 2D fixture tolerances are currently above `1e-1`.

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
