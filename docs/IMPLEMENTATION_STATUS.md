# Implementation Status

This crate exposes public entry points for the current one-dimensional
`pybaselines.Baseline` algorithm set. The implementation is intentionally
staged: APIs and safety/quality gates are in place first, then behavior parity
is tightened with generated pybaselines fixtures.

## Rust API Foundation

- Public algorithm entry points live in their family modules, such as
  `baselines::whittaker::asls`; root exports are reserved for core data and
  error types.
- `Fit1D` is the primary one-dimensional output type. `Fit` remains as a
  compatibility alias while call sites migrate.
- `Fit2D`, `MatrixView`, and `MatrixViewMut` provide the row-major,
  slice-based foundation for two-dimensional algorithms.
- Correction helpers validate input and output lengths instead of silently
  truncating mismatched slices.

## Two-Dimensional Implementations

- Morphology/smoothing: `rolling_ball`, `tophat`, `mor`, `imor`, and
  `noise_median` are implemented under `baselines::two_d::morphology` with
  owned and `_into` row-major APIs.
- Polynomial: `poly`, `modpoly`, `imodpoly`, `penalized_poly`, and `quant_reg`
  are implemented under `baselines::two_d::polynomial` with owned and `_into`
  row-major APIs.
- Whittaker: `asls`, `iasls`, `airpls`, `arpls`, `drpls`, `iarpls`, `aspls`,
  `psalsa`, `brpls`, and `lsrpls` are implemented under
  `baselines::two_d::whittaker` with owned and `_into` row-major APIs backed by
  a matrix-free conjugate-gradient smoother.
- Penalized spline: `pspline_asls`, `pspline_iasls`, `pspline_airpls`,
  `pspline_arpls`, `pspline_iarpls`, `pspline_psalsa`, `pspline_brpls`,
  `pspline_lsrpls`, `irsqr`, and `mixture_model` are implemented under
  `baselines::two_d::spline` with separable row/column P-spline passes.
- Optimizer/meta: `adaptive_minmax`, `individual_axes`, and `collab_pls` are
  implemented under `baselines::two_d::optimizers`.
- All 33 pinned `pybaselines.Baseline2D` 1.2.1 public methods now have native
  Rust entry points and fixture-backed first-pass behavior.

## Dedicated First-Pass Implementations

- Whittaker core: `asls`, `airpls`, `arpls`, `drpls`, `iasls`, `iarpls`,
  `aspls`, `psalsa`, `derpsalsa`, `brpls`, `lsrpls`
- Polynomial core: `poly`, `modpoly`, `imodpoly`, `loess`,
  `penalized_poly`, `quant_reg`, `goldindec`
- Morphology core: `rolling_ball`, `mwmv`, `tophat`, `mor`, `mpls`, `imor`,
  `mormol`, `amormol`, `mpspline`, `jbcd`
- Smoothing core: `noise_median`, `snip`, `swima`, `ipsa`, `ria`, `peak_filling`
- Classification core: `rubberband`, `dietrich`, `golotvin`,
  `std_distribution`, `fastchrom`, `cwt_br`, `fabc`
- Spline core: `pspline_asls`, `pspline_iasls`, `pspline_airpls`, `pspline_arpls`,
  `pspline_drpls`, `pspline_iarpls`, `pspline_aspls`, `pspline_psalsa`,
  `pspline_derpsalsa`, `pspline_lsrpls`, `pspline_brpls`, `pspline_mpls`,
  `corner_cutting`, `irsqr`, `mixture_model`
- Optimizer/meta core: `adaptive_minmax`, `optimize_extended_range`, `custom_bc`,
  `collab_pls`
- Misc core: `interp_pts`, `beads`

## Compatibility Entry Points Needing Fixture Tuning

- Whittaker variants: none currently tracked
- Morphology variants: none currently tracked
- Spline family: none currently tracked
- Classification variants: none currently tracked
- Optimizer/meta methods: none currently tracked
- Misc: none currently tracked

## Future Hardening Work

- Broaden the fixture matrix with more signal shapes and parameter sets.
- Broaden 2D fixture coverage across additional parameter sets and surface
  shapes while keeping the current pinned fixture tolerances below `1e-1`.
- Optimize dense first-pass paths such as BEADS with banded or sparse solvers.
- Keep CubeCL WGPU behind `gpu-wgpu`; current real-device-tested kernels cover
  batched `f32` moving minimum, moving maximum, opening, and the top-hat
  baseline primitive. Further kernels should keep the unsafe launch boundary
  isolated to `src/backend/cubecl_wgpu.rs`.
