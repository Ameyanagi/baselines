# Implementation Status

This crate exposes public entry points for the current one-dimensional
`pybaselines.Baseline` algorithm set. The implementation is intentionally
staged: APIs and safety/quality gates are in place first, then behavior parity
is tightened with generated pybaselines fixtures.

## Dedicated First-Pass Implementations

- Whittaker core: `asls`, `airpls`, `arpls`, `psalsa`
- Polynomial core: `poly`, `modpoly`, `imodpoly`, `loess`, `quant_reg`
- Morphology core: `rolling_ball`, `mwmv`, `tophat`, `mor`, `imor`, `amormol`
- Smoothing core: `noise_median`, `snip`, `swima`, `ipsa`, `ria`, `peak_filling`
- Classification core: `rubberband`
- Misc core: `interp_pts`

## Compatibility Entry Points Needing Fixture Tuning

- Whittaker variants: `iasls`, `drpls`, `iarpls`, `aspls`, `derpsalsa`,
  `brpls`, `lsrpls`
- Morphology variants: `mpls`, `mormol`, `mpspline`, `jbcd`
- Spline family: `mixture_model`, `irsqr`, `corner_cutting`, all `pspline_*`
  methods
- Classification variants: `dietrich`, `golotvin`, `std_distribution`,
  `fastchrom`, `cwt_br`, `fabc`
- Optimizer/meta methods: `collab_pls`, `optimize_extended_range`,
  `adaptive_minmax`, `custom_bc`
- Misc: `beads`
- Polynomial variants: `penalized_poly`, `goldindec`

## Next Compatibility Work

- Install a pinned pybaselines version in an isolated Python environment and run
  `scripts/generate_pybaselines_fixtures.py`.
- Add Rust golden tests that compare dedicated implementations against the
  generated fixtures with algorithm-specific tolerances.
- Replace compatibility wrappers with dedicated implementations family by
  family, preserving the public API and passing fixture tests.
- Keep CubeCL WGPU behind `gpu-wgpu`; current real-device-tested kernels cover
  batched `f32` moving minimum, moving maximum, opening, and the top-hat
  baseline primitive. Further kernels should keep the unsafe launch boundary
  isolated to `src/backend/cubecl_wgpu.rs`.
