# Changelog

All notable changes to this project are documented here.

## [0.2.0] - 2026-08-15

### Added

- Validated configurable options for the simple Rust, Python, and WebAssembly
  APIs, covering Whittaker, morphology, and polynomial parameters.
- `fit` and `fit_2d` binding functions that return baselines, corrected data,
  and convergence reports.
- Python array-like coercion, type stubs, and a `py.typed` marker.
- Dedicated auto-initializing npm entry points for browsers and Node.js.
- Nested JavaScript matrix inputs and typed npm package exports.

### Changed

- The npm API now accepts `baseline(data, options)` and `correct(data, options)`
  while retaining the `*With` compatibility aliases.
- npm method discovery now returns arrays instead of comma-separated strings.

### Fixed

- Three-point Whittaker inputs no longer panic while constructing the
  second-difference penalty.

## [0.1.2] - 2026-08-15

### Added

- An auto-initializing `baselines-rs/auto` npm entry point for modern browsers,
  while retaining the explicit initializer at `baselines-rs`.
- Typed npm package exports and installed-package smoke coverage for both WASM
  initialization modes.

## [0.1.1] - 2026-08-14

### Added

- A small default-based API with `baseline`, `correct`, and matching 2D helpers.
- Stable `Method` and `Method2D` names shared by Rust, Python, and WebAssembly.
- ABI3 Python bindings for NumPy arrays under `bindings/python`.
- WebAssembly bindings under `bindings/wasm`.
- X-aware Whittaker fitting with range and boolean masks.
- Three generated icon concepts under `docs/assets/branding`.

### Fixed

- Reject oversized polynomial orders without integer-overflow panics.
- Reject zero `max_iter` for two-dimensional IMor, consistently with other
  iterative algorithms.

### Changed

- Reduced the crates.io package by excluding repository-only tests, fixtures,
  benchmark records, scripts, bindings, and generated image assets.
- The `rayon` feature now parallelizes batched CPU SNIP correction instead of
  only enabling an unused dependency.

## [0.1.0]

- Initial crates.io release with one- and two-dimensional baseline correction
  families, method-chain APIs, pybaselines fixtures, and experimental WGPU
  morphology kernels.
