# Changelog

All notable changes to this project are documented here.

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
