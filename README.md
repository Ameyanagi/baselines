# baselines

`baselines` is a Rust crate for baseline correction of signals, spectra, and
row-major two-dimensional surfaces. It is an independent Rust implementation
inspired by the baseline correction literature and by the public behavior of
[`pybaselines`](https://pybaselines.readthedocs.io/).

```rust
use baselines::whittaker::{AslsParams, asls};

let y = vec![1.0, 1.1, 4.2, 1.2, 1.0];
let fit = asls(&y, AslsParams::default())?;
let corrected = fit.corrected(&y)?;
# Ok::<(), baselines::BaselineError>(())
```

## Scope

The crate starts with CPU `f64` implementations and public entry points for
the current one-dimensional `pybaselines.Baseline` algorithm families:
polynomial, Whittaker, morphology, penalized spline, smoothing,
classification, optimizer, and miscellaneous methods. Two-dimensional support
is staged under `baselines::two_d`, with morphology/smoothing, polynomial, and
Whittaker methods available first.

Algorithms are organized by family module. Core data types such as `Fit1D`,
`Fit2D`, and row-major matrix views are available at the crate root.

Golden fixtures generated from a pinned `pybaselines` release check the
one-dimensional algorithms with algorithm-specific tolerances. GPU support is
feature-gated behind `gpu-wgpu`; the experimental WGPU path provides batched
`f32` morphology kernels for moving minimum, moving maximum, opening, and the
top-hat baseline primitive.

See `docs/PARITY.md` for the current one-dimensional pybaselines parity matrix
and known limits.

## Attribution

This project does not copy implementation code from `pybaselines`. The Python
project is used as a documentation and behavioral reference, and golden
fixtures should record the pybaselines version that generated them.

Please cite the original algorithm papers as appropriate. See `NOTICE.md` and
`CITATION.cff` for project-level attribution.
