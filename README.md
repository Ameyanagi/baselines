# baselines

`baselines` is a Rust crate for baseline correction of one-dimensional signals
and spectra. It is an independent Rust implementation inspired by the baseline
correction literature and by the public behavior of
[`pybaselines`](https://pybaselines.readthedocs.io/).

```rust
use baselines::{asls, AslsParams};

let y = vec![1.0, 1.1, 4.2, 1.2, 1.0];
let fit = asls(&y, AslsParams::default())?;
let corrected = fit.corrected(&y);
# Ok::<(), baselines::BaselineError>(())
```

## Scope

The crate starts with CPU `f64` implementations and public entry points for the
current one-dimensional `pybaselines.Baseline` algorithm families: polynomial,
Whittaker, morphology, penalized spline, smoothing, classification, optimizer,
and miscellaneous methods.

Some algorithms currently share conservative Rust-native engines while golden
fixtures are added. GPU support is intentionally feature-gated and experimental
while CPU behavior is validated.

## Attribution

This project does not copy implementation code from `pybaselines`. The Python
project is used as a documentation and behavioral reference, and golden
fixtures should record the pybaselines version that generated them.

Please cite the original algorithm papers as appropriate. See `NOTICE.md` and
`CITATION.cff` for project-level attribution.
