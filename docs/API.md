# Rust API

The recommended API is method-chain based. It is a thin ergonomic layer over
the family modules, so defaults, validation, references, and numerical behavior
are the same as the underlying functions.

## 1D signals

```rust
use baselines::prelude::*;

let fit = Baseline::new(&y)
    .asls()
    .lambda(1.0e6)
    .p(0.01)
    .max_iter(50)
    .tol(1.0e-3)
    .fit()?;

let corrected = fit.corrected(&y)?;
# Ok::<(), baselines::BaselineError>(())
```

## 2D row-major data

```rust
use baselines::prelude::*;

let fit = Baseline2D::row_major(&data, rows, cols)?
    .asls()
    .lambda(8.0e3)
    .p(0.01)
    .cg_tol(1.0e-6)
    .fit()?;

let corrected = fit.corrected(&data)?;
# Ok::<(), baselines::BaselineError>(())
```

## Full parameter structs

Use `with_params` when params are easier to construct directly.

```rust
use baselines::prelude::*;
use baselines::whittaker::{AslsParams, WhittakerParams};

let params = AslsParams {
    whittaker: WhittakerParams {
        lambda: 1.0e6,
        max_iter: 50,
        tol: 1.0e-3,
    },
    p: 0.01,
};

let fit = Baseline::new(&y).asls().with_params(params).fit()?;
# Ok::<(), baselines::BaselineError>(())
```

## Low-level API

The family modules remain public and are still useful for advanced workflows,
including reusable workspaces and exact parameter struct calls.

```rust
use baselines::whittaker::{AslsParams, WhittakerWorkspace, asls_into};

let mut baseline = vec![0.0; y.len()];
let mut workspace = WhittakerWorkspace::new(y.len());
let report = asls_into(
    &y,
    AslsParams::default(),
    &mut baseline,
    &mut workspace,
)?;
# Ok::<(), baselines::BaselineError>(())
```

Algorithm references are documented on the underlying family functions. The
builder docs intentionally point back to those functions instead of duplicating
citations.
