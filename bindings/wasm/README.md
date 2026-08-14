# baselines-rs

<p align="center">
  <img src="https://raw.githubusercontent.com/Ameyanagi/baselines/main/docs/assets/branding/baselines-icon-midnight-hex-v2.png" alt="baselines logo" width="220">
</p>

WebAssembly bindings for the Rust `baselines` crate. Regular arrays and typed
arrays are accepted and converted to `Float64Array` values.

Browser applications can use the auto-initializing entry point:

```js
import { baseline, fit, methods } from "baselines-rs/browser";

const y = [1.0, 1.1, 4.2, 1.2, 1.0];
const estimated = baseline(y, {
  method: "asls",
  lambda: 1e5,
  p: 0.05,
});
const result = fit(y, { method: "arpls", tol: 1e-4 });
console.log(result.baseline, result.corrected, result.report);
console.log(methods());
```

Node.js has a separate auto-initializing entry point that loads the WASM file
from disk:

```js
import { correct } from "baselines-rs/node";

const corrected = correct(y, { method: "rolling_ball", windowSize: 31 });
```

Nested matrices are accepted by the 2D API and flattened internally:

```js
import { baseline2d } from "baselines-rs/browser";

const estimated = baseline2d(
  [
    [1, 1, 1],
    [1, 4, 1],
  ],
  { method: "rolling_ball", windowRows: 3, windowCols: 3 },
);
```

To control how the binary is loaded, use the explicit root entry point:

```js
import init, { baseline } from "baselines-rs";

await init();
const estimated = baseline(y, { method: "arpls" });
```

`baselineWith` and `correctWith`, plus the positional 2D signatures, remain as
backward-compatible aliases. `baselines-rs/auto` remains an alias for the
browser entry point.

Build the publishable package with `npm run build` from this directory. The
generated `pkg/` directory is the npm package; the surrounding workspace is
private to prevent accidentally publishing the wrapper instead. Automatic
entry points require a runtime that supports top-level `await` in ES modules.
