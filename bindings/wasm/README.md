# baselines-wasm

<p align="center">
  <img src="https://raw.githubusercontent.com/Ameyanagi/baselines/main/docs/assets/branding/baselines-icon-midnight-hex-v2.png" alt="baselines logo" width="220">
</p>

WebAssembly bindings for the Rust `baselines` crate.

```js
import init, { baselineWith } from "./pkg/baselines_wasm.js";

await init();
const y = Float64Array.from([1.0, 1.1, 4.2, 1.2, 1.0]);
const estimated = baselineWith(y, "arpls");
```

`baseline` and `correct` use AsLS defaults. The `*With` variants select a
method, while `baseline2d` and `correct2d` accept flat row-major arrays and an
explicit shape.
