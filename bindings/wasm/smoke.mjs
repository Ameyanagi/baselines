import { readFile } from "node:fs/promises";

import init, {
  baseline,
  baseline2d,
  baselineWith,
  fit,
  methods,
  methods2d,
} from "./pkg/index.js";

const wasm = await readFile(
  new URL("./pkg/baselines_rs_bg.wasm", import.meta.url),
);
await init({ module_or_path: wasm });

const y = [1.0, 1.1, 4.2, 1.2, 1.0];
const estimated = baseline(y, {
  method: "asls",
  lam: 1.0e4,
  p: 0.05,
  maxIter: 10,
});
const result = fit(y, { method: "arpls", tol: 1.0e-4 });
const legacy = baselineWith(y, "arpls");

if (
  estimated.length !== y.length ||
  result.corrected.length !== y.length ||
  legacy.length !== y.length
) {
  throw new Error("WASM binding returned an unexpected output shape");
}
if (![...estimated, ...result.corrected, ...legacy].every(Number.isFinite)) {
  throw new Error("WASM binding returned a non-finite value");
}
if (result.report.iterations < 1) {
  throw new Error("WASM binding did not return convergence metadata");
}
if (!methods().includes("asls") || !methods2d().includes("rolling_ball")) {
  throw new Error("WASM binding did not expose array-valued method lists");
}

const surface = [
  [1, 1, 1],
  [1, 4, 1],
];
const surfaceBaseline = baseline2d(surface, {
  method: "rolling_ball",
  windowRows: 3,
  windowCols: 3,
});
if (surfaceBaseline.length !== 6) {
  throw new Error("WASM binding did not infer a nested matrix shape");
}

let rejected = false;
try {
  baseline(y, { method: "polynomial", lambda: 1.0e4 });
} catch (error) {
  rejected = String(error).includes("not supported");
}
if (!rejected) {
  throw new Error("WASM binding silently accepted an unsupported option");
}

console.log("WASM runtime smoke test: OK");
