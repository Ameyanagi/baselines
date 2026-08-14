import { readFile } from "node:fs/promises";

import init, {
  availableMethods,
  baselineWith,
  correctWith,
} from "./pkg/baselines_rs.js";

const wasm = await readFile(
  new URL("./pkg/baselines_rs_bg.wasm", import.meta.url),
);
await init({ module_or_path: wasm });

const y = Float64Array.from([1.0, 1.1, 4.2, 1.2, 1.0]);
const estimated = baselineWith(y, "arpls");
const corrected = correctWith(y, "arpls");

if (estimated.length !== y.length || corrected.length !== y.length) {
  throw new Error("WASM binding returned an unexpected output shape");
}
if (![...estimated, ...corrected].every(Number.isFinite)) {
  throw new Error("WASM binding returned a non-finite value");
}
if (!availableMethods().split(",").includes("asls")) {
  throw new Error("WASM binding did not expose the expected methods");
}

console.log("WASM runtime smoke test: OK");
