import { fit, methods } from "./pkg/node.js";

const result = fit([1.0, 1.1, 4.2, 1.2, 1.0], {
  method: "asls",
  p: 0.05,
});
const minimumLength = fit([3.0, 2.0, 3.0]);

if (result.baseline.length !== 5 || !result.report.converged) {
  throw new Error("Node.js entry point did not initialize or fit correctly");
}
if (
  minimumLength.baseline.length !== 3 ||
  !minimumLength.baseline.every(Number.isFinite)
) {
  throw new Error("Node.js entry point failed for a three-point signal");
}
if (!methods().includes("polynomial")) {
  throw new Error("Node.js entry point did not expose the public API");
}

console.log("Auto-initialized Node.js WASM smoke test: OK");
