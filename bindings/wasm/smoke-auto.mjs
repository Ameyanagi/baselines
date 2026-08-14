import { readFile } from "node:fs/promises";

const nativeFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = input instanceof Request ? new URL(input.url) : new URL(input);
  if (url.protocol === "file:") {
    return new Response(await readFile(url), {
      headers: { "Content-Type": "application/wasm" },
    });
  }
  return nativeFetch(input, init);
};

const { availableMethods, baselineWith, correctWith } = await import(
  "./pkg/auto.js"
);

const y = Float64Array.from([1.0, 1.1, 4.2, 1.2, 1.0]);
const estimated = baselineWith(y, "arpls");
const corrected = correctWith(y, "arpls");

if (estimated.length !== y.length || corrected.length !== y.length) {
  throw new Error("Auto-initialized WASM binding returned an unexpected shape");
}
if (![...estimated, ...corrected].every(Number.isFinite)) {
  throw new Error("Auto-initialized WASM binding returned a non-finite value");
}
if (!availableMethods().split(",").includes("asls")) {
  throw new Error("Auto-initialized WASM binding is missing expected methods");
}

console.log("Auto-initialized WASM runtime smoke test: OK");
