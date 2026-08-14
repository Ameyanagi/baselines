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

const { baseline, methods } = await import("./pkg/auto.js");
const browser = await import("./pkg/browser.js");

const y = [1.0, 1.1, 4.2, 1.2, 1.0];
const estimated = baseline(y, { method: "arpls" });
if (estimated.length !== y.length || ![...estimated].every(Number.isFinite)) {
  throw new Error("Auto-initialized WASM binding returned an invalid baseline");
}
if (!methods().includes("asls") || !browser.methods().includes("arpls")) {
  throw new Error("Browser entry point did not expose the public API");
}

console.log("Auto-initialized browser WASM smoke test: OK");
