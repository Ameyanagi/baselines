import { readFile } from "node:fs/promises";

import init from "./index.js";

const module_or_path = await readFile(
  new URL("./baselines_rs_bg.wasm", import.meta.url),
);
await init({ module_or_path });

export * from "./index.js";
