import { copyFile, readFile, writeFile } from "node:fs/promises";

const root = new URL("./", import.meta.url);
const output = new URL("./pkg/", root);
const sourcePackagePath = new URL("package.json", root);
const outputPackagePath = new URL("package.json", output);

const sourcePackage = JSON.parse(await readFile(sourcePackagePath, "utf8"));
const outputPackage = JSON.parse(await readFile(outputPackagePath, "utf8"));

if (
  outputPackage.name !== sourcePackage.name ||
  outputPackage.version !== sourcePackage.version
) {
  throw new Error(
    `wasm-pack generated ${outputPackage.name}@${outputPackage.version}, expected ${sourcePackage.name}@${sourcePackage.version}`,
  );
}

await Promise.all(
  [
    "index.js",
    "index.d.ts",
    "auto.js",
    "auto.d.ts",
    "browser.js",
    "browser.d.ts",
    "node.js",
    "node.d.ts",
  ].map((filename) =>
    copyFile(new URL(filename, root), new URL(filename, output)),
  ),
);

outputPackage.description = sourcePackage.description;
outputPackage.repository = sourcePackage.repository;
outputPackage.homepage = sourcePackage.homepage;
outputPackage.bugs = sourcePackage.bugs;
outputPackage.keywords = sourcePackage.keywords;
outputPackage.files = [
  ...outputPackage.files,
  "index.js",
  "index.d.ts",
  "auto.js",
  "auto.d.ts",
  "browser.js",
  "browser.d.ts",
  "node.js",
  "node.d.ts",
];
outputPackage.main = "index.js";
outputPackage.module = "index.js";
outputPackage.types = "index.d.ts";
outputPackage.sideEffects = [
  ...(outputPackage.sideEffects ?? []),
  "./auto.js",
  "./browser.js",
  "./node.js",
];
outputPackage.exports = {
  ".": {
    types: "./index.d.ts",
    import: "./index.js",
    default: "./index.js",
  },
  "./auto": {
    types: "./auto.d.ts",
    import: "./auto.js",
    default: "./auto.js",
  },
  "./browser": {
    types: "./browser.d.ts",
    import: "./browser.js",
    default: "./browser.js",
  },
  "./node": {
    types: "./node.d.ts",
    import: "./node.js",
    default: "./node.js",
  },
};

await writeFile(
  outputPackagePath,
  `${JSON.stringify(outputPackage, null, 2)}\n`,
);
