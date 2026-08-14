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
  ["auto.js", "auto.d.ts"].map((filename) =>
    copyFile(new URL(filename, root), new URL(filename, output)),
  ),
);

outputPackage.description = sourcePackage.description;
outputPackage.repository = sourcePackage.repository;
outputPackage.homepage = sourcePackage.homepage;
outputPackage.bugs = sourcePackage.bugs;
outputPackage.keywords = sourcePackage.keywords;
outputPackage.files = [...outputPackage.files, "auto.js", "auto.d.ts"];
outputPackage.exports = {
  ".": {
    types: "./baselines_rs.d.ts",
    import: "./baselines_rs.js",
    default: "./baselines_rs.js",
  },
  "./auto": {
    types: "./auto.d.ts",
    import: "./auto.js",
    default: "./auto.js",
  },
};

await writeFile(
  outputPackagePath,
  `${JSON.stringify(outputPackage, null, 2)}\n`,
);
