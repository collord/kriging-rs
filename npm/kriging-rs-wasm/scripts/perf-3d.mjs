// Single-realization WASM SGS perf measurement. Lets us turn the M11
// "native * 3x slowdown" projection into a real number on the same
// hardware that produced the native benchmark.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pkgUrl = resolve(here, "..", "pkg", "kriging_rs.js");
const wasmUrl = resolve(here, "..", "pkg", "kriging_rs_bg.wasm");

const mod = await import(pkgUrl);
const wasmBytes = await readFile(wasmUrl);
await mod.default({ module_or_path: wasmBytes });

const { gaussianSimulation3D } = mod;

// Load the M5 dataset to keep parity with the native bench.
const samplesCsv = await readFile(
  resolve(here, "..", "..", "..", "tests", "parity_3d", "fixtures", "skgstat_3d", "samples.csv"),
  "utf-8",
);
const sampleLines = samplesCsv.split("\n").slice(1).filter(l => l.length > 0);
const xs = new Float64Array(sampleLines.map(l => Number(l.split(",")[0])));
const ys = new Float64Array(sampleLines.map(l => Number(l.split(",")[1])));
const zs = new Float64Array(sampleLines.map(l => Number(l.split(",")[2])));
const values = new Float64Array(sampleLines.map(l => Number(l.split(",")[3])));
console.log(`Loaded ${xs.length} samples`);

const variogramArgs = ["exponential", 0.05, 1.0, 30.0, undefined];

function makeGridArgs(n) {
  const nz = Math.max(1, Math.round(n / 2));
  return [
    n, n, nz,
    0.0, 0.0, 0.0,
    100 / (n - 1), 100 / (n - 1), 50 / Math.max(1, nz - 1),
  ];
}

function timeOne(n) {
  const t0 = process.hrtime.bigint();
  gaussianSimulation3D(
    xs, ys, zs, values,
    0, 0, 0, 1, 1,
    ...variogramArgs,
    ...makeGridArgs(n),
    1234n,
    1,
    false,
    (_idx, _grid) => undefined,
  );
  const t1 = process.hrtime.bigint();
  return Number(t1 - t0) / 1e6; // milliseconds
}

const sizes = [10, 20, 30, 40, 50];
console.log("\nWASM SGS single-realization wall time:\n");
console.log("| n   | n_cells | wall ms | per-cell us |");
console.log("|-----|---------|---------|-------------|");
for (const n of sizes) {
  // Warm up.
  timeOne(n);
  // Measure.
  let best = Infinity;
  for (let i = 0; i < 3; i++) {
    const t = timeOne(n);
    if (t < best) best = t;
  }
  const nCells = n * n * Math.max(1, Math.round(n / 2));
  const perCell = (best * 1000) / nCells;
  console.log(
    `| ${String(n).padStart(3)} | ${String(nCells).padStart(7)} | ${best.toFixed(0).padStart(7)} | ${perCell.toFixed(2).padStart(11)} |`,
  );
}
