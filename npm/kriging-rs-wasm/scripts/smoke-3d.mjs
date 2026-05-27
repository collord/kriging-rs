// Smoke test for the M13 3-D WASM bindings. Talks directly to the
// wasm-bindgen output in pkg/ (no TS wrapper layer required).
//
// Verifies:
//   1. The 3-D OK/SK/UK classes instantiate and predict finite values.
//   2. Anisotropy parameters flow through correctly (different anis ->
//      different prediction).
//   3. The SGS streaming callback fires once per realization in
//      realization-index order with a Float64Array of the expected
//      length.
//
// Run with: node scripts/smoke-3d.mjs

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pkgUrl = resolve(here, "..", "pkg", "kriging_rs.js");
const wasmUrl = resolve(here, "..", "pkg", "kriging_rs_bg.wasm");

const mod = await import(pkgUrl);

// wasm-bindgen "web" target wants explicit init with the wasm bytes.
const wasmBytes = await readFile(wasmUrl);
await mod.default({ module_or_path: wasmBytes });

const {
  WasmOrdinaryKriging3D,
  WasmSimpleKriging3D,
  WasmUniversalKriging3D,
  gaussianSimulation3D,
} = mod;

// 8 corners of a unit cube + center, with a smooth-ish field.
const xs = new Float64Array([0, 10, 0, 10, 0, 10, 0, 10, 5]);
const ys = new Float64Array([0, 0, 10, 10, 0, 0, 10, 10, 5]);
const zs = new Float64Array([0, 0, 0, 0, 10, 10, 10, 10, 5]);
const values = new Float64Array(xs.map((x, i) => 1 + 2 * x + 3 * ys[i] + 4 * zs[i]));

// Identity anisotropy (no rotation, no stretch).
const ANG = [0, 0, 0, 1, 1];

// ---- OK3D ----
const ok = WasmOrdinaryKriging3D.fromArrays(
  xs, ys, zs, values,
  ...ANG,
  "exponential", 0.01, 100.0, 15.0,
  undefined, undefined, undefined,  // shape, max_radius, max_neighbors
);
const okPred = ok.predict(5.0, 5.0, 5.0);
if (!Number.isFinite(okPred.value) || !Number.isFinite(okPred.variance)) {
  throw new Error(`OK3D.predict produced non-finite: ${JSON.stringify(okPred)}`);
}
console.log(
  `OK3D at (5,5,5): value=${okPred.value.toFixed(3)} var=${okPred.variance.toFixed(3)} ` +
    `cond=${okPred.conditionNumber.toExponential(2)}`,
);

const okBatch = ok.predictBatch(
  new Float64Array([5, 2, 8]),
  new Float64Array([5, 3, 7]),
  new Float64Array([5, 4, 6]),
);
if (okBatch.values.length !== 3) {
  throw new Error(`OK3D.predictBatch length mismatch: ${okBatch.values.length}`);
}
console.log(`OK3D batch values: [${[...okBatch.values].map(v => v.toFixed(2)).join(", ")}]`);

// ---- Anisotropy sanity: stretched z (anis2=0.1 i.e. 10x z-stretch) ----
const okAniso = WasmOrdinaryKriging3D.fromArrays(
  xs, ys, zs, values,
  0, 0, 0, 1.0, 0.1,
  "exponential", 0.01, 100.0, 15.0,
  undefined, undefined, undefined,
);
// Target near a face (off the symmetry axes) so anisotropy is detectable.
const ASYM_TARGET = [2.0, 4.0, 1.5];
const okIsoAsym = ok.predict(...ASYM_TARGET);
const okAnisoAsym = okAniso.predict(...ASYM_TARGET);
const anisoDiff = Math.abs(okIsoAsym.value - okAnisoAsym.value);
console.log(
  `OK3D at (${ASYM_TARGET.join(",")}): iso=${okIsoAsym.value.toFixed(3)} ` +
    `anis2=0.1=${okAnisoAsym.value.toFixed(3)} diff=${anisoDiff.toFixed(3)}`,
);
if (anisoDiff < 1e-3) {
  throw new Error("Anisotropy did not meaningfully change the prediction");
}

// ---- SK3D ----
const meanValue = [...values].reduce((a, b) => a + b, 0) / values.length;
const sk = WasmSimpleKriging3D.fromArrays(
  xs, ys, zs, values,
  meanValue,
  ...ANG,
  "exponential", 0.01, 100.0, 15.0,
  undefined,
);
const skPred = sk.predict(5.0, 5.0, 5.0);
if (!Number.isFinite(skPred.value)) {
  throw new Error("SK3D produced non-finite");
}
console.log(`SK3D at (5,5,5): value=${skPred.value.toFixed(3)} var=${skPred.variance.toFixed(3)}`);

// ---- UK3D (linear trend) ----
const uk = WasmUniversalKriging3D.fromArraysLinear(
  xs, ys, zs, values,
  ...ANG,
  "exponential", 0.01, 100.0, 15.0,
  undefined,
);
const ukPred = uk.predict(5.0, 5.0, 5.0);
const expectedTrend = 1 + 2 * 5 + 3 * 5 + 4 * 5;
const ukErr = Math.abs(ukPred.value - expectedTrend);
console.log(`UK3D at (5,5,5): value=${ukPred.value.toFixed(3)} (linear trend expected ${expectedTrend}, err ${ukErr.toFixed(3)})`);
if (ukErr > 1.0) {
  throw new Error(`UK3D recovered trend with error ${ukErr}, expected < 1.0`);
}

// ---- SGS3D streaming ----
const N_REAL = 3;
const NX = 5, NY = 5, NZ = 3;
const seen = [];
gaussianSimulation3D(
  xs, ys, zs, values,
  ...ANG,
  "exponential", 0.01, 100.0, 15.0,
  undefined,
  NX, NY, NZ,
  0.0, 0.0, 0.0,      // origin
  2.5, 2.5, 5.0,      // spacing
  42n,                // seed (BigInt for u64)
  N_REAL,
  false,              // score_space
  (idx, grid) => {
    if (typeof idx !== "number") {
      throw new Error(`SGS callback got non-numeric idx: ${idx}`);
    }
    if (!(grid instanceof Float64Array)) {
      throw new Error(`SGS callback got non-Float64Array grid: ${typeof grid}`);
    }
    if (grid.length !== NX * NY * NZ) {
      throw new Error(`SGS grid length ${grid.length} != ${NX * NY * NZ}`);
    }
    seen.push(idx);
    // Spot-check finiteness.
    const nNaN = [...grid].filter(v => !Number.isFinite(v)).length;
    if (nNaN > 0) {
      console.log(`  realization ${idx}: ${nNaN}/${grid.length} NaN cells (kriging system failures)`);
    }
    return undefined;  // continue
  },
);

if (seen.length !== N_REAL) {
  throw new Error(`SGS expected ${N_REAL} callbacks, got ${seen.length}`);
}
for (let i = 0; i < seen.length; i++) {
  if (seen[i] !== i) {
    throw new Error(`SGS callback fired out of order: expected ${i}, got ${seen[i]}`);
  }
}
console.log(`SGS3D streamed ${N_REAL} realizations of ${NX}x${NY}x${NZ} grid in order`);

console.log("\n3D smoke test passed.");
