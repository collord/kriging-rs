// Headless check of src/core.ts, the same code the worker runs: `npm run check`.
// Node has no fetch for file URLs, so init() gets the .wasm bytes from disk via the
// package's `kriging-rs-wasm/wasm` export.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import init from "kriging-rs-wasm";
import {
  extentOf,
  krigeGrid,
  makeSynthetic,
  padExtent,
  syntheticField,
  variogram,
} from "../src/core.ts";

await init(await readFile(fileURLToPath(import.meta.resolve("kriging-rs-wasm/wasm"))));

const failures: string[] = [];
const expect = (ok: boolean, what: string) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${what}`);
  if (!ok) failures.push(what);
};

for (const n of [300, 3000]) {
  const samples = makeSynthetic(n, 42);
  const vg = variogram(samples, "spherical", 15);
  const m = vg.model;
  console.log(
    `\nn=${n}: ${m.variogramType} nugget=${m.nugget.toFixed(3)} sill=${m.sill.toFixed(2)} ` +
      `range=${m.range.toFixed(0)} m (from ${vg.nUsed} samples)`,
  );
  expect(m.range > 100 && m.range < 5000, "fitted range is in metres for a 3 km site");

  const nx = 120, ny = 80;
  const extent = padExtent(extentOf(samples));
  const grid = krigeGrid(samples, m, nx, ny, extent, 32);
  console.log(`kriged ${nx}x${ny} in ${grid.ms.toFixed(0)} ms`);

  expect(grid.estimate.every(Number.isFinite), "every estimate is finite");
  expect(grid.variance.every((v) => Number.isFinite(v) && v >= 0), "every variance is finite and >= 0");

  // Row 0 is north: the plume's peak sits in the north-east of the site.
  let best = -Infinity, bestK = 0;
  grid.estimate.forEach((v, k) => {
    if (v > best) [best, bestK] = [v, k];
  });
  const row = Math.floor(bestK / nx), col = bestK % nx;
  expect(row < ny / 2 && col > nx / 2, `maximum (row ${row}, col ${col}) is in the north-east quadrant`);

  // Accuracy against the noise-free field, at cell centres.
  const dx = (extent.xmax - extent.xmin) / nx, dy = (extent.ymax - extent.ymin) / ny;
  let se = 0;
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      const x = extent.xmin + (c + 0.5) * dx;
      const y = extent.ymax - (r + 0.5) * dy; // north-up
      se += (grid.estimate[r * nx + c] - syntheticField(x, y)) ** 2;
    }
  }
  const rmse = Math.sqrt(se / (nx * ny));
  console.log(`RMSE vs true field: ${rmse.toFixed(2)}`);
  expect(rmse < (n >= 3000 ? 1.5 : 4), "RMSE against the true field is small");
}

if (failures.length) {
  console.error(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nAll checks passed.");
