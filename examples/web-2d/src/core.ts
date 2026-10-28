// The kriging pipeline, free of DOM code: the Web Worker runs it in the browser and
// scripts/check.ts runs it in Node. Coordinates are projected metres (e.g. UTM).
//
// Library calls used (all from kriging-rs-wasm; `init()` must have resolved first):
//   computeEmpiricalVariogramProjected  - experimental variogram on x/y
//   fitVariogramProjected               - parametric fit; range comes back in metres
//   evaluateNestedVariogram             - the model curve, from the library's own formulas
//   ProjectedKriging + predictGrid      - ordinary kriging on a regular grid
import {
  ProjectedKriging,
  computeEmpiricalVariogramProjected,
  evaluateNestedVariogram,
  fitVariogramProjected,
} from "kriging-rs-wasm";
import type { FittedVariogram, VariogramTypeName } from "kriging-rs-wasm";

export interface Samples {
  xs: Float64Array;
  ys: Float64Array;
  values: Float64Array;
}

export interface Extent {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
}

export interface VariogramResult {
  maxDistance: number;
  empirical: { distances: number[]; semivariances: number[]; counts: number[] };
  model: FittedVariogram;
  curve: { distances: number[]; semivariances: number[] };
  /** Samples the variogram was computed from (subsampled above `maxPairsN`). */
  nUsed: number;
}

export interface GridResult {
  nx: number;
  ny: number;
  extent: Extent;
  /** Row-major, **row 0 = north** (top of an image), column 0 = west. */
  estimate: Float32Array;
  variance: Float32Array;
  ms: number;
}

export function extentOf(s: Samples): Extent {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < s.xs.length; i++) {
    xmin = Math.min(xmin, s.xs[i]);
    xmax = Math.max(xmax, s.xs[i]);
    ymin = Math.min(ymin, s.ys[i]);
    ymax = Math.max(ymax, s.ys[i]);
  }
  return { xmin, xmax, ymin, ymax };
}

/** Pad an extent by a fraction of its span so edge samples aren't on the border. */
export function padExtent(e: Extent, frac = 0.02): Extent {
  const px = (e.xmax - e.xmin) * frac || 1;
  const py = (e.ymax - e.ymin) * frac || 1;
  return { xmin: e.xmin - px, xmax: e.xmax + px, ymin: e.ymin - py, ymax: e.ymax + py };
}

/** Deterministic PRNG so the synthetic dataset is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Synthetic contaminant plume over a 3 km × 2 km site in UTM-scale coordinates. The hot
 * spot is in the north-east corner, so a flipped map is obvious.
 */
export function syntheticField(x: number, y: number): number {
  const u = (x - 500_000) / 3000; // 0..1 west → east
  const v = (y - 4_100_000) / 2000; // 0..1 south → north
  const plume = 40 * Math.exp(-((u - 0.8) ** 2 / 0.02 + (v - 0.8) ** 2 / 0.03));
  const secondary = 12 * Math.exp(-((u - 0.25) ** 2 / 0.01 + (v - 0.3) ** 2 / 0.01));
  return 5 + plume + secondary + 3 * Math.sin(u * 9) * Math.cos(v * 7);
}

export function makeSynthetic(n = 300, seed = 42): Samples {
  const rnd = mulberry32(seed);
  const xs = new Float64Array(n), ys = new Float64Array(n), values = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = 500_000 + rnd() * 3000;
    ys[i] = 4_100_000 + rnd() * 2000;
    values[i] = syntheticField(xs[i], ys[i]) + (rnd() - 0.5) * 2; // ±1 measurement noise
  }
  return { xs, ys, values };
}

/** Parse a CSV whose header contains `x`, `y`, `value` (any order, case-insensitive). */
export function parseCsv(text: string): Samples {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error("CSV needs a header row and at least one data row");
  const sep = lines[0].includes(";") && !lines[0].includes(",") ? ";" : ",";
  const header = lines[0].split(sep).map((h) => h.trim().toLowerCase().replace(/^"|"$/g, ""));
  const [ix, iy, iv] = ["x", "y", "value"].map((name) => header.indexOf(name));
  if (ix < 0 || iy < 0 || iv < 0) {
    throw new Error(`CSV header must contain x, y, value (found: ${header.join(", ")})`);
  }
  const xs: number[] = [], ys: number[] = [], vs: number[] = [];
  for (const line of lines.slice(1)) {
    const cells = line.split(sep);
    const x = Number(cells[ix]), y = Number(cells[iy]), v = Number(cells[iv]);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(v)) {
      xs.push(x);
      ys.push(y);
      vs.push(v);
    }
  }
  if (xs.length < 3) throw new Error("Need at least 3 rows with numeric x, y, value");
  return { xs: Float64Array.from(xs), ys: Float64Array.from(ys), values: Float64Array.from(vs) };
}

/** Random subsample: the empirical variogram is O(n²) in the number of samples. */
function subsample(s: Samples, maxN: number): Samples {
  const n = s.xs.length;
  if (n <= maxN) return s;
  const rnd = mulberry32(7);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < maxN; i++) {
    const j = i + Math.floor(rnd() * (n - i));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const pick = idx.slice(0, maxN);
  return {
    xs: Float64Array.from(pick, (i) => s.xs[i]),
    ys: Float64Array.from(pick, (i) => s.ys[i]),
    values: Float64Array.from(pick, (i) => s.values[i]),
  };
}

/** Experimental variogram, fitted model, and the model curve for plotting. */
export function variogram(
  s: Samples,
  family: VariogramTypeName,
  nBins = 15,
  maxPairsN = 1500,
): VariogramResult {
  const sub = subsample(s, maxPairsN);
  const e = extentOf(sub);
  // Half the site diagonal: beyond that, few pairs and edge effects dominate.
  const maxDistance = Math.hypot(e.xmax - e.xmin, e.ymax - e.ymin) / 2;
  const opts = { xs: sub.xs, ys: sub.ys, values: sub.values, maxDistance, nBins };

  const emp = computeEmpiricalVariogramProjected(opts);
  const model = fitVariogramProjected({ ...opts, variogramType: family });

  // γ(0) = 0 exactly and the nugget is the jump just past it, so start the curve at a
  // tiny positive lag; the nugget then draws as the intercept.
  const n = 120;
  const hs = Float64Array.from({ length: n }, (_, i) => (Math.max(i, 1e-6) / (n - 1)) * maxDistance);
  const curve = evaluateNestedVariogram([model], hs);

  return {
    maxDistance,
    empirical: {
      distances: Array.from(emp.distances),
      semivariances: Array.from(emp.semivariances),
      counts: Array.from(emp.counts),
    },
    model,
    curve: { distances: Array.from(curve.distances), semivariances: Array.from(curve.semivariances) },
    nUsed: sub.xs.length,
  };
}

/**
 * The library's grids are row 0 = **south** (`yMin`), like a map's y axis. Images and
 * canvases are row 0 = **top**. Flip once, here, so everything downstream is north-up.
 */
export function toNorthUp(flat: Float64Array, nx: number, ny: number): Float32Array {
  const out = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    out.set(flat.subarray(j * nx, (j + 1) * nx), (ny - 1 - j) * nx);
  }
  return out;
}

/**
 * Ordinary kriging onto an `nx × ny` grid of cell centres covering `extent`.
 *
 * The search neighborhood keeps this fast for thousands of samples: without it every
 * model factorizes the full n × n system. Coordinates can be raw UTM; the package
 * centres them before its f32 core sees them.
 */
export function krigeGrid(
  s: Samples,
  model: FittedVariogram,
  nx: number,
  ny: number,
  extent: Extent,
  maxNeighbors = 32,
): GridResult {
  const t0 = performance.now();
  const kriging = ProjectedKriging.fromFitted({
    xs: s.xs,
    ys: s.ys,
    values: s.values,
    fittedVariogram: model,
    neighborhood: { maxNeighbors: Math.min(maxNeighbors, s.xs.length) },
  });
  try {
    const grid = kriging.predictGrid({
      xMin: extent.xmin,
      xMax: extent.xmax,
      yMin: extent.ymin,
      yMax: extent.ymax,
      xCells: nx,
      yCells: ny,
    });
    return {
      nx,
      ny,
      extent,
      estimate: toNorthUp(grid.valuesFlat, nx, ny),
      variance: toNorthUp(grid.variancesFlat, nx, ny),
      ms: performance.now() - t0,
    };
  } finally {
    // WASM memory is not garbage-collected; free models you are done with.
    kriging.free();
  }
}
