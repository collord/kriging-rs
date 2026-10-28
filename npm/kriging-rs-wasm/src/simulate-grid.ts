/**
 * Grid-shaped binomial conditional simulation helpers.
 *
 * The {@link conditionalSimulateBinomial} API operates on flat target lists; for
 * disease prevalence mapping callers almost always want results shaped as a
 * `[yCells][xCells]` raster matching {@link BinomialGridOutput}. This module
 * provides three layered helpers built on top of the per-realization and
 * many-realization binomial samplers:
 *
 * - {@link simulateBinomialGrid} — one realization, returned as nested 2-D
 *   arrays `[yCells][xCells]` for both logits and prevalences.
 * - {@link simulateBinomialGridEnsemble} — `nRealizations` realizations as
 *   row-major flat buffers (the same shape as
 *   {@link BinomialSimulationManyResult}) so they can be passed directly into
 *   the `ensemble*` aggregators without re-flattening.
 * - {@link simulateBinomialGridSummary} — one-shot ensemble + reduction; returns
 *   per-cell `[yCells][xCells]` mean / variance / quantile / exceedance maps.
 *
 * Cell layout: cell `(j, i)` has center `lat = south + (j + 0.5) * dy`,
 * `lon = west + (i + 0.5) * dx`, where `dy = (north - south) / yCells` and
 * `dx = (east - west) / xCells`. Row index `j` runs along latitude
 * (south → north) and column index `i` along longitude (west → east). This
 * matches the layout used everywhere else in the package
 * ({@link BinomialGridOutput}, {@link InterpolateBinomialToGridResult}).
 *
 * @module
 */

import { KrigingError } from "./errors.js";
import {
  conditionalSimulateBinomial,
  conditionalSimulateManyBinomial,
} from "./simulation.js";
import type {
  BinomialGridEnsemble,
  BinomialGridSimulation,
  BinomialGridSummary,
  GeoGridBounds,
  SimulateBinomialGridEnsembleOptions,
  SimulateBinomialGridOptions,
  SimulateBinomialGridSummaryOptions,
} from "./types.js";
import {
  ensembleExceedanceProbability,
  ensembleMean,
  ensembleQuantiles,
  ensembleVariance,
} from "./aggregate.js";

function requirePositiveIntCells(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new KrigingError(`${label} must be a positive integer`, {
      code: "invalid_input",
    });
  }
  return value;
}

function validateBounds(bounds: GeoGridBounds): {
  xCells: number;
  yCells: number;
  dx: number;
  dy: number;
} {
  const xCells = requirePositiveIntCells(bounds.xCells, "xCells");
  const yCells = requirePositiveIntCells(bounds.yCells, "yCells");
  if (!(bounds.east > bounds.west)) {
    throw new KrigingError(
      `east (${bounds.east}) must be > west (${bounds.west})`,
      { code: "invalid_input" }
    );
  }
  if (!(bounds.north > bounds.south)) {
    throw new KrigingError(
      `north (${bounds.north}) must be > south (${bounds.south})`,
      { code: "invalid_input" }
    );
  }
  return {
    xCells,
    yCells,
    dx: (bounds.east - bounds.west) / xCells,
    dy: (bounds.north - bounds.south) / yCells,
  };
}

/**
 * Build flat lat/lon arrays for the centers of every cell in `bounds`, in
 * row-major `(j, i)` order with `i` (longitude) fastest. Length is
 * `yCells * xCells`.
 *
 * Exported for callers that want to drive the underlying flat samplers
 * directly while still using the standard cell layout.
 */
export function gridCellCenters(bounds: GeoGridBounds): {
  lats: Float64Array;
  lons: Float64Array;
} {
  const { xCells, yCells, dx, dy } = validateBounds(bounds);
  const n = xCells * yCells;
  const lats = new Float64Array(n);
  const lons = new Float64Array(n);
  for (let j = 0; j < yCells; j++) {
    const lat = bounds.south + (j + 0.5) * dy;
    const rowOff = j * xCells;
    for (let i = 0; i < xCells; i++) {
      lats[rowOff + i] = lat;
      lons[rowOff + i] = bounds.west + (i + 0.5) * dx;
    }
  }
  return { lats, lons };
}

/**
 * Reshape one realization (`nTargets`-length view of a flat row-major
 * ensemble buffer) into nested `[yCells][xCells]` arrays. This is the inverse
 * of the gridCellCenters layout.
 */
export function reshapeGridRow(
  flat: ArrayLike<number>,
  xCells: number,
  yCells: number
): number[][] {
  if (flat.length !== xCells * yCells) {
    throw new KrigingError(
      `flat length (${flat.length}) must equal xCells * yCells (${xCells * yCells})`,
      { code: "mismatched_arrays" }
    );
  }
  const out: number[][] = new Array(yCells);
  for (let j = 0; j < yCells; j++) {
    const row = new Array<number>(xCells);
    const off = j * xCells;
    for (let i = 0; i < xCells; i++) row[i] = flat[off + i];
    out[j] = row;
  }
  return out;
}

/**
 * Draw a single binomial SGS realization over the requested grid and reshape it
 * to nested `[yCells][xCells]` arrays. Deterministic for a given `seed`.
 */
export function simulateBinomialGrid(
  options: SimulateBinomialGridOptions
): BinomialGridSimulation {
  const { xCells, yCells } = validateBounds(options);
  const { lats: targetLats, lons: targetLons } = gridCellCenters(options);
  const result = conditionalSimulateBinomial({
    conditioningLats: options.lats,
    conditioningLons: options.lons,
    successes: options.successes,
    trials: options.trials,
    targetLats,
    targetLons,
    variogram: options.variogram,
    prior: options.prior,
    seed: options.seed,
  });
  return {
    logitSamples: reshapeGridRow(result.logitSamples, xCells, yCells),
    prevalences: reshapeGridRow(result.prevalenceSamples, xCells, yCells),
  };
}

/**
 * Draw `nRealizations` independent binomial SGS realizations over the requested
 * grid. Returns flat row-major ensemble buffers of length
 * `nRealizations * (xCells * yCells)` so the result can be fed directly into
 * `ensembleMean` / `ensembleVariance` / `ensembleQuantiles` /
 * `ensembleExceedanceProbability` without re-flattening.
 *
 * The k-th realization (cells `[k * nTargets .. (k + 1) * nTargets]`) is
 * bit-identical to {@link simulateBinomialGrid} called with `seed = baseSeed + k`.
 */
export function simulateBinomialGridEnsemble(
  options: SimulateBinomialGridEnsembleOptions
): BinomialGridEnsemble {
  const { xCells, yCells } = validateBounds(options);
  const { lats: targetLats, lons: targetLons } = gridCellCenters(options);
  const result = conditionalSimulateManyBinomial({
    conditioningLats: options.lats,
    conditioningLons: options.lons,
    successes: options.successes,
    trials: options.trials,
    targetLats,
    targetLons,
    variogram: options.variogram,
    prior: options.prior,
    nRealizations: options.nRealizations,
    baseSeed: options.baseSeed,
  });
  return {
    west: options.west,
    south: options.south,
    east: options.east,
    north: options.north,
    xCells,
    yCells,
    nRealizations: result.nRealizations,
    nTargets: result.nTargets,
    logitSamples: result.logitSamples,
    prevalenceSamples: result.prevalenceSamples,
  };
}

function reshapeQuantileBuffer(
  flat: Float64Array,
  nq: number,
  nTargets: number,
  xCells: number,
  yCells: number,
  probabilities: ReadonlyArray<number>
): { probability: number; values: number[][] }[] {
  const out: { probability: number; values: number[][] }[] = new Array(nq);
  for (let q = 0; q < nq; q++) {
    const slice = flat.subarray(q * nTargets, (q + 1) * nTargets);
    out[q] = {
      probability: probabilities[q],
      values: reshapeGridRow(slice, xCells, yCells),
    };
  }
  return out;
}

/**
 * One-shot grid simulation + reduction. Draws an ensemble and computes per-cell
 * point estimates (mean and variance on the logit scale, mean on the prevalence
 * scale) plus optional quantile and exceedance maps on the chosen
 * `summarizeOn` scale (`"prevalence"` by default).
 *
 * For very large grids or large ensembles, prefer
 * {@link simulateBinomialGridEnsemble} to keep the underlying flat buffers in
 * memory only as long as needed.
 */
export function simulateBinomialGridSummary(
  options: SimulateBinomialGridSummaryOptions
): BinomialGridSummary {
  const ensemble = simulateBinomialGridEnsemble(options);
  const {
    xCells,
    yCells,
    nRealizations,
    nTargets,
    logitSamples,
    prevalenceSamples,
  } = ensemble;
  const summarizeOn = options.summarizeOn ?? "prevalence";
  const quantiles = options.quantiles ?? [];
  const exceedanceThresholds = options.exceedanceThresholds ?? [];

  const meanLogitFlat = ensembleMean(logitSamples, nRealizations, nTargets);
  const varianceLogitFlat =
    nRealizations >= 2
      ? ensembleVariance(logitSamples, nRealizations, nTargets)
      : new Float64Array(nTargets); // 0s when only one realization
  const meanPrevalenceFlat = ensembleMean(
    prevalenceSamples,
    nRealizations,
    nTargets
  );

  const sourceFlat = summarizeOn === "logit" ? logitSamples : prevalenceSamples;
  const quantileMaps =
    quantiles.length > 0
      ? reshapeQuantileBuffer(
          ensembleQuantiles(sourceFlat, nRealizations, nTargets, quantiles),
          quantiles.length,
          nTargets,
          xCells,
          yCells,
          quantiles
        )
      : [];

  const exceedanceMaps: { threshold: number; values: number[][] }[] = [];
  for (const t of exceedanceThresholds) {
    const flat = ensembleExceedanceProbability(
      sourceFlat,
      nRealizations,
      nTargets,
      t
    );
    exceedanceMaps.push({
      threshold: t,
      values: reshapeGridRow(flat, xCells, yCells),
    });
  }

  return {
    west: ensemble.west,
    south: ensemble.south,
    east: ensemble.east,
    north: ensemble.north,
    xCells,
    yCells,
    nRealizations,
    meanLogit: reshapeGridRow(meanLogitFlat, xCells, yCells),
    varianceLogit: reshapeGridRow(varianceLogitFlat, xCells, yCells),
    meanPrevalence: reshapeGridRow(meanPrevalenceFlat, xCells, yCells),
    quantiles: quantileMaps,
    exceedances: exceedanceMaps,
    summarizeOn,
  };
}
