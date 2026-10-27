/**
 * Polygon-level aggregation of binomial ensembles.
 *
 * The grid simulators ({@link simulateBinomialGridEnsemble},
 * {@link conditionalSimulateManyBinomial}) return per-cell ensembles, but most
 * disease-mapping deliverables are reported at *administrative units*: districts,
 * counties, health-zones. This module bridges the two:
 *
 * - {@link aggregatePrevalenceByPolygon} reduces an ensemble to per-polygon
 *   posterior summaries (mean, variance, optional quantiles), with a choice of
 *   summarizing on the `"prevalence"` scale (default; intuitive for reporting)
 *   or the `"logit"` scale (often more well-behaved for high-variance polygons).
 * - {@link polygonCellsFromMask} turns a 2-D `[yCells][xCells]` mask into a
 *   flat `(indices, weights)` polygon record. A binary indicator mask gives a
 *   simple area mean; a population raster gives a population-weighted mean.
 *
 * Reduction is performed in Rust through the native WASM batch entry
 * `aggregatePolygonsOverEnsemble`, which avoids per-polygon JS<->WASM crossings
 * and operates directly on `Real`-typed buffers.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "./errors.js";
import { requireLoadedModule } from "./internal/module.js";
import { toFloat64Array, toUint32Array } from "./internal/convert.js";
import type {
  AggregatePrevalenceByPolygonOptions,
  PolygonAggregateResult,
  PolygonCells,
  PolygonCellsFromMaskOptions,
  BinomialGridEnsemble,
  BinomialSimulationManyResult,
} from "./types.js";

function selectSamples(
  ensemble: BinomialGridEnsemble | BinomialSimulationManyResult,
  summarizeOn: "prevalence" | "logit"
): Float64Array {
  return summarizeOn === "logit"
    ? ensemble.logitSamples
    : ensemble.prevalenceSamples;
}

function validatePolygons(
  polygons: ReadonlyArray<PolygonCells>,
  nTargets: number
): { indices: Uint32Array[]; weights: Float64Array[] } {
  if (polygons.length === 0) {
    throw new KrigingError("polygons must contain at least one entry", {
      code: "invalid_input",
    });
  }
  const indices: Uint32Array[] = new Array(polygons.length);
  const weights: Float64Array[] = new Array(polygons.length);
  for (let p = 0; p < polygons.length; p++) {
    const poly = polygons[p];
    const idx = toUint32Array(poly.indices);
    const w = toFloat64Array(poly.weights);
    if (idx.length === 0) {
      throw new KrigingError(`polygon ${p} has no cells`, {
        code: "too_few_points",
      });
    }
    if (idx.length !== w.length) {
      throw new KrigingError(
        `polygon ${p}: indices length (${idx.length}) must equal weights length (${w.length})`,
        { code: "mismatched_arrays" }
      );
    }
    let total = 0;
    for (let i = 0; i < idx.length; i++) {
      const ci = idx[i];
      if (ci >= nTargets) {
        throw new KrigingError(
          `polygon ${p}: cell index ${ci} is out of range (nTargets = ${nTargets})`,
          { code: "invalid_input" }
        );
      }
      const wi = w[i];
      if (!Number.isFinite(wi) || wi < 0) {
        throw new KrigingError(
          `polygon ${p}: weights must be finite and non-negative (got ${wi} at ${i})`,
          { code: "invalid_input" }
        );
      }
      total += wi;
    }
    if (!(total > 0)) {
      throw new KrigingError(
        `polygon ${p}: weights must sum to a strictly positive value`,
        { code: "invalid_input" }
      );
    }
    indices[p] = idx;
    weights[p] = w;
  }
  return { indices, weights };
}

function validateQuantiles(
  quantiles: ReadonlyArray<number> | undefined
): Float64Array {
  if (!quantiles || quantiles.length === 0) return new Float64Array(0);
  for (let i = 0; i < quantiles.length; i++) {
    const p = quantiles[i];
    if (!(p >= 0 && p <= 1)) {
      throw new KrigingError(
        `quantile probabilities must be in [0, 1] (got ${p} at index ${i})`,
        { code: "invalid_input" }
      );
    }
  }
  return toFloat64Array(quantiles as number[]);
}

interface RawPolygonSummary {
  nRealizations: number;
  totalWeight: number;
  mean: number;
  variance: number | null;
  quantileProbabilities: Float64Array;
  quantileValues: Float64Array;
}

/**
 * Reduce one or more polygons over a binomial ensemble buffer. Returns one
 * summary per input polygon, in the same order. Quantiles are reported in the
 * same order as the input `quantiles`. See {@link AggregatePrevalenceByPolygonOptions}.
 *
 * The reduction is performed in Rust via the WASM `aggregatePolygonsOverEnsemble`
 * batch entry: all polygons are flattened into CSR-style `(indices, weights,
 * offsets)` buffers and crossed over to WASM exactly once.
 */
export function aggregatePrevalenceByPolygon(
  options: AggregatePrevalenceByPolygonOptions
): PolygonAggregateResult[] {
  const summarizeOn = options.summarizeOn ?? "prevalence";
  const ensemble = options.ensemble;
  const nRealizations = ensemble.nRealizations;
  const nTargets = ensemble.nTargets;
  if (!Number.isInteger(nRealizations) || nRealizations < 1) {
    throw new KrigingError(
      "ensemble.nRealizations must be a positive integer",
      {
        code: "invalid_input",
      }
    );
  }
  if (!Number.isInteger(nTargets) || nTargets < 1) {
    throw new KrigingError("ensemble.nTargets must be a positive integer", {
      code: "invalid_input",
    });
  }
  const samples = selectSamples(ensemble, summarizeOn);
  if (samples.length !== nRealizations * nTargets) {
    throw new KrigingError(
      `ensemble buffer length (${samples.length}) must equal nRealizations * nTargets (${nRealizations * nTargets})`,
      { code: "mismatched_arrays" }
    );
  }

  const { indices, weights } = validatePolygons(options.polygons, nTargets);
  const quantilesF64 = validateQuantiles(options.quantiles);

  const ids = options.polygons.map((p) => p.id);

  const mod = requireLoadedModule();

  const totalCells = indices.reduce((acc, a) => acc + a.length, 0);
  const flatIndices = new Uint32Array(totalCells);
  const flatWeights = new Float64Array(totalCells);
  const offsets = new Uint32Array(indices.length + 1);
  let off = 0;
  for (let p = 0; p < indices.length; p++) {
    flatIndices.set(indices[p], off);
    flatWeights.set(weights[p], off);
    off += indices[p].length;
    offsets[p + 1] = off;
  }
  try {
    const raw = mod.aggregatePolygonsOverEnsemble(
      samples,
      nRealizations,
      nTargets,
      flatIndices,
      flatWeights,
      offsets,
      quantilesF64
    ) as RawPolygonSummary[];
    return raw.map((s, i) => normalizeSummary(s, ids[i], summarizeOn));
  } catch (e) {
    throw wrapThrown(e);
  }
}

function normalizeSummary(
  raw: RawPolygonSummary,
  id: string | undefined,
  summarizeOn: "prevalence" | "logit"
): PolygonAggregateResult {
  const nq = raw.quantileProbabilities.length;
  const quantiles: { probability: number; value: number }[] = new Array(nq);
  for (let i = 0; i < nq; i++) {
    quantiles[i] = {
      probability: raw.quantileProbabilities[i],
      value: raw.quantileValues[i],
    };
  }
  return {
    id,
    nRealizations: raw.nRealizations,
    totalWeight: raw.totalWeight,
    mean: raw.mean,
    variance: raw.variance,
    quantiles,
    summarizeOn,
  };
}

/**
 * Convert a `[yCells][xCells]` mask aligned with a grid ensemble into a
 * {@link PolygonCells} record suitable for {@link aggregatePrevalenceByPolygon}.
 *
 * - Falsy entries (`0`, `false`, `null`, `undefined`, negative numbers, `NaN`)
 *   are excluded.
 * - Truthy numeric entries are kept as the cell's weight (so a population
 *   raster works as a population-weighted mask).
 * - `true` is treated as weight `1` (binary indicator masks).
 *
 * Cell layout matches {@link gridCellCenters}: cell `(j, i)` lives at flat
 * index `j * xCells + i`, with `j` running south → north and `i` running
 * west → east.
 *
 * @throws {KrigingError} If `mask` does not have shape `[yCells][xCells]` or
 *   if no cells are selected.
 */
export function polygonCellsFromMask(
  options: PolygonCellsFromMaskOptions
): PolygonCells {
  const { mask, xCells, yCells, id } = options;
  if (!Number.isInteger(xCells) || xCells < 1) {
    throw new KrigingError("xCells must be a positive integer", {
      code: "invalid_input",
    });
  }
  if (!Number.isInteger(yCells) || yCells < 1) {
    throw new KrigingError("yCells must be a positive integer", {
      code: "invalid_input",
    });
  }
  if (mask.length !== yCells) {
    throw new KrigingError(
      `mask outer length (${mask.length}) must equal yCells (${yCells})`,
      { code: "mismatched_arrays" }
    );
  }
  // First pass: count selected cells so we can size typed arrays exactly.
  let count = 0;
  for (let j = 0; j < yCells; j++) {
    const row = mask[j];
    if (row.length !== xCells) {
      throw new KrigingError(
        `mask row ${j} length (${row.length}) must equal xCells (${xCells})`,
        { code: "mismatched_arrays" }
      );
    }
    for (let i = 0; i < xCells; i++) {
      const v = row[i];
      if (v === true) {
        count += 1;
      } else if (typeof v === "number" && Number.isFinite(v) && v > 0) {
        count += 1;
      }
    }
  }
  if (count === 0) {
    throw new KrigingError("mask selects no cells", {
      code: "too_few_points",
    });
  }
  const indices = new Uint32Array(count);
  const weights = new Float64Array(count);
  let k = 0;
  for (let j = 0; j < yCells; j++) {
    const row = mask[j];
    const rowOff = j * xCells;
    for (let i = 0; i < xCells; i++) {
      const v = row[i];
      let w: number;
      if (v === true) w = 1;
      else if (typeof v === "number" && Number.isFinite(v) && v > 0) w = v;
      else continue;
      indices[k] = rowOff + i;
      weights[k] = w;
      k += 1;
    }
  }
  return id !== undefined ? { id, indices, weights } : { indices, weights };
}
