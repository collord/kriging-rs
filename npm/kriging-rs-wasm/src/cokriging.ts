/**
 * Cokriging: multi-variable interpolation and cosimulation.
 *
 * - {@link cokrigeCollocated} / {@link collocatedCosimulate} — collocated cokriging &
 *   cosimulation of a primary from a densely-sampled secondary (Markov Model 1). Build the
 *   secondary characterization with {@link collocatedSecondaryFromPaired}.
 * - {@link cokrige} / {@link cosimulateMultiVariable} — full block cokriging and joint
 *   cosimulation over a Linear Model of Coregionalization ({@link CoregionalizationSpec}).
 * - {@link computeCrossVariogram} / {@link fitLmc} — empirical cross-variograms and
 *   Goulard–Voltz LMC fitting (returns an admissible coregionalization for the functions above).
 *
 * @module
 */

import { wrapThrown } from "./errors.js";
import {
  asRecord,
  requireFloat64Array,
  requireNumber,
  toFloat64Array,
} from "./internal/convert.js";
import { mapOrdinaryPredictionArray } from "./internal/mappers.js";
import { requireLoadedModule } from "./internal/module.js";
import type {
  CokrigeCollocatedOptions,
  CokrigeOptions,
  CollocatedCosimulateOptions,
  CollocatedSecondary,
  CosimulateMultiVariableOptions,
  CosimulationResult,
  CrossVariogramOptions,
  CrossVariogramResult,
  FitLmcOptions,
  LmcFitResult,
  MultiVariableData,
  NumericArrayInput,
  OrdinaryPrediction,
} from "./types.js";

function toNumberArray(input: NumericArrayInput): number[] {
  return Array.from(toFloat64Array(input));
}

function normalizeSeed(seed: number | bigint | undefined): bigint {
  return typeof seed === "bigint" ? seed : BigInt(seed ?? 0);
}

function normalizeTargetOrder(
  order: ArrayLike<number> | Uint32Array | undefined
): Uint32Array | undefined {
  if (!order) return undefined;
  return order instanceof Uint32Array
    ? order
    : Uint32Array.from(order as ArrayLike<number>);
}

function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected a numeric array");
  }
  return value.map((v) => requireNumber(v));
}

function numberMatrix(value: unknown): number[][] {
  if (!Array.isArray(value)) {
    throw new Error("Expected a matrix (array of arrays)");
  }
  return value.map((row) => {
    if (!Array.isArray(row)) {
      throw new Error("Expected each matrix row to be an array");
    }
    return row.map((v) => requireNumber(v));
  });
}

/** Pack public {@link MultiVariableData} into the plain-array payload the WASM boundary expects. */
function packData(data: MultiVariableData): Record<string, unknown> {
  if ("perVariable" in data) {
    return {
      perVariable: data.perVariable.map((v) => ({
        lats: toNumberArray(v.lats),
        lons: toNumberArray(v.lons),
        values: toNumberArray(v.values),
      })),
    };
  }
  return {
    lats: toNumberArray(data.lats),
    lons: toNumberArray(data.lons),
    variables: data.variables.map(toNumberArray),
  };
}

/**
 * Collocated simple cokriging of a primary variable given a dense secondary known at every
 * target. Returns one `{ value, variance }` per target, in input order.
 */
export function cokrigeCollocated(
  options: CokrigeCollocatedOptions
): OrdinaryPrediction[] {
  const mod = requireLoadedModule();
  try {
    const out = mod.cokrigeCollocated({
      lats: toNumberArray(options.lats),
      lons: toNumberArray(options.lons),
      values: toNumberArray(options.values),
      variogram: options.variogram,
      primaryMean: options.primaryMean,
      secondary: options.secondary,
      targetLats: toNumberArray(options.targetLats),
      targetLons: toNumberArray(options.targetLons),
      targetSecondaryValues: toNumberArray(options.targetSecondaryValues),
    });
    return mapOrdinaryPredictionArray(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Estimate a {@link CollocatedSecondary} (`{ mean, stdDev, correlation }`) from primary and
 * secondary samples measured at the same locations.
 */
export function collocatedSecondaryFromPaired(
  primary: NumericArrayInput,
  secondary: NumericArrayInput
): CollocatedSecondary {
  const mod = requireLoadedModule();
  try {
    const out = mod.collocatedSecondaryFromPaired(
      toFloat64Array(primary),
      toFloat64Array(secondary)
    );
    const rec = asRecord(out);
    return {
      mean: requireNumber(rec.mean),
      stdDev: requireNumber(rec.stdDev),
      correlation: requireNumber(rec.correlation),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Sequential Gaussian cosimulation of a primary honoring a collocated secondary. Returns a
 * `Float64Array` of primary realizations in input target order; deterministic for a `seed`.
 */
export function collocatedCosimulate(
  options: CollocatedCosimulateOptions
): Float64Array {
  const mod = requireLoadedModule();
  try {
    const out = mod.collocatedCosimulate({
      conditioningLats: toNumberArray(options.conditioningLats),
      conditioningLons: toNumberArray(options.conditioningLons),
      conditioningValues: toNumberArray(options.conditioningValues),
      targetLats: toNumberArray(options.targetLats),
      targetLons: toNumberArray(options.targetLons),
      targetSecondaryValues: toNumberArray(options.targetSecondaryValues),
      variogram: options.variogram,
      primaryMean: options.primaryMean,
      secondary: options.secondary,
      seed: normalizeSeed(options.seed),
      targetOrder: normalizeTargetOrder(options.targetOrder),
    });
    return requireFloat64Array(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Full block cokriging (simple or ordinary) of one `targetVariable` over an LMC, using every
 * variable's data. Returns one `{ value, variance }` per target.
 */
export function cokrige(options: CokrigeOptions): OrdinaryPrediction[] {
  const mod = requireLoadedModule();
  try {
    const out = mod.cokrige({
      data: packData(options.data),
      coregionalization: options.coregionalization,
      kind: options.kind,
      targetVariable: options.targetVariable,
      targetLats: toNumberArray(options.targetLats),
      targetLons: toNumberArray(options.targetLons),
    });
    return mapOrdinaryPredictionArray(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Multivariate sequential Gaussian cosimulation of all variables jointly over an LMC (simple
 * cokriging with known `means`). Returns `{ nVariables, nTargets, samples }`.
 */
export function cosimulateMultiVariable(
  options: CosimulateMultiVariableOptions
): CosimulationResult {
  const mod = requireLoadedModule();
  try {
    const out = mod.cosimulate({
      data: packData(options.data),
      coregionalization: options.coregionalization,
      means: options.means,
      targetLats: toNumberArray(options.targetLats),
      targetLons: toNumberArray(options.targetLons),
      seed: normalizeSeed(options.seed),
      targetOrder: normalizeTargetOrder(options.targetOrder),
    });
    const rec = asRecord(out);
    return {
      nVariables: requireNumber(rec.nVariables),
      nTargets: requireNumber(rec.nTargets),
      samples: numberMatrix(rec.samples),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Empirical (isotopic) cross-variogram matrix per lag. Returns
 * `{ nVariables, distances, nPairs, gamma }` where `gamma[bin]` is the flattened `p × p` matrix.
 */
export function computeCrossVariogram(
  options: CrossVariogramOptions
): CrossVariogramResult {
  const mod = requireLoadedModule();
  try {
    const out = mod.computeCrossVariogram({
      lats: toNumberArray(options.lats),
      lons: toNumberArray(options.lons),
      variables: options.variables.map(toNumberArray),
      nBins: options.nBins,
      maxDistance: options.maxDistance,
    });
    const rec = asRecord(out);
    return {
      nVariables: requireNumber(rec.nVariables),
      distances: numberArray(rec.distances),
      nPairs: numberArray(rec.nPairs),
      gamma: numberMatrix(rec.gamma),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Fit an LMC to isotopic data by Goulard–Voltz: computes the empirical cross-variogram, then
 * fits each basic structure's PSD sill matrix. Returns `{ coregionalization, residual,
 * iterations }`; the `coregionalization` can be passed directly to {@link cokrige} /
 * {@link cosimulateMultiVariable}.
 */
export function fitLmc(options: FitLmcOptions): LmcFitResult {
  const mod = requireLoadedModule();
  try {
    const out = mod.fitLmc({
      lats: toNumberArray(options.lats),
      lons: toNumberArray(options.lons),
      variables: options.variables.map(toNumberArray),
      nBins: options.nBins,
      maxDistance: options.maxDistance,
      bases: options.bases,
      maxIterations: options.maxIterations,
      tolerance: options.tolerance,
    });
    const rec = asRecord(out);
    return {
      coregionalization: rec.coregionalization as LmcFitResult["coregionalization"],
      residual: requireNumber(rec.residual),
      iterations: requireNumber(rec.iterations),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}
