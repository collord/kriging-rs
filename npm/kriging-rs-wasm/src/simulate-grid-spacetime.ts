/**
 * Grid-shaped binomial **space-time** conditional simulation helpers.
 *
 * Companion to `simulate-grid.ts` for callers whose data are space-time count
 * observations conditioned with a fitted space-time variogram. Each helper
 * mirrors its purely spatial sibling but also takes the conditioning `times`
 * array and a scalar `time` (or {@link Date}) at which to evaluate the grid:
 *
 * - {@link simulateBinomialSpaceTimeGrid} — one realization at fixed `time`,
 *   shaped as `[yCells][xCells]`.
 * - {@link simulateBinomialSpaceTimeGridEnsemble} — `nRealizations` realizations
 *   at fixed `time` as flat row-major buffers (feed straight into the
 *   `ensemble*` aggregators).
 * - {@link simulateBinomialSpaceTimeGridSummary} — one-shot ensemble + per-cell
 *   reduction at fixed `time`.
 *
 * Each function also has a `*AtDate` variant that accepts JS `Date` objects;
 * dates are projected onto the numeric time axis with {@link timesFromDates}
 * (defaults: `timeUnit = "days"`, `epoch = new Date(0)`). Use whichever epoch
 * / unit you used when building the variogram.
 *
 * Cell layout matches `simulate-grid.ts` and {@link BinomialGridOutput}:
 * cell `(j, i)` has center `lat = south + (j + 0.5) * dy`,
 * `lon = west + (i + 0.5) * dx`; row index `j` runs south → north and column
 * index `i` west → east. Cells are emitted in row-major `(j, i)` order with
 * `i` (longitude) fastest.
 *
 * @module
 */

import { KrigingError } from "./errors.js";
import {
  conditionalSimulateSpaceTimeBinomial,
  conditionalSimulateManySpaceTimeBinomial,
} from "./simulation.js";
import { gridCellCenters, reshapeGridRow } from "./simulate-grid.js";
import { timesFromDates } from "./time.js";
import {
  ensembleExceedanceProbability,
  ensembleMean,
  ensembleQuantiles,
  ensembleVariance,
} from "./aggregate.js";
import type {
  BinomialGridEnsemble,
  BinomialGridSimulation,
  BinomialGridSummary,
  DateAxisOptions,
  SimulateBinomialSpaceTimeGridAtDateOptions,
  SimulateBinomialSpaceTimeGridEnsembleAtDateOptions,
  SimulateBinomialSpaceTimeGridEnsembleOptions,
  SimulateBinomialSpaceTimeGridOptions,
  SimulateBinomialSpaceTimeGridSummaryAtDateOptions,
  SimulateBinomialSpaceTimeGridSummaryOptions,
} from "./types.js";

function ensureFiniteNumber(value: number, label: string): number {
  if (!Number.isFinite(value)) {
    throw new KrigingError(`${label} must be a finite number`, {
      code: "invalid_input",
    });
  }
  return value;
}

function dateToTime(date: Date, options: DateAxisOptions): number {
  return timesFromDates([date], options.timeUnit, options.epoch)[0];
}

/**
 * Draw a single binomial space-time SGS realization over the requested grid at
 * a fixed `time` slice and reshape it to nested `[yCells][xCells]` arrays.
 * Deterministic for a given `seed`.
 */
export function simulateBinomialSpaceTimeGrid(
  options: SimulateBinomialSpaceTimeGridOptions
): BinomialGridSimulation {
  ensureFiniteNumber(options.time, "time");
  const { lats: targetLats, lons: targetLons } = gridCellCenters(options);
  const targetTimes = new Float64Array(targetLats.length);
  targetTimes.fill(options.time);
  const result = conditionalSimulateSpaceTimeBinomial({
    conditioningLats: options.lats,
    conditioningLons: options.lons,
    conditioningTimes: options.times,
    successes: options.successes,
    trials: options.trials,
    targetLats,
    targetLons,
    targetTimes,
    variogram: options.variogram,
    prior: options.prior,
    seed: options.seed,
  });
  return {
    logitSamples: reshapeGridRow(
      result.logitSamples,
      options.xCells,
      options.yCells
    ),
    prevalences: reshapeGridRow(
      result.prevalenceSamples,
      options.xCells,
      options.yCells
    ),
  };
}

/**
 * Draw `nRealizations` independent binomial space-time SGS realizations at a
 * fixed `time` slice. Returns flat row-major ensemble buffers of length
 * `nRealizations * (xCells * yCells)`, ready to feed into `ensembleMean`,
 * `ensembleQuantiles`, etc.
 */
export function simulateBinomialSpaceTimeGridEnsemble(
  options: SimulateBinomialSpaceTimeGridEnsembleOptions
): BinomialGridEnsemble {
  ensureFiniteNumber(options.time, "time");
  const { lats: targetLats, lons: targetLons } = gridCellCenters(options);
  const targetTimes = new Float64Array(targetLats.length);
  targetTimes.fill(options.time);
  const result = conditionalSimulateManySpaceTimeBinomial({
    conditioningLats: options.lats,
    conditioningLons: options.lons,
    conditioningTimes: options.times,
    successes: options.successes,
    trials: options.trials,
    targetLats,
    targetLons,
    targetTimes,
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
    xCells: options.xCells,
    yCells: options.yCells,
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
 * One-shot space-time grid simulation at a fixed `time` slice + per-cell
 * reduction. Returns mean / variance on the logit scale, mean on the
 * prevalence scale, plus optional quantile and exceedance maps on the chosen
 * `summarizeOn` scale (`"prevalence"` by default).
 *
 * For very large grids or large ensembles, prefer
 * {@link simulateBinomialSpaceTimeGridEnsemble} so the underlying flat buffers
 * stay in memory only as long as you actually need them.
 */
export function simulateBinomialSpaceTimeGridSummary(
  options: SimulateBinomialSpaceTimeGridSummaryOptions
): BinomialGridSummary {
  const ensemble = simulateBinomialSpaceTimeGridEnsemble(options);
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
      : new Float64Array(nTargets);
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

// -----------------------------------------------------------------------------
// Date-aware variants
// -----------------------------------------------------------------------------

function adaptDateOptions<T extends DateAxisOptions>(
  opts: T
): {
  times: Float64Array;
  time: number;
} {
  const times = timesFromDates(
    (opts as unknown as { dates: readonly Date[] }).dates,
    opts.timeUnit,
    opts.epoch
  );
  const time = dateToTime((opts as unknown as { date: Date }).date, opts);
  return { times, time };
}

/** Date-aware variant of {@link simulateBinomialSpaceTimeGrid}. */
export function simulateBinomialSpaceTimeGridAtDate(
  options: SimulateBinomialSpaceTimeGridAtDateOptions
): BinomialGridSimulation {
  const { times, time } = adaptDateOptions(options);
  return simulateBinomialSpaceTimeGrid({
    west: options.west,
    south: options.south,
    east: options.east,
    north: options.north,
    xCells: options.xCells,
    yCells: options.yCells,
    lats: options.lats,
    lons: options.lons,
    times,
    successes: options.successes,
    trials: options.trials,
    variogram: options.variogram,
    prior: options.prior,
    time,
    seed: options.seed,
  });
}

/** Date-aware variant of {@link simulateBinomialSpaceTimeGridEnsemble}. */
export function simulateBinomialSpaceTimeGridEnsembleAtDate(
  options: SimulateBinomialSpaceTimeGridEnsembleAtDateOptions
): BinomialGridEnsemble {
  const { times, time } = adaptDateOptions(options);
  return simulateBinomialSpaceTimeGridEnsemble({
    west: options.west,
    south: options.south,
    east: options.east,
    north: options.north,
    xCells: options.xCells,
    yCells: options.yCells,
    lats: options.lats,
    lons: options.lons,
    times,
    successes: options.successes,
    trials: options.trials,
    variogram: options.variogram,
    prior: options.prior,
    time,
    nRealizations: options.nRealizations,
    baseSeed: options.baseSeed,
  });
}

/** Date-aware variant of {@link simulateBinomialSpaceTimeGridSummary}. */
export function simulateBinomialSpaceTimeGridSummaryAtDate(
  options: SimulateBinomialSpaceTimeGridSummaryAtDateOptions
): BinomialGridSummary {
  const { times, time } = adaptDateOptions(options);
  return simulateBinomialSpaceTimeGridSummary({
    west: options.west,
    south: options.south,
    east: options.east,
    north: options.north,
    xCells: options.xCells,
    yCells: options.yCells,
    lats: options.lats,
    lons: options.lons,
    times,
    successes: options.successes,
    trials: options.trials,
    variogram: options.variogram,
    prior: options.prior,
    time,
    nRealizations: options.nRealizations,
    baseSeed: options.baseSeed,
    quantiles: options.quantiles,
    exceedanceThresholds: options.exceedanceThresholds,
    summarizeOn: options.summarizeOn,
  });
}
