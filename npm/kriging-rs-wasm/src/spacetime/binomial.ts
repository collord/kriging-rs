/**
 * Space-time binomial kriging: prevalence/proportion surfaces from count data
 * that vary in space and time.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array, toUint32Array } from "../internal/convert.js";
import { buildGridLatsLons, reshapeFlatToGrid } from "../internal/grid.js";
import {
  mapBinomialBatchArrayOutput,
  mapBinomialBuildNotes,
  mapBinomialPrediction,
  mapBinomialPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import {
  fittedToSpaceTimeVariogramParams,
} from "../internal/spacetime.js";
import { timesFromDates } from "../time.js";
import type { WasmSpaceTimeBinomialInstance } from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialGridOutput,
  BinomialPrediction,
  DateAxisOptions,
  NumericArrayInput,
  PredictGridAtDateOptions,
  PredictGridAtTimeOptions,
  SpaceTimeBinomialKrigingFromFittedOptions,
  SpaceTimeBinomialKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeBinomialKriging model has been freed";

/**
 * Space-time binomial kriging model. Observations are `(successes, trials)` pairs
 * located at `(lat, lon, time)`; predictions are reported on both the prevalence
 * (`[0, 1]`) and logit scales.
 */
export class SpaceTimeBinomialKriging {
  private inner: WasmSpaceTimeBinomialInstance | null;

  constructor(options: SpaceTimeBinomialKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeBinomialKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeBinomialKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.times),
        toUint32Array(options.successes),
        toUint32Array(options.trials),
        options.variogram
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmSpaceTimeBinomialInstance {
    if (this.inner === null) {
      throw new KrigingError(FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /** Build a binomial-kriging model from sample counts plus a fitted space-time variogram. */
  static fromFitted(
    options: SpaceTimeBinomialKrigingFromFittedOptions
  ): SpaceTimeBinomialKriging {
    return new SpaceTimeBinomialKriging({
      lats: options.lats,
      lons: options.lons,
      times: options.times,
      successes: options.successes,
      trials: options.trials,
      variogram: fittedToSpaceTimeVariogramParams(options.fittedVariogram),
    });
  }

  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /** Build-time diagnostics (prior, dropped rows, logit inflation, …). */
  getBuildNotes(): BinomialBuildNotes {
    try {
      return mapBinomialBuildNotes(this.requireInner().getBuildNotes());
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  predict(lat: number, lon: number, time: number): BinomialPrediction {
    return mapBinomialPrediction(this.requireInner().predict(lat, lon, time));
  }

  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): BinomialPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapBinomialPredictionArray(out);
  }

  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapBinomialBatchArrayOutput(out);
  }

  /**
   * Predict prevalence on a rectangular lat/lon grid at a fixed `time` slice.
   * Builds cell-center coordinates in row-major order, fills a `time` array
   * with the same scalar for every cell, and reshapes results into 2D arrays
   * of shape `[yCells][xCells]`.
   */
  predictGridAtTime(options: PredictGridAtTimeOptions): BinomialGridOutput {
    const inner = this.requireInner();
    const nRows = Math.max(1, Math.floor(options.yCells));
    const nCols = Math.max(1, Math.floor(options.xCells));
    const { lats, lons } = buildGridLatsLons(options);
    const times = new Float64Array(lats.length);
    times.fill(options.time);
    const out = inner.predictBatchArrays(lats, lons, times);
    const {
      prevalences: prevFlat,
      logitValues: logitFlat,
      variances: varFlat,
      prevalenceVariances: prevVarFlat,
    } = mapBinomialBatchArrayOutput(out);
    return {
      prevalences: reshapeFlatToGrid(prevFlat, nRows, nCols),
      logitValues: reshapeFlatToGrid(logitFlat, nRows, nCols),
      variances: reshapeFlatToGrid(varFlat, nRows, nCols),
      prevalenceVariances: reshapeFlatToGrid(prevVarFlat, nRows, nCols),
    };
  }

  // -------------------------------------------------------------------------
  // Date-aware convenience helpers
  //
  // Wrap the numeric `times` API with JS `Date` inputs. Conversion uses the
  // same `timesFromDates` helper exported from this package, so callers can
  // round-trip with `datesFromTimes`. `timeUnit` and `epoch` default to
  // `"days"` and the Unix epoch — match whatever convention you used to build
  // the model.
  // -------------------------------------------------------------------------

  /**
   * Like {@link predict} but takes a JS `Date` and projects it onto the
   * model's numeric time axis using `timeUnit` / `epoch` (defaults `"days"` /
   * Unix epoch). See {@link DateAxisOptions}.
   */
  predictAtDate(
    lat: number,
    lon: number,
    date: Date,
    options: DateAxisOptions = {}
  ): BinomialPrediction {
    const time = dateToTime(date, options);
    return this.predict(lat, lon, time);
  }

  /**
   * Like {@link predictBatch} but takes JS `Date` objects. All `dates` are
   * converted with the same `timeUnit` / `epoch`.
   */
  predictBatchAtDates(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    dates: readonly Date[],
    options: DateAxisOptions = {}
  ): BinomialPrediction[] {
    const times = timesFromDates(dates, options.timeUnit, options.epoch);
    return this.predictBatch(lats, lons, times);
  }

  /**
   * Like {@link predictBatchArrays} but takes JS `Date` objects.
   */
  predictBatchArraysAtDates(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    dates: readonly Date[],
    options: DateAxisOptions = {}
  ): BinomialBatchArrayOutput {
    const times = timesFromDates(dates, options.timeUnit, options.epoch);
    return this.predictBatchArrays(lats, lons, times);
  }

  /**
   * Like {@link predictGridAtTime} but takes a JS `Date`. Useful for slicing
   * a time-series of grids at calendar-aligned dates.
   */
  predictGridAtDate(options: PredictGridAtDateOptions): BinomialGridOutput {
    const time = dateToTime(options.date, options);
    return this.predictGridAtTime({ ...options, time });
  }
}

function dateToTime(date: Date, options: DateAxisOptions): number {
  return timesFromDates([date], options.timeUnit, options.epoch)[0];
}
