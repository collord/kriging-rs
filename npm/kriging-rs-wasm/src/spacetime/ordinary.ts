/**
 * Space-time ordinary kriging over geographic coordinates with a scalar time axis.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import { buildGridLatsLons, reshapeFlatToGrid } from "../internal/grid.js";
import {
  mapOrdinaryBatchArrayOutput,
  mapOrdinaryPrediction,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import {
  fittedToSpaceTimeVariogramParams,
  packSpaceTimeVariogram,
} from "../internal/spacetime.js";
import type { WasmSpaceTimeInstance } from "../internal/wasm-shapes.js";
import type {
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryGridOutput,
  OrdinaryPrediction,
  PredictGridAtTimeOptions,
  SpaceTimeOrdinaryKrigingFromFittedOptions,
  SpaceTimeOrdinaryKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeOrdinaryKriging model has been freed";

/**
 * Space-time ordinary kriging model. Spatial coordinates are interpreted as
 * geographic `(lat, lon)` in degrees (Haversine distances in km); `time` is an
 * arbitrary scalar (days, seconds, etc.).
 */
export class SpaceTimeOrdinaryKriging {
  private inner: WasmSpaceTimeInstance | null;

  constructor(options: SpaceTimeOrdinaryKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeOrdinaryKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeOrdinaryKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const packed = packSpaceTimeVariogram(options.variogram);
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.times),
        toFloat64Array(options.values),
        packed.family,
        packed.spatialType,
        packed.spatialNugget,
        packed.spatialSill,
        packed.spatialRange,
        packed.spatialShape,
        packed.spatialShape2,
        packed.temporalType,
        packed.temporalNugget,
        packed.temporalSill,
        packed.temporalRange,
        packed.temporalShape,
        packed.temporalShape2,
        packed.k1,
        packed.k2,
        packed.k3
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmSpaceTimeInstance {
    if (this.inner === null) {
      throw new KrigingError(FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Build a model from sample data plus a fitted space-time variogram (e.g.
   * from {@link fitSpaceTimeVariogram}). Drops `residuals` and preserves the
   * discriminated-union arms so `fit.fit` flows straight through.
   */
  static fromFitted(
    options: SpaceTimeOrdinaryKrigingFromFittedOptions
  ): SpaceTimeOrdinaryKriging {
    return new SpaceTimeOrdinaryKriging({
      lats: options.lats,
      lons: options.lons,
      times: options.times,
      values: options.values,
      variogram: fittedToSpaceTimeVariogramParams(options.fittedVariogram),
    });
  }

  /** Release WASM-held resources. Safe to call multiple times. */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /** Single-point prediction at `(lat, lon, time)`. */
  predict(lat: number, lon: number, time: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(lat, lon, time));
  }

  /**
   * Batch prediction returning typed arrays `{ values, variances }`.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput,
    times: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons),
      toFloat64Array(times)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }

  /**
   * Predict on a rectangular lat/lon grid at a fixed `time` slice. Builds
   * cell-center coordinates in row-major order, fills a `time` array with the
   * same scalar for every cell, and reshapes results into 2D arrays of shape
   * `[yCells][xCells]`.
   */
  predictGridAtTime(options: PredictGridAtTimeOptions): OrdinaryGridOutput {
    const inner = this.requireInner();
    const nRows = Math.max(1, Math.floor(options.yCells));
    const nCols = Math.max(1, Math.floor(options.xCells));
    const { lats, lons } = buildGridLatsLons(options);
    const times = new Float64Array(lats.length);
    times.fill(options.time);
    const out = inner.predictBatchArrays(lats, lons, times);
    const { values: valuesFlat, variances: variancesFlat } =
      mapOrdinaryBatchArrayOutput(out);
    return {
      values: reshapeFlatToGrid(valuesFlat, nRows, nCols),
      variances: reshapeFlatToGrid(variancesFlat, nRows, nCols),
    };
  }
}
