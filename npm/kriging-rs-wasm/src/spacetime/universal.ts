/**
 * Space-time universal kriging with polynomial drift bases in space and/or time.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import {
  mapOrdinaryBatchArrayOutput,
  mapOrdinaryPrediction,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import {
  fittedToSpaceTimeVariogramParams,
  packSpaceTimeVariogram,
  requireSpaceTimeUniversalTrend,
} from "../internal/spacetime.js";
import type { WasmSpaceTimeInstance } from "../internal/wasm-shapes.js";
import type {
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  SpaceTimeUniversalKrigingFromFittedOptions,
  SpaceTimeUniversalKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeUniversalKriging model has been freed";

/**
 * Space-time universal kriging model with a polynomial drift in the spatial and/or
 * temporal axes (see {@link SpaceTimeUniversalTrend} for available bases).
 */
export class SpaceTimeUniversalKriging {
  private inner: WasmSpaceTimeInstance | null;

  constructor(options: SpaceTimeUniversalKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeUniversalKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeUniversalKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const packed = packSpaceTimeVariogram(options.variogram);
    const trend = requireSpaceTimeUniversalTrend(options.trend);
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.times),
        toFloat64Array(options.values),
        trend,
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

  /** Build a universal-kriging model from a fitted space-time variogram and a trend. */
  static fromFitted(
    options: SpaceTimeUniversalKrigingFromFittedOptions
  ): SpaceTimeUniversalKriging {
    return new SpaceTimeUniversalKriging({
      lats: options.lats,
      lons: options.lons,
      times: options.times,
      values: options.values,
      trend: options.trend,
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

  predict(lat: number, lon: number, time: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(lat, lon, time));
  }

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
}
