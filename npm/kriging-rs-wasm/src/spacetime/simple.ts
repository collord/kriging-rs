/**
 * Space-time simple kriging (known mean) over geographic coordinates.
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
} from "../internal/spacetime.js";
import type { WasmSpaceTimeInstance } from "../internal/wasm-shapes.js";
import type {
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  SpaceTimeSimpleKrigingFromFittedOptions,
  SpaceTimeSimpleKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeSimpleKriging model has been freed";

/**
 * Space-time simple kriging model with a known, spatially-temporally constant mean.
 * Useful when the process mean is estimated externally (e.g. from a deterministic
 * trend removed before kriging).
 */
export class SpaceTimeSimpleKriging {
  private inner: WasmSpaceTimeInstance | null;

  constructor(options: SpaceTimeSimpleKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeSimpleKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeSimpleKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.times),
        toFloat64Array(options.values),
        options.mean,
        options.variogram
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

  /** Build a simple-kriging model from a fitted space-time variogram plus a known mean. */
  static fromFitted(
    options: SpaceTimeSimpleKrigingFromFittedOptions
  ): SpaceTimeSimpleKriging {
    return new SpaceTimeSimpleKriging({
      lats: options.lats,
      lons: options.lons,
      times: options.times,
      values: options.values,
      mean: options.mean,
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
