/**
 * Space-time ordinary kriging on projected (planar) coordinates with 2-D anisotropy.
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
  SpaceTimeProjectedOrdinaryKrigingFromFittedOptions,
  SpaceTimeProjectedOrdinaryKrigingOptions,
} from "../types.js";

const FREED = "SpaceTimeProjectedOrdinaryKriging model has been freed";

/**
 * Space-time ordinary kriging on projected planar coordinates `(x, y)` with 2-D
 * anisotropy. Useful when inputs are already projected (e.g. meters) and spatial
 * correlation has a preferred direction.
 */
export class SpaceTimeProjectedOrdinaryKriging {
  private inner: WasmSpaceTimeInstance | null;

  constructor(options: SpaceTimeProjectedOrdinaryKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSpaceTimeOrdinaryProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "SpaceTimeProjectedOrdinaryKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toFloat64Array(options.times),
        toFloat64Array(options.values),
        options.majorAngleDeg,
        options.rangeRatio,
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

  /**
   * Build a projected space-time ordinary kriging model from planar sample data
   * plus a fitted space-time variogram.
   */
  static fromFitted(
    options: SpaceTimeProjectedOrdinaryKrigingFromFittedOptions
  ): SpaceTimeProjectedOrdinaryKriging {
    return new SpaceTimeProjectedOrdinaryKriging({
      xs: options.xs,
      ys: options.ys,
      times: options.times,
      values: options.values,
      majorAngleDeg: options.majorAngleDeg,
      rangeRatio: options.rangeRatio,
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

  predict(x: number, y: number, time: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(x, y, time));
  }

  predictBatchArrays(
    xs: NumericArrayInput,
    ys: NumericArrayInput,
    times: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(xs),
      toFloat64Array(ys),
      toFloat64Array(times)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }
}
