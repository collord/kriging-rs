/**
 * Simple kriging: ordinary kriging's fixed-mean analogue. Useful when the spatial mean
 * is estimated externally or known from domain knowledge.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import {
  mapOrdinaryBatchArrayOutput,
  mapOrdinaryPrediction,
  mapOrdinaryPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import type { WasmSimpleInstance } from "../internal/wasm-shapes.js";
import type {
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  SimpleKrigingOptions,
} from "../types.js";

const SIMPLE_FREED = "SimpleKriging model has been freed";

/**
 * Simple kriging model: ordinary kriging's fixed-mean analogue. Requires a known
 * `mean` supplied at construction time. Useful when the spatial mean is estimated
 * externally or known from domain knowledge.
 */
export class SimpleKriging {
  private inner: WasmSimpleInstance | null;

  /**
   * Build a simple kriging model from sample data, variogram parameters, and a known mean.
   *
   * @param options - Sample coordinates, values, variogram, and the known `mean`.
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are invalid
   *   (mismatched array lengths, singular covariance), or when the WASM package was built
   *   without this class (code `backend_unavailable`).
   */
  constructor(options: SimpleKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSimpleKriging;
    if (!ctor) {
      throw new KrigingError(
        "SimpleKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.lats),
        toFloat64Array(options.lons),
        toFloat64Array(options.values),
        options.mean,
        options.variogram
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmSimpleInstance {
    if (this.inner === null) {
      throw new KrigingError(SIMPLE_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /** The known mean used to build the model. */
  get mean(): number {
    return this.requireInner().mean();
  }

  /**
   * Release WASM-held resources. Safe to call multiple times; subsequent calls are no-ops.
   * Typically invoked in a `finally` block after model use.
   */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /**
   * Single-point prediction at `(lat, lon)` in degrees.
   *
   * @returns Interpolated value and kriging variance at the location.
   */
  predict(lat: number, lon: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(lat, lon));
  }

  /**
   * Batch prediction at multiple `(lat, lon)` pairs. For large grids prefer
   * {@link SimpleKriging.predictBatchArrays} to avoid per-point object allocation.
   */
  predictBatch(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): OrdinaryPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapOrdinaryPredictionArray(out);
  }

  /**
   * Batch prediction returning typed arrays `{ values, variances }` (one entry per
   * input pair). Prefer over {@link SimpleKriging.predictBatch} for large grids.
   */
  predictBatchArrays(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(lats),
      toFloat64Array(lons)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }
}
