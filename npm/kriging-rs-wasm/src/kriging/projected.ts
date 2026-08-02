/**
 * Projected / planar ordinary kriging on `(x, y)` coordinates with 2D anisotropy.
 * Distances are Euclidean.
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
import type { WasmProjectedInstance } from "../internal/wasm-shapes.js";
import type {
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  ProjectedKrigingOptions,
} from "../types.js";

const PROJECTED_FREED = "ProjectedKriging model has been freed";

/**
 * Projected / planar ordinary kriging on `(x, y)` coordinates with 2D anisotropy.
 * Distances are Euclidean. Useful when the input is already projected (e.g. meters
 * in a local coordinate system) and directional correlation matters.
 */
export class ProjectedKriging {
  private inner: WasmProjectedInstance | null;

  /**
   * Build a projected / planar ordinary kriging model with 2D anisotropy.
   *
   * @param options - Planar `(x, y)` coordinates, values, variogram, `majorAngleDeg`
   *   (degrees CCW from +x), and `rangeRatio` in `(0, 1]` (minor-to-major range ratio).
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are invalid
   *   (mismatched array lengths, singular covariance, non-finite angle, `rangeRatio ∉ (0,1]`),
   *   or when the WASM package was built without this class (code `backend_unavailable`).
   */
  constructor(options: ProjectedKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "ProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toFloat64Array(options.values),
        options.variogram,
        options.majorAngleDeg,
        options.rangeRatio
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmProjectedInstance {
    if (this.inner === null) {
      throw new KrigingError(PROJECTED_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Release WASM-held resources. Safe to call multiple times; subsequent calls are no-ops.
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
   * Single-point prediction at planar `(x, y)`.
   *
   * @returns Interpolated value and kriging variance at the location.
   */
  predict(x: number, y: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(x, y));
  }

  /**
   * Batch prediction at multiple `(x, y)` pairs. For large grids prefer
   * {@link ProjectedKriging.predictBatchArrays} to avoid per-point object allocation.
   */
  predictBatch(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): OrdinaryPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(xs),
      toFloat64Array(ys)
    );
    return mapOrdinaryPredictionArray(out);
  }

  /**
   * Batch prediction returning typed arrays `{ values, variances }`. Prefer over
   * {@link ProjectedKriging.predictBatch} for large grids.
   */
  predictBatchArrays(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): OrdinaryBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(xs),
      toFloat64Array(ys)
    );
    return mapOrdinaryBatchArrayOutput(out);
  }
}
