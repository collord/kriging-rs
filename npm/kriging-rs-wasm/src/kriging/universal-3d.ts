/**
 * 3-D universal kriging with a linear drift basis `[1, x, y, z]` over
 * Cartesian coordinates.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array } from "../internal/convert.js";
import {
  mapBatch3DArrayOutput,
  mapPrediction3D,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import { anisotropyOrIdentity } from "../internal/three-d.js";
import type { WasmKriging3DInstance } from "../internal/wasm-shapes.js";
import type {
  Batch3DArrayOutput,
  NumericArrayInput,
  Prediction3D,
  UniversalKriging3DOptions,
} from "../types.js";

const UNIVERSAL_3D_FREED = "UniversalKriging3D model has been freed";

/**
 * Universal kriging model over 3-D Cartesian coordinates with the linear
 * drift basis `[1, x, y, z]` (the only basis exposed in v1). Use when the
 * field has a systematic linear trend the stationary models cannot capture.
 */
export class UniversalKriging3D {
  private inner: WasmKriging3DInstance | null;

  /**
   * Build a 3-D universal kriging model from sample data and variogram
   * parameters.
   *
   * @param options - Sample coordinates, values, variogram, and optional
   *   anisotropy.
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are
   *   invalid, or when the WASM package was built without the 3-D classes
   *   (code `backend_unavailable`).
   */
  constructor(options: UniversalKriging3DOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmUniversalKriging3D;
    if (!ctor) {
      throw new KrigingError(
        "UniversalKriging3D is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const anis = anisotropyOrIdentity(options.anisotropy);
    try {
      this.inner = ctor.fromArraysLinear(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toFloat64Array(options.zs),
        toFloat64Array(options.values),
        anis.ang1,
        anis.ang2,
        anis.ang3,
        anis.anis1,
        anis.anis2,
        options.variogram.variogramType,
        options.variogram.nugget,
        options.variogram.sill,
        options.variogram.range,
        options.variogram.shape
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmKriging3DInstance {
    if (this.inner === null) {
      throw new KrigingError(UNIVERSAL_3D_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Release WASM-held resources. Safe to call multiple times; subsequent calls
   * are no-ops.
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

  /** Single-point prediction at `(x, y, z)` in coordinate units. */
  predict(x: number, y: number, z: number): Prediction3D {
    try {
      return mapPrediction3D(this.requireInner().predict(x, y, z));
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Batch prediction returning parallel typed arrays, one entry per target.
   */
  predictBatch(
    xs: NumericArrayInput,
    ys: NumericArrayInput,
    zs: NumericArrayInput
  ): Batch3DArrayOutput {
    try {
      const out = this.requireInner().predictBatch(
        toFloat64Array(xs),
        toFloat64Array(ys),
        toFloat64Array(zs)
      );
      return mapBatch3DArrayOutput(out);
    } catch (e) {
      throw wrapThrown(e);
    }
  }
}
