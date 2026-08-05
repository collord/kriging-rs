/**
 * 3-D ordinary kriging: interpolation of continuous values over Cartesian
 * `(x, y, z)` coordinates with GSLib-convention anisotropy.
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
  OrdinaryKriging3DOptions,
  Prediction3D,
} from "../types.js";

const ORDINARY_3D_FREED = "OrdinaryKriging3D model has been freed";

/**
 * Ordinary kriging model over 3-D Cartesian coordinates (z positive upward).
 * Distances are anisotropic Euclidean in coordinate units; supply
 * {@link Anisotropy3DParams} for rotated / stretched correlation ellipsoids,
 * and an optional kd-tree search neighborhood for large sample sets.
 *
 * @throws {KrigingError} When the WASM module is not loaded, or when inputs are
 * invalid (e.g. mismatched array lengths, singular covariance).
 */
export class OrdinaryKriging3D {
  private inner: WasmKriging3DInstance | null;

  /**
   * Build a 3-D ordinary kriging model from sample locations, values, and
   * variogram parameters.
   *
   * @param options - Sample coordinates, values, variogram, and optional
   *   anisotropy / neighborhood.
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are
   *   invalid, or when the WASM package was built without the 3-D classes
   *   (code `backend_unavailable`).
   */
  constructor(options: OrdinaryKriging3DOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmOrdinaryKriging3D;
    if (!ctor) {
      throw new KrigingError(
        "OrdinaryKriging3D is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const anis = anisotropyOrIdentity(options.anisotropy);
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toFloat64Array(options.zs),
        toFloat64Array(options.values),
        anis.ang1,
        anis.ang2,
        anis.ang3,
        anis.anis1,
        anis.anis2,
        options.variogram,
        options.neighborhood?.maxRadius,
        options.neighborhood?.maxNeighbors
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmKriging3DInstance {
    if (this.inner === null) {
      throw new KrigingError(ORDINARY_3D_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Release WASM-held resources. Safe to call multiple times; subsequent calls
   * are no-ops. Typically invoked in a `finally` block after model use.
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
   * Single-point prediction at `(x, y, z)` in coordinate units.
   *
   * @returns Interpolated value, kriging variance, and solver diagnostics.
   */
  predict(x: number, y: number, z: number): Prediction3D {
    try {
      return mapPrediction3D(this.requireInner().predict(x, y, z));
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Batch prediction returning parallel typed arrays (values, variances,
   * condition numbers, nugget-inflation flags), one entry per target.
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
