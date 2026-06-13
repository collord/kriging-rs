/**
 * 3-D simple kriging: ordinary kriging's fixed-mean analogue over Cartesian
 * `(x, y, z)` coordinates.
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
  SimpleKriging3DOptions,
} from "../types.js";

const SIMPLE_3D_FREED = "SimpleKriging3D model has been freed";

/**
 * Simple kriging model over 3-D Cartesian coordinates. Requires a known global
 * `mean` supplied at construction time; useful when the spatial mean is
 * estimated externally or known from domain knowledge.
 */
export class SimpleKriging3D {
  private inner: WasmKriging3DInstance | null;

  /**
   * Build a 3-D simple kriging model from sample data, variogram parameters,
   * and a known mean.
   *
   * @param options - Sample coordinates, values, variogram, the known `mean`,
   *   and optional anisotropy.
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are
   *   invalid, or when the WASM package was built without the 3-D classes
   *   (code `backend_unavailable`).
   */
  constructor(options: SimpleKriging3DOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmSimpleKriging3D;
    if (!ctor) {
      throw new KrigingError(
        "SimpleKriging3D is not available; rebuild the WASM package",
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
        options.mean,
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
      throw new KrigingError(SIMPLE_3D_FREED, { code: "model_freed" });
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
