/**
 * Projected / planar ordinary kriging on `(x, y)` coordinates with 2D anisotropy.
 * Distances are Euclidean.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "../errors.js";
import { asRecord, toFloat64Array } from "../internal/convert.js";
import {
  fittedToVariogramParamsWithNuggetOverride,
  reshapeFlatToGrid,
} from "../internal/grid.js";
import {
  mapOrdinaryBatchArrayOutput,
  mapOrdinaryPrediction,
  mapOrdinaryPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import type { WasmProjectedInstance } from "../internal/wasm-shapes.js";
import type {
  NeighborhoodOptions,
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  PredictGridProjectedOptions,
  ProjectedGridOutput,
  ProjectedKrigingFromFittedOptions,
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
   * @param options - Planar `(x, y)` coordinates, values, a variogram whose `range` is in
   *   coordinate units, optional `majorAngleDeg` (degrees CCW from +x, default 0) and
   *   `rangeRatio` in `(0, 1]` (minor-to-major, default 1), and an optional `neighborhood`.
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
        options.majorAngleDeg ?? 0,
        options.rangeRatio ?? 1,
        options.neighborhood?.maxNeighbors,
        options.neighborhood?.maxRadius
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Build a model from a fitted variogram (e.g. {@link fitVariogramProjected}) without
   * restating its fields.
   *
   * @example
   * const fitted = fitVariogramProjected({ xs, ys, values, variogramType: "spherical" });
   * using model = ProjectedKriging.fromFitted({
   *   xs, ys, values, fittedVariogram: fitted, neighborhood: { maxNeighbors: 32 },
   * });
   */
  static fromFitted(
    options: ProjectedKrigingFromFittedOptions
  ): ProjectedKriging {
    return new ProjectedKriging({
      xs: options.xs,
      ys: options.ys,
      values: options.values,
      variogram: fittedToVariogramParamsWithNuggetOverride(
        options.fittedVariogram,
        options.nuggetOverride
      ),
      majorAngleDeg: options.majorAngleDeg,
      rangeRatio: options.rangeRatio,
      neighborhood: options.neighborhood,
    });
  }

  /**
   * Restrict predictions to the `maxNeighbors` nearest stations and/or those within
   * `maxRadius` (coordinate units, measured with the model's anisotropy). Pass `{}` or
   * nothing to use every station again; on a model built with a neighborhood that first
   * factorizes the full system.
   *
   * @throws {KrigingError} On `maxNeighbors < 1`, a non-positive `maxRadius`, or a
   *   singular full system when clearing.
   */
  setNeighborhood(options?: NeighborhoodOptions): void {
    const inner = this.requireInner();
    try {
      inner.setNeighborhood(options?.maxNeighbors, options?.maxRadius);
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /** The active search neighborhood, or `null` when every station is used. */
  neighborhood(): NeighborhoodOptions | null {
    const out = this.requireInner().neighborhood();
    if (out === null || out === undefined) return null;
    const rec = asRecord(out);
    const res: NeighborhoodOptions = {};
    if (typeof rec.maxNeighbors === "number")
      res.maxNeighbors = rec.maxNeighbors;
    if (typeof rec.maxRadius === "number") res.maxRadius = rec.maxRadius;
    return res;
  }

  /**
   * Predict at the centres of a regular `xCells × yCells` grid over
   * `[xMin, xMax] × [yMin, yMax]`. Row 0 is the southern row (`yMin`); see
   * {@link ProjectedGridOutput} for drawing north-up.
   *
   * @throws {KrigingError} With code `invalid_input` for non-finite or empty bounds.
   */
  predictGrid(options: PredictGridProjectedOptions): ProjectedGridOutput {
    const { xMin, xMax, yMin, yMax } = options;
    const nCols = Math.max(1, Math.floor(options.xCells));
    const nRows = Math.max(1, Math.floor(options.yCells));
    if (
      ![xMin, xMax, yMin, yMax].every(Number.isFinite) ||
      xMax <= xMin ||
      yMax <= yMin
    ) {
      throw new KrigingError(
        "predictGrid needs finite bounds with xMax > xMin and yMax > yMin",
        { code: "invalid_input" }
      );
    }
    const dx = (xMax - xMin) / nCols;
    const dy = (yMax - yMin) / nRows;
    const xCenters = Float64Array.from(
      { length: nCols },
      (_, i) => xMin + (i + 0.5) * dx
    );
    const yCenters = Float64Array.from(
      { length: nRows },
      (_, j) => yMin + (j + 0.5) * dy
    );
    const xs = new Float64Array(nRows * nCols);
    const ys = new Float64Array(nRows * nCols);
    for (let j = 0; j < nRows; j++) {
      for (let i = 0; i < nCols; i++) {
        xs[j * nCols + i] = xCenters[i];
        ys[j * nCols + i] = yCenters[j];
      }
    }
    const { values, variances } = this.predictBatchArrays(xs, ys);
    return {
      values: reshapeFlatToGrid(values, nRows, nCols),
      variances: reshapeFlatToGrid(variances, nRows, nCols),
      valuesFlat: values,
      variancesFlat: variances,
      xCenters,
      yCenters,
    };
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
