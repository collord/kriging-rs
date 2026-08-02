/**
 * Ordinary kriging: spatial interpolation of continuous values over geographic
 * coordinates (Haversine / great-circle distances in km).
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
import type { WasmOrdinaryInstance } from "../internal/wasm-shapes.js";
import type {
  NeighborhoodOptions,
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryGridOutput,
  OrdinaryKrigingFromFittedOptions,
  OrdinaryKrigingOptions,
  OrdinaryPrediction,
  PredictGridOptions,
} from "../types.js";

const ORDINARY_FREED = "OrdinaryKriging model has been freed";

/**
 * Ordinary kriging model for spatial interpolation of continuous values.
 * Coordinates are in degrees (latitude, longitude); distances use Haversine (great-circle).
 * Pass a single options object with data and variogram parameters. For the common
 * "fit variogram then build model" flow, use {@link OrdinaryKriging.fromFitted}.
 *
 * @throws {KrigingError} When the WASM module is not loaded, or when inputs are invalid
 * (e.g. mismatched array lengths, singular covariance).
 */
export class OrdinaryKriging {
  private inner: WasmOrdinaryInstance | null;

  /**
   * Build an ordinary kriging model from sample locations, values, and variogram parameters.
   *
   * @param options - Sample coordinates, values, and variogram (nugget, sill, range, type, optional shape).
   * @throws {KrigingError} When the WASM module is not loaded or when inputs are invalid.
   */
  constructor(options: OrdinaryKrigingOptions) {
    const mod = requireLoadedModule();
    try {
      if (options.variogram.shape2 != null) {
        // Two-shape families (confluent hypergeometric) carry a second shape parameter that
        // the flat `fromArrays` factory cannot express; the options-object constructor
        // threads it through. Only taken for these models, so the common path is unaffected.
        this.inner = new mod.WasmOrdinaryKriging({
          lats: Array.from(toFloat64Array(options.lats)),
          lons: Array.from(toFloat64Array(options.lons)),
          values: Array.from(toFloat64Array(options.values)),
          variogram: {
            variogramType: String(options.variogram.variogramType),
            nugget: options.variogram.nugget,
            sill: options.variogram.sill,
            range: options.variogram.range,
            shape: options.variogram.shape,
            shape2: options.variogram.shape2,
          },
        });
      } else {
        // Prefer the zero-object-overhead `fromArrays` factory: it skips the
        // `serde_wasm_bindgen::from_value` deserialization that would otherwise dominate
        // constructor cost for large sample sets.
        this.inner = mod.WasmOrdinaryKriging.fromArrays(
          toFloat64Array(options.lats),
          toFloat64Array(options.lons),
          toFloat64Array(options.values),
          options.variogram.variogramType,
          options.variogram.nugget,
          options.variogram.sill,
          options.variogram.range,
          options.variogram.shape
        );
      }
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmOrdinaryInstance {
    if (this.inner === null) {
      throw new KrigingError(ORDINARY_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Build an ordinary kriging model from sample data and a fitted variogram (e.g. from
   * {@link fitVariogram}). Avoids manually spreading nugget, sill, range, and shape.
   *
   * @param options - Sample lats, lons, values, and a {@link FittedVariogram}.
   * @returns A new OrdinaryKriging instance.
   * @throws {KrigingError} When the WASM module is not loaded or when inputs are invalid.
   */
  static fromFitted(
    options: OrdinaryKrigingFromFittedOptions
  ): OrdinaryKriging {
    return new OrdinaryKriging({
      lats: options.lats,
      lons: options.lons,
      values: options.values,
      variogram: fittedToVariogramParamsWithNuggetOverride(
        options.fittedVariogram,
        options.nuggetOverride
      ),
    });
  }

  /**
   * Configure a search neighborhood that restricts which stations are used at each
   * prediction location. Pass `{}` (or call with no arguments) to clear any active
   * neighborhood and return to the full-data fast path.
   *
   * - `maxNeighbors` — keep only the `k` closest stations (k-nearest).
   * - `maxRadius` — keep only stations within a great-circle distance in kilometers.
   *   If both are given, the intersection is used.
   *
   * @throws {KrigingError} When the WASM module is not loaded or inputs are invalid.
   */
  setNeighborhood(options?: NeighborhoodOptions): void {
    const inner = this.requireInner();
    try {
      inner.setNeighborhood(options?.maxNeighbors, options?.maxRadius);
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * Returns the currently active search neighborhood (`{ maxNeighbors?, maxRadius? }`),
   * or `null` when the model uses all data at every prediction location.
   */
  neighborhood(): NeighborhoodOptions | null {
    const inner = this.requireInner();
    const out = inner.neighborhood();
    if (out === null || out === undefined) return null;
    const rec = asRecord(out);
    const res: NeighborhoodOptions = {};
    if (typeof rec.maxNeighbors === "number")
      res.maxNeighbors = rec.maxNeighbors;
    if (typeof rec.maxRadius === "number") res.maxRadius = rec.maxRadius;
    return res;
  }

  /**
   * Release WASM-held resources. Call when the model is no longer needed.
   * Safe to call multiple times; subsequent calls are no-ops. You can call
   * free() in a finally block after creating a model.
   */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") {
      this.inner.free();
    }
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /**
   * Single-point prediction at (lat, lon) in degrees.
   *
   * @param lat - Latitude in degrees.
   * @param lon - Longitude in degrees.
   * @returns Interpolated value and kriging variance at the location.
   */
  predict(lat: number, lon: number): OrdinaryPrediction {
    return mapOrdinaryPrediction(this.requireInner().predict(lat, lon));
  }

  /**
   * Batch prediction at multiple (lat, lon) pairs. For large grids, prefer
   * {@link OrdinaryKriging.predictBatchArrays} to avoid per-point object allocation.
   *
   * @param lats - Latitudes in degrees (same length as lons).
   * @param lons - Longitudes in degrees (same length as lats).
   * @returns Array of predictions (value and variance) for each location.
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
   * Batch prediction returning typed arrays (values, variances). Prefer over
   * {@link OrdinaryKriging.predictBatch} for large grids to avoid per-point object allocation.
   *
   * @param lats - Latitudes in degrees (same length as lons).
   * @param lons - Longitudes in degrees (same length as lats).
   * @returns Object with `values` and `variances` Float64Arrays, each of length lats.length.
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

  /**
   * Predict on a rectangular grid defined by bounds and cell counts. Builds cell-center
   * coordinates in row-major order, runs batch prediction, and returns 2D arrays
   * so you avoid manual grid build and reshape. Grid shape is [yCells][xCells];
   * values[j][i] is the prediction at row j (latitude), column i (longitude).
   *
   * @param options - west, south, east, north (degrees), xCells, yCells
   * @returns Object with `values` and `variances` as number[][], shape [yCells][xCells]
   */
  predictGrid(options: PredictGridOptions): OrdinaryGridOutput {
    const inner = this.requireInner();
    const nRows = Math.max(1, Math.floor(options.yCells));
    const nCols = Math.max(1, Math.floor(options.xCells));
    const out = inner.predictGridArrays(
      options.west,
      options.east,
      options.south,
      options.north,
      nCols,
      nRows
    );
    const { values: valuesFlat, variances: variancesFlat } =
      mapOrdinaryBatchArrayOutput(out);
    return {
      values: reshapeFlatToGrid(valuesFlat, nRows, nCols),
      variances: reshapeFlatToGrid(variancesFlat, nRows, nCols),
    };
  }

  /**
   * Batch prediction using WebGPU. Requires building with `npm run build:wasm:gpu`.
   * Throws {@link KrigingError} with code `backend_unavailable` if the GPU path fails for any
   * reason (no adapter, Matérn variogram, readback failure, etc.). Use
   * {@link OrdinaryKriging.predictBatchGpuOrCpu} when silent CPU fallback is desired.
   *
   * @param lats - Latitudes in degrees (same length as lons).
   * @param lons - Longitudes in degrees (same length as lats).
   * @returns Promise resolving to an array of predictions (value and variance) for each location.
   * @throws {KrigingError} code `backend_unavailable` if the GPU path fails, or if the package
   *   was not built with the `gpu` feature.
   */
  async predictBatchGpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<OrdinaryPrediction[]> {
    const inner = this.requireInner();
    if (typeof inner.predictBatchGpu !== "function") {
      throw new KrigingError(
        'predictBatchGpu not available; rebuild WASM package with feature "gpu"',
        { code: "backend_unavailable" }
      );
    }
    try {
      const out = await inner.predictBatchGpu(
        toFloat64Array(lats),
        toFloat64Array(lons)
      );
      return mapOrdinaryPredictionArray(out);
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /**
   * GPU-first batch prediction with automatic CPU fallback on any GPU error. Use when the
   * client GPU availability is unknown and a best-effort path is acceptable. For strict GPU
   * execution that surfaces adapter/shader errors, use {@link OrdinaryKriging.predictBatchGpu}.
   */
  async predictBatchGpuOrCpu(
    lats: NumericArrayInput,
    lons: NumericArrayInput
  ): Promise<OrdinaryPrediction[]> {
    const inner = this.requireInner();
    const latArr = toFloat64Array(lats);
    const lonArr = toFloat64Array(lons);
    if (typeof inner.predictBatchGpuOrCpu === "function") {
      try {
        const out = await inner.predictBatchGpuOrCpu(latArr, lonArr);
        return mapOrdinaryPredictionArray(out);
      } catch (e) {
        throw wrapThrown(e);
      }
    }
    return this.predictBatch(latArr, lonArr);
  }
}
