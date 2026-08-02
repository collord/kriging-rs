/**
 * Variogram entry points: parametric fitting (`fitVariogram`) plus direct empirical
 * variogram APIs (`computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`)
 * and nested-variogram evaluation (`evaluateNestedVariogram`).
 *
 * @module
 */

import { wrapThrown } from "./errors.js";
import {
  asRecord,
  requireFloat64Array,
  requireNumber,
  requireVariogramType,
  toFloat64Array,
} from "./internal/convert.js";
import { mapEmpiricalVariogram } from "./internal/mappers.js";
import {
  requireLoadedModule,
  variogramTypeToNumber,
} from "./internal/module.js";
import type {
  ComputeDirectionalEmpiricalVariogramOptions,
  ComputeEmpiricalVariogramOptions,
  EmpiricalVariogramResult,
  FitVariogramOptions,
  FittedVariogram,
  NestedVariogramComponent,
  NestedVariogramEvaluation,
  NumericArrayInput,
} from "./types.js";

/**
 * Fit a variogram model to sample data by computing an empirical variogram and fitting
 * the specified model type. Use the returned {@link FittedVariogram} to construct an
 * {@link OrdinaryKriging} model.
 *
 * @param options - Single options object with sample data and variogram type
 * @returns Fitted variogram parameters
 * @example
 * const lats = [37.7, 37.71, 37.72];
 * const lons = [-122.45, -122.44, -122.43];
 * const values = [10, 12, 11];
 * const fitted = fitVariogram({ sampleLats: lats, sampleLons: lons, values, variogramType: "exponential" });
 * const model = new OrdinaryKriging({
 *   lats, lons, values,
 *   variogram: { variogramType: fitted.variogramType, nugget: fitted.nugget, sill: fitted.sill, range: fitted.range, shape: fitted.shape },
 * });
 */
export function fitVariogram(options: FitVariogramOptions): FittedVariogram {
  const {
    sampleLats,
    sampleLons,
    values,
    variogramType,
    maxDistance,
    nBins = 12,
    estimator,
  } = options;
  const mod = requireLoadedModule();
  const variogramTypeNum = variogramTypeToNumber(variogramType);
  let out: unknown;
  try {
    out = mod.fitVariogram(
      toFloat64Array(sampleLats),
      toFloat64Array(sampleLons),
      toFloat64Array(values),
      maxDistance,
      nBins,
      variogramTypeNum,
      estimator
    );
  } catch (e) {
    throw wrapThrown(e);
  }
  const result = asRecord(out);
  const fitted: FittedVariogram = {
    variogramType: requireVariogramType(result.variogramType),
    nugget: requireNumber(result.nugget),
    sill: requireNumber(result.sill),
    range: requireNumber(result.range),
    residuals: requireNumber(result.residuals),
  };
  if (
    result.shape !== undefined &&
    typeof result.shape === "number" &&
    Number.isFinite(result.shape)
  ) {
    fitted.shape = result.shape;
  }
  if (
    result.shape2 !== undefined &&
    typeof result.shape2 === "number" &&
    Number.isFinite(result.shape2)
  ) {
    fitted.shape2 = result.shape2;
  }
  return fitted;
}

/**
 * Compute the (isotropic) empirical variogram cloud on geographic data without fitting
 * a parametric model. Distances use Haversine (kilometers). Use the result to inspect
 * spatial structure before choosing a parametric model, or to feed a custom fitter.
 */
export function computeEmpiricalVariogram(
  options: ComputeEmpiricalVariogramOptions
): EmpiricalVariogramResult {
  const mod = requireLoadedModule();
  const {
    sampleLats,
    sampleLons,
    values,
    maxDistance,
    nBins = 12,
    estimator,
  } = options;
  try {
    const out = mod.computeEmpiricalVariogram(
      toFloat64Array(sampleLats),
      toFloat64Array(sampleLons),
      toFloat64Array(values),
      maxDistance,
      nBins,
      estimator
    );
    return mapEmpiricalVariogram(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Compute a **directional** empirical variogram on planar `(x, y)` data: only point
 * pairs whose vector makes an angle within `toleranceDeg` of `azimuthDeg` (measured
 * counter-clockwise from +x) contribute to each bin. Distances are Euclidean.
 *
 * Useful for diagnosing anisotropy: compare variograms along several azimuths to
 * estimate `majorAngleDeg` / `rangeRatio` for {@link ProjectedKriging}.
 */
export function computeDirectionalEmpiricalVariogram(
  options: ComputeDirectionalEmpiricalVariogramOptions
): EmpiricalVariogramResult {
  const mod = requireLoadedModule();
  const { xs, ys, values, maxDistance, nBins, azimuthDeg, toleranceDeg } =
    options;
  try {
    const out = mod.computeDirectionalEmpiricalVariogram(
      toFloat64Array(xs),
      toFloat64Array(ys),
      toFloat64Array(values),
      maxDistance,
      nBins,
      azimuthDeg,
      toleranceDeg
    );
    return mapEmpiricalVariogram(out);
  } catch (e) {
    throw wrapThrown(e);
  }
}

/**
 * Evaluate a nested (additive) variogram at the given distances. Components are
 * summed on the semivariance scale; the covariance is derived from `C₀ − γ(h)`.
 *
 * Useful for diagnosing composite spatial structure (e.g. a short-range exponential
 * component on top of a long-range spherical component). Nested models are not yet
 * accepted by the kriging constructors; use this entry point for offline evaluation
 * and plotting for now.
 */
export function evaluateNestedVariogram(
  components: NestedVariogramComponent[],
  distances: NumericArrayInput
): NestedVariogramEvaluation {
  const mod = requireLoadedModule();
  try {
    const out = mod.evaluateNestedVariogram(
      components.map((c) => ({
        variogramType: c.variogramType,
        nugget: c.nugget,
        sill: c.sill,
        range: c.range,
        shape: c.shape,
        shape2: c.shape2,
      })),
      toFloat64Array(distances)
    );
    const rec = asRecord(out);
    return {
      distances: requireFloat64Array(rec.distances),
      semivariances: requireFloat64Array(rec.semivariances),
      covariances: requireFloat64Array(rec.covariances),
    };
  } catch (e) {
    throw wrapThrown(e);
  }
}
