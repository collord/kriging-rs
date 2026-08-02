/**
 * Internal: helpers for packing the space-time variogram parameters that all
 * `WasmSpaceTime*` factories accept.
 *
 * @module
 */

import type {
  FittedSpaceTimeVariogram,
  SpaceTimeUniversalTrend,
  SpaceTimeVariogramParams,
} from "../types.js";

/**
 * Convert a fitted space-time variogram into the parameter shape expected by
 * the kriging constructors. Drops the `residuals` field and preserves the
 * discriminated union.
 */
export function fittedToSpaceTimeVariogramParams(
  fit: FittedSpaceTimeVariogram
): SpaceTimeVariogramParams {
  if (fit.family === "productSum") {
    return {
      family: "productSum",
      spatial: fit.spatial,
      temporal: fit.temporal,
      k1: fit.k1,
      k2: fit.k2,
      k3: fit.k3,
    };
  }
  return {
    family: "separable",
    spatial: fit.spatial,
    temporal: fit.temporal,
  };
}

/** Validate the universal trend string. Throws `Error` when unknown. */
export function requireSpaceTimeUniversalTrend(
  trend: SpaceTimeUniversalTrend
): SpaceTimeUniversalTrend {
  switch (trend) {
    case "constant":
    case "linearInTime":
    case "quadraticInTime":
    case "linearInSpace":
    case "linearInSpaceAndTime":
    case "quadraticInSpaceAndTime":
      return trend;
    default: {
      const value: string = trend;
      throw new Error(`Unknown space-time universal trend: ${value}`);
    }
  }
}
