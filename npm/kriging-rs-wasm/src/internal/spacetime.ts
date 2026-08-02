/**
 * Internal: helpers for packing the space-time variogram parameters that all
 * `WasmSpaceTime*` factories accept.
 *
 * @module
 */

import type {
  FittedSpaceTimeVariogram,
  SpaceTimeUniversalTrend,
  SpaceTimeVariogramFamily,
  SpaceTimeVariogramParams,
} from "../types.js";

/**
 * Positional layout of the space-time variogram arguments in the WASM `fromArrays`
 * factories. Keeping this in one place so adding a future parameter only touches the
 * wrapper glue. The `family` field uses the same {@link SpaceTimeVariogramFamily}
 * camelCase strings as the public TS API.
 */
export interface SpaceTimeVariogramArgs {
  family: SpaceTimeVariogramFamily;
  spatialType: string;
  spatialNugget: number;
  spatialSill: number;
  spatialRange: number;
  spatialShape: number | undefined;
  spatialShape2: number | undefined;
  temporalType: string;
  temporalNugget: number;
  temporalSill: number;
  temporalRange: number;
  temporalShape: number | undefined;
  temporalShape2: number | undefined;
  k1: number | undefined;
  k2: number | undefined;
  k3: number | undefined;
}

export function packSpaceTimeVariogram(
  variogram: SpaceTimeVariogramParams
): SpaceTimeVariogramArgs {
  const base = {
    family: variogram.family,
    spatialType: variogram.spatial.variogramType,
    spatialNugget: variogram.spatial.nugget,
    spatialSill: variogram.spatial.sill,
    spatialRange: variogram.spatial.range,
    spatialShape: variogram.spatial.shape,
    spatialShape2: variogram.spatial.shape2,
    temporalType: variogram.temporal.variogramType,
    temporalNugget: variogram.temporal.nugget,
    temporalSill: variogram.temporal.sill,
    temporalRange: variogram.temporal.range,
    temporalShape: variogram.temporal.shape,
    temporalShape2: variogram.temporal.shape2,
  };
  if (variogram.family === "productSum") {
    return {
      ...base,
      k1: variogram.k1,
      k2: variogram.k2,
      k3: variogram.k3,
    };
  }
  return {
    ...base,
    k1: undefined,
    k2: undefined,
    k3: undefined,
  };
}

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
