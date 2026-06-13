/**
 * 3-D variography: gamv-style directional experimental variograms and
 * anisotropic spherical model fitting (joint, two-stage, fixed-nugget, and
 * single-axis 1-D).
 *
 * The typical workflow mirrors GSLib practice: compute three directional
 * experimental variograms along the suspected major / minor / vertical axes
 * with {@link computeDirectionalVariogram3D}, then fit one anisotropic
 * spherical model across them with {@link fitSpherical3DJoint} (or the
 * two-stage / fixed-nugget variants for noisy drillhole-style data).
 *
 * @module
 */

import { KrigingError, wrapThrown } from "./errors.js";
import { toFloat64Array } from "./internal/convert.js";
import {
  mapDirectionalVariogram3DResult,
  mapFittedSpherical1D,
  mapFittedSpherical3D,
} from "./internal/mappers.js";
import { requireLoadedModule } from "./internal/module.js";
import type { RawModule } from "./internal/wasm-shapes.js";
import type {
  AxisVariogramInput,
  DirectionalVariogram3DOptions,
  DirectionalVariogram3DResult,
  FitSpherical1DOptions,
  FitSpherical3DFixedNuggetOptions,
  FitSpherical3DOptions,
  FitSpherical3DTwoStageOptions,
  FittedSpherical1D,
  FittedSpherical3D,
} from "./types.js";

/** Effectively-unbounded bandwidth default (matches the omnidirectional recipe). */
const UNBOUNDED_BANDWIDTH = 1.0e10;

function require3DFn<K extends keyof RawModule>(
  mod: RawModule,
  name: K
): NonNullable<RawModule[K]> {
  const fn = mod[name];
  if (!fn) {
    throw new KrigingError(
      `${String(name)} is not available; rebuild the WASM package`,
      { code: "backend_unavailable" }
    );
  }
  return fn;
}

/**
 * Compute a 3-D experimental directional variogram with GSLib `gamv`-style
 * lag-centred binning and a cone + bandwidth direction filter.
 *
 * For an omnidirectional variogram pass `azimuthToleranceDeg: 90` and
 * `dipToleranceDeg: 90` (leaving the bandwidths unbounded); the engine
 * detects this and double-counts pairs to match `gamv`.
 *
 * @param options - Sample data, lag definition, and direction filter.
 * @returns Parallel arrays per non-empty lag (empty lags are omitted).
 * @throws {KrigingError} When the WASM module is not loaded or inputs are invalid.
 */
export function computeDirectionalVariogram3D(
  options: DirectionalVariogram3DOptions
): DirectionalVariogram3DResult {
  const mod = requireLoadedModule();
  const fn = require3DFn(mod, "computeDirectionalVariogram3D");
  const {
    lagDistance,
    lagTolerance = lagDistance / 2,
    nLags,
    azimuthDeg,
    azimuthToleranceDeg = 22.5,
    dipDeg,
    dipToleranceDeg = 22.5,
    horizontalBandwidth = UNBOUNDED_BANDWIDTH,
    verticalBandwidth = UNBOUNDED_BANDWIDTH,
  } = options;
  let out: unknown;
  try {
    out = fn(
      toFloat64Array(options.xs),
      toFloat64Array(options.ys),
      toFloat64Array(options.zs),
      toFloat64Array(options.values),
      lagDistance,
      lagTolerance,
      nLags,
      azimuthDeg,
      azimuthToleranceDeg,
      horizontalBandwidth,
      dipDeg,
      dipToleranceDeg,
      verticalBandwidth
    );
  } catch (e) {
    throw wrapThrown(e);
  }
  return mapDirectionalVariogram3DResult(out);
}

/** Flatten one axis of experimental variogram input into typed arrays. */
function axisArrays(
  axis: AxisVariogramInput
): [Float64Array, Float64Array, Float64Array] {
  return [
    toFloat64Array(axis.distances),
    toFloat64Array(axis.semivariances),
    toFloat64Array(axis.nPairs),
  ];
}

/**
 * Joint least-squares fit of a 3-D anisotropic spherical variogram across
 * three axis-aligned experimental variograms (major / minor / vertical).
 * Bins are weighted by pair count.
 *
 * @returns One nugget and sill with per-axis ranges; see {@link FittedSpherical3D}.
 * @throws {KrigingError} When the WASM module is not loaded or inputs are invalid.
 */
export function fitSpherical3DJoint(
  options: FitSpherical3DOptions
): FittedSpherical3D {
  const mod = requireLoadedModule();
  const fn = require3DFn(mod, "fitSpherical3DJoint");
  let out: unknown;
  try {
    out = fn(
      ...axisArrays(options.major),
      ...axisArrays(options.minor),
      ...axisArrays(options.vertical)
    );
  } catch (e) {
    throw wrapThrown(e);
  }
  return mapFittedSpherical3D(out);
}

/**
 * Two-stage spherical fit: nugget, sill, and vertical range are fitted on the
 * vertical experimental alone, then held while the horizontal ranges are
 * fitted. Avoids the joint fit's tendency to drive the nugget toward zero
 * when horizontal axes have noisy short-lag bins (typical drillhole pattern).
 *
 * @throws {KrigingError} When the WASM module is not loaded or inputs are invalid.
 */
export function fitSpherical3DTwoStage(
  options: FitSpherical3DTwoStageOptions
): FittedSpherical3D {
  const mod = requireLoadedModule();
  const fn = require3DFn(mod, "fitSpherical3DTwoStage");
  let out: unknown;
  try {
    out = fn(
      ...axisArrays(options.major),
      ...axisArrays(options.minor),
      ...axisArrays(options.vertical),
      options.dataVariance ?? 0
    );
  } catch (e) {
    throw wrapThrown(e);
  }
  return mapFittedSpherical3D(out);
}

/**
 * Refit the 3-D spherical model with the nugget held at a user-supplied value
 * (sill plus the three ranges are fitted). Used after a two-stage fit when
 * the nugget has been read off the vertical's short-lag intercept manually.
 *
 * @throws {KrigingError} When the WASM module is not loaded or inputs are invalid.
 */
export function fitSpherical3DFixedNugget(
  options: FitSpherical3DFixedNuggetOptions
): FittedSpherical3D {
  const mod = requireLoadedModule();
  const fn = require3DFn(mod, "fitSpherical3DFixedNugget");
  let out: unknown;
  try {
    out = fn(
      ...axisArrays(options.major),
      ...axisArrays(options.minor),
      ...axisArrays(options.vertical),
      options.nugget
    );
  } catch (e) {
    throw wrapThrown(e);
  }
  return mapFittedSpherical3D(out);
}

/**
 * Fit a 1-D spherical model to a single precomputed experimental variogram
 * (e.g. one output of {@link computeDirectionalVariogram3D}). Useful for
 * scoring candidate orientations by fitted range during an anisotropy search.
 *
 * @throws {KrigingError} When the WASM module is not loaded or inputs are invalid.
 */
export function fitSpherical1D(
  options: FitSpherical1DOptions
): FittedSpherical1D {
  const mod = requireLoadedModule();
  const fn = require3DFn(mod, "fitSpherical1D");
  let out: unknown;
  try {
    out = fn(...axisArrays(options));
  } catch (e) {
    throw wrapThrown(e);
  }
  return mapFittedSpherical1D(out);
}
