/**
 * Internal: small coercion and validation primitives used across the wrapper. Kept free
 * of dependencies on `KrigingError` so they can be used from any module without
 * introducing import cycles.
 *
 * @module
 */

import type {
  IntegerArrayInput,
  NumericArrayInput,
  VariogramTypeName,
} from "../types.js";

/** Coerce a numeric input to `Float64Array` without copying when already one. */
export function toFloat64Array(input: NumericArrayInput): Float64Array {
  return input instanceof Float64Array ? input : Float64Array.from(input);
}

/** Coerce an integer count input to `Uint32Array` without copying when already one. */
export function toUint32Array(input: IntegerArrayInput): Uint32Array {
  return input instanceof Uint32Array ? input : Uint32Array.from(input);
}

/** Narrow an unknown WASM return value to a plain object record. */
export function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null) {
    throw new Error("Expected object output from WASM");
  }
  return value as Record<string, unknown>;
}

export function requireFloat64Array(value: unknown): Float64Array {
  if (!(value instanceof Float64Array)) {
    throw new Error("Expected Float64Array output from WASM");
  }
  return value;
}

export function requireUint32Array(value: unknown): Uint32Array {
  if (!(value instanceof Uint32Array)) {
    throw new Error("Expected Uint32Array output from WASM");
  }
  return value;
}

export function requireNumber(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error("Expected finite numeric output from WASM");
  }
  return value;
}

/**
 * Like {@link requireNumber} but allows `NaN` through. Use for statistics that are
 * legitimately undefined when there is no data to summarize (e.g. RMSE / MSDR over
 * zero evaluated residuals): `NaN` is the honest answer, while a non-number
 * payload still indicates a serialization bug and throws.
 */
export function requireFiniteOrNaN(value: unknown): number {
  if (typeof value !== "number") {
    throw new Error("Expected numeric output from WASM");
  }
  if (!Number.isFinite(value) && !Number.isNaN(value)) {
    throw new Error("Expected finite or NaN numeric output from WASM");
  }
  return value;
}

export function requireVariogramType(value: unknown): VariogramTypeName {
  if (
    value === "spherical" ||
    value === "exponential" ||
    value === "gaussian" ||
    value === "cubic" ||
    value === "stable" ||
    value === "matern" ||
    value === "power" ||
    value === "holeeffect" ||
    value === "confluenthypergeometric"
  ) {
    return value;
  }
  throw new Error("Expected known variogram type from WASM");
}
