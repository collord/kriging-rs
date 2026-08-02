/**
 * WASM module loader, `init`, public `VariogramType` proxy, `webgpuAvailable`, and the
 * shared `requireLoadedModule` / `variogramTypeToNumber` helpers used across the package.
 *
 * `init` and `VariogramType` (and `webgpuAvailable`) are re-exported from the package
 * root. The helpers in this file are consumed by every kriging class and top-level
 * function to guard against calls before `init()` has resolved.
 *
 * @module
 */

import { KrigingError } from "../errors.js";
import type { VariogramTypeName } from "../types.js";
import type { RawModule } from "./wasm-shapes.js";

import * as wasmPkg from "../../pkg/kriging_rs.js";

let rawModulePromise: Promise<RawModule> | null = null;
let rawModule: RawModule | null = null;
// Set only after mod.default() (WASM instantiation) resolves. Guards against the race where a
// constructor runs between the JS glue being available and the WASM bytes being instantiated.
let instanceReady = false;

function loadRawModule(): Promise<RawModule> {
  if (!rawModulePromise) {
    const mod = wasmPkg as RawModule;
    rawModulePromise = Promise.resolve(mod);
    rawModule = mod;
  }
  return rawModulePromise;
}

/**
 * Internal: return the loaded WASM module or throw a {@link KrigingError} with
 * code `not_loaded` if {@link init} has not yet resolved.
 */
export function requireLoadedModule(): RawModule {
  if (rawModule === null || !instanceReady) {
    throw new KrigingError(
      "WASM module is not loaded; call and await init() before using APIs",
      { code: "not_loaded" }
    );
  }
  return rawModule;
}

/**
 * Variogram type enum (use e.g. `VariogramType.Exponential`). Only safe to use **after**
 * you have called and awaited {@link init}. Accessing a property before init() will throw
 * when the value is used (e.g. when calling {@link fitVariogram} with
 * `VariogramType.Exponential`). You can pass string names like `"exponential"` instead.
 */
export const VariogramType: RawModule["WasmVariogramType"] = new Proxy(
  {} as RawModule["WasmVariogramType"],
  {
    get(_, prop) {
      return (
        requireLoadedModule().WasmVariogramType as Record<string, number>
      )[prop as string];
    },
  }
);

/**
 * Initialize the WebAssembly module. Call and await this once before using any other API.
 * Can be called with no arguments (loads WASM from the default location) or with pre-fetched
 * WASM bytes (e.g. from a custom URL or bundler) for offline or custom loading.
 *
 * @param input - Optional: `ArrayBuffer` or `Response` of WASM bytes; omit to use default loader
 * @returns Promise that resolves with `undefined` when initialization is complete
 * @example
 * await init();
 * const model = new OrdinaryKriging({ lats, lons, values, variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1, range: 100 } });
 * const pred = model.predict(37.7, -122.4);
 */
export async function init(input?: unknown): Promise<void> {
  const mod = await loadRawModule();
  if (input === undefined) {
    await mod.default();
  } else if (
    typeof input === "object" &&
    input !== null &&
    "module_or_path" in (input as Record<string, unknown>)
  ) {
    await mod.default(input);
  } else {
    await mod.default({ module_or_path: input });
  }
  rawModule = mod;
  instanceReady = true;
}

/**
 * Check whether WebGPU-backed batch prediction is available. Requires that {@link init} has
 * been awaited; returns `false` if the package was built without the `gpu` feature or if the
 * environment does not support WebGPU.
 *
 * @returns `true` if WebGPU-backed prediction is available; otherwise `false`.
 * @throws {KrigingError} If called before {@link init} has resolved.
 */
export async function webgpuAvailable(): Promise<boolean> {
  const mod = requireLoadedModule();
  if (typeof mod.webgpuAvailable !== "function") {
    return false;
  }
  return Boolean(await mod.webgpuAvailable());
}

const VariogramTypeNameToKey: Record<
  VariogramTypeName,
  keyof RawModule["WasmVariogramType"]
> = {
  spherical: "Spherical",
  exponential: "Exponential",
  gaussian: "Gaussian",
  cubic: "Cubic",
  stable: "Stable",
  matern: "Matern",
  power: "Power",
  holeeffect: "HoleEffect",
  confluenthypergeometric: "ConfluentHypergeometric",
};

/**
 * Convert a string variogram type name (or a numeric enum value already produced by the
 * WASM enum) to the numeric identifier expected by the WASM `fitVariogram` entry point.
 */
export function variogramTypeToNumber(
  variogramType: VariogramTypeName | number
): number {
  if (typeof variogramType === "number" && Number.isFinite(variogramType)) {
    return variogramType;
  }
  const name = variogramType as VariogramTypeName;
  const key = VariogramTypeNameToKey[name];
  if (!key) {
    throw new Error(`Unknown variogram type: ${String(variogramType)}`);
  }
  return requireLoadedModule().WasmVariogramType[key];
}
