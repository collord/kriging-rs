/**
 * 3-D sequential Gaussian simulation with a streaming realization API: each
 * realization is handed to a callback and then discarded by the engine, so
 * memory stays flat regardless of `nRealizations`.
 *
 * @module
 */

import { KrigingError, wrapThrown } from "./errors.js";
import { toFloat64Array } from "./internal/convert.js";
import { requireLoadedModule } from "./internal/module.js";
import { anisotropyOrIdentity } from "./internal/three-d.js";
import type {
  GaussianSimulation3DOptions,
  GaussianSimulation3DResult,
} from "./types.js";

/**
 * Run 3-D sequential Gaussian simulation conditioned on the samples,
 * streaming realizations through `options.onRealization` in realization-index
 * order. The grid passed to the callback is a fresh `Float64Array` (safe to
 * retain), laid out x-fastest: linear index `i + nx * j + nx * ny * k`.
 *
 * Return `true` from the callback to stop early; the function then returns
 * `{ aborted: true }`. Exceptions thrown by the callback abort the stream and
 * are rethrown to the caller.
 *
 * Cells whose kriging system fails are marked `NaN` rather than silently
 * filled; check for `NaN` when consuming realizations from challenging data.
 *
 * @param options - Samples, variogram, grid definition, seed, and callback.
 * @returns `{ aborted }` — whether the callback stopped the stream early.
 * @throws {KrigingError} When the WASM module is not loaded or inputs are
 *   invalid (e.g. code `insufficient_data` for too few samples).
 */
export function gaussianSimulation3D(
  options: GaussianSimulation3DOptions
): GaussianSimulation3DResult {
  const mod = requireLoadedModule();
  const fn = mod.gaussianSimulation3D;
  if (!fn) {
    throw new KrigingError(
      "gaussianSimulation3D is not available; rebuild the WASM package",
      { code: "backend_unavailable" }
    );
  }
  const anis = anisotropyOrIdentity(options.anisotropy);
  const { grid } = options;
  const seed = options.seed ?? 0n;

  // Wrap the user callback so a thrown exception is preserved across the WASM
  // boundary (the Rust side only reports "closure threw") and so only an
  // explicit `true` aborts the stream.
  let thrown: unknown;
  let sawThrow = false;
  let userAborted = false;
  const wrapped = (idx: number, realization: Float64Array): boolean => {
    try {
      if (options.onRealization(idx, realization) === true) {
        userAborted = true;
        return true;
      }
      return false;
    } catch (e) {
      thrown = e;
      sawThrow = true;
      return true;
    }
  };

  try {
    fn(
      toFloat64Array(options.xs),
      toFloat64Array(options.ys),
      toFloat64Array(options.zs),
      toFloat64Array(options.values),
      anis.ang1,
      anis.ang2,
      anis.ang3,
      anis.anis1,
      anis.anis2,
      options.variogram.variogramType,
      options.variogram.nugget,
      options.variogram.sill,
      options.variogram.range,
      options.variogram.shape,
      grid.nx,
      grid.ny,
      grid.nz,
      grid.originX,
      grid.originY,
      grid.originZ,
      grid.spacingX,
      grid.spacingY,
      grid.spacingZ,
      typeof seed === "bigint" ? seed : BigInt(Math.floor(seed)),
      options.nRealizations,
      options.scoreSpace ?? false,
      wrapped
    );
  } catch (e) {
    if (sawThrow) throw thrown;
    const err = wrapThrown(e);
    if (userAborted && err.code === "callback_aborted") {
      return { aborted: true };
    }
    throw err;
  }
  return { aborted: false };
}
