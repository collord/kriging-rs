/**
 * Internal: shared helpers for the 3-D kriging / variography / SGS facade
 * modules.
 *
 * @module
 */

import type { Anisotropy3DParams } from "../types.js";

/** GSLib identity anisotropy: no rotation, no stretch. */
export const IDENTITY_ANISOTROPY_3D: Anisotropy3DParams = {
  ang1: 0,
  ang2: 0,
  ang3: 0,
  anis1: 1,
  anis2: 1,
};

/** Resolve an optional anisotropy spec to concrete GSLib parameters. */
export function anisotropyOrIdentity(
  anisotropy: Anisotropy3DParams | undefined
): Anisotropy3DParams {
  return anisotropy ?? IDENTITY_ANISOTROPY_3D;
}
