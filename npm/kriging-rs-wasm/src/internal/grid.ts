/**
 * Internal: grid construction/reshape helpers and a small helper for turning a
 * fitted-variogram record into `VariogramParams` with an optional nugget override.
 *
 * @module
 */

import type {
  FittedVariogram,
  PredictGridOptions,
  VariogramParams,
} from "../types.js";

/**
 * Build flat lat/lon arrays for the centers of a rectangular grid defined by bounds
 * and cell counts. Row-major order: first row = south, last row = north; within each
 * row, west to east.
 */
export function buildGridLatsLons(options: PredictGridOptions): {
  lats: Float64Array;
  lons: Float64Array;
} {
  const { west, south, east, north, xCells, yCells } = options;
  const nRows = Math.max(1, Math.floor(yCells));
  const nCols = Math.max(1, Math.floor(xCells));
  const n = nRows * nCols;
  const lats = new Float64Array(n);
  const lons = new Float64Array(n);
  const latStep = (north - south) / nRows;
  const lonStep = (east - west) / nCols;
  let k = 0;
  for (let j = 0; j < nRows; j++) {
    const lat = south + (j + 0.5) * latStep;
    for (let i = 0; i < nCols; i++) {
      lats[k] = lat;
      lons[k] = west + (i + 0.5) * lonStep;
      k++;
    }
  }
  return { lats, lons };
}

/** Reshape a flat row-major `Float64Array` into a `[nRows][nCols]` 2D array. */
export function reshapeFlatToGrid(
  flat: Float64Array,
  nRows: number,
  nCols: number
): number[][] {
  const grid: number[][] = [];
  for (let j = 0; j < nRows; j++) {
    const row: number[] = [];
    for (let i = 0; i < nCols; i++) {
      row.push(flat[j * nCols + i]);
    }
    grid.push(row);
  }
  return grid;
}

/**
 * Turn a {@link FittedVariogram} into {@link VariogramParams}, applying an optional
 * nugget override. Keeps the other fields (type, sill, range, shape) unchanged.
 */
export function fittedToVariogramParamsWithNuggetOverride(
  f: FittedVariogram,
  nuggetOverride: number | undefined
): VariogramParams {
  return {
    variogramType: f.variogramType,
    nugget: nuggetOverride ?? f.nugget,
    sill: f.sill,
    range: f.range,
    shape: f.shape,
    shape2: f.shape2,
  };
}
