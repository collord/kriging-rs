declare module "../pkg/kriging_rs.js" {
  const init: (input?: unknown) => Promise<unknown>;
  export default init;

  export const initSync: (module: unknown) => unknown;
  /** Use `fromArrays` for the normal fast path; see internal/wasm-shapes.ts. */
  export const WasmOrdinaryKriging: {
    new (options: unknown): unknown;
    fromArrays(...args: unknown[]): unknown;
  };
  /** Factory methods; instance shape includes getBuildNotes and predictGridArrays. */
  export const WasmBinomialKriging: {
    new (options: unknown): unknown;
    newWithPrior(options: unknown): unknown;
    fromArrays(...args: unknown[]): unknown;
    fromPrecomputedLogits(...args: unknown[]): unknown;
  };
  export const WasmVariogramType: {
    readonly Spherical: number;
    readonly Exponential: number;
    readonly Gaussian: number;
    readonly Cubic: number;
    readonly Stable: number;
    readonly Matern: number;
  };
  export const fitVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number
  ) => unknown;
  export const webgpuAvailable: (...args: unknown[]) => Promise<unknown>;

  /** 3-D kriging classes; instance shapes in internal/wasm-shapes.ts. */
  export const WasmOrdinaryKriging3D: {
    fromArrays(...args: unknown[]): unknown;
  };
  export const WasmSimpleKriging3D: {
    fromArrays(...args: unknown[]): unknown;
  };
  export const WasmUniversalKriging3D: {
    fromArraysLinear(...args: unknown[]): unknown;
  };
  export const gaussianSimulation3D: (...args: unknown[]) => void;
  export const computeDirectionalVariogram3D: (...args: unknown[]) => unknown;
  export const fitSpherical3DJoint: (...args: unknown[]) => unknown;
  export const fitSpherical3DTwoStage: (...args: unknown[]) => unknown;
  export const fitSpherical3DFixedNugget: (...args: unknown[]) => unknown;
  export const fitSpherical1D: (...args: unknown[]) => unknown;
}
