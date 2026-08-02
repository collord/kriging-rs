/**
 * Internal: TypeScript shapes for the WASM-exposed classes, functions, and options.
 * Purely type-level; not part of the public API.
 *
 * @module
 */

/** WASM ordinary kriging instance shape. */
export interface WasmOrdinaryInstance {
  predict(lat: number, lon: number): unknown;
  predictBatch(lats: Float64Array, lons: Float64Array): unknown;
  predictBatchArrays(lats: Float64Array, lons: Float64Array): unknown;
  predictGridArrays(
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    xCells: number,
    yCells: number
  ): unknown;
  setNeighborhood(
    maxNeighbors: number | undefined,
    maxRadius: number | undefined
  ): void;
  neighborhood(): unknown;
  free?: () => void;
  predictBatchGpu?(lats: Float64Array, lons: Float64Array): Promise<unknown>;
  predictBatchGpuOrCpu?(
    lats: Float64Array,
    lons: Float64Array
  ): Promise<unknown>;
}

/** WASM simple kriging instance shape. */
export interface WasmSimpleInstance {
  predict(lat: number, lon: number): unknown;
  predictBatch(lats: Float64Array, lons: Float64Array): unknown;
  predictBatchArrays(lats: Float64Array, lons: Float64Array): unknown;
  mean(): number;
  free?: () => void;
}

/** WASM universal kriging instance shape. */
export interface WasmUniversalInstance {
  predict(lat: number, lon: number): unknown;
  predictBatch(lats: Float64Array, lons: Float64Array): unknown;
  predictBatchArrays(lats: Float64Array, lons: Float64Array): unknown;
  free?: () => void;
}

/** WASM projected kriging instance shape. */
export interface WasmProjectedInstance {
  predict(x: number, y: number): unknown;
  predictBatch(xs: Float64Array, ys: Float64Array): unknown;
  predictBatchArrays(xs: Float64Array, ys: Float64Array): unknown;
  free?: () => void;
}

/** WASM projected binomial kriging instance shape. */
export interface WasmBinomialProjectedInstance {
  predict(x: number, y: number): unknown;
  predictBatch(xs: Float64Array, ys: Float64Array): unknown;
  predictBatchArrays(xs: Float64Array, ys: Float64Array): unknown;
  getBuildNotes(): unknown;
  free?: () => void;
}

/** WASM binomial kriging instance shape. */
export interface WasmBinomialInstance {
  predict(lat: number, lon: number): unknown;
  predictBatch(lats: Float64Array, lons: Float64Array): unknown;
  predictBatchArrays(lats: Float64Array, lons: Float64Array): unknown;
  getBuildNotes(): unknown;
  predictGridArrays(
    xMin: number,
    xMax: number,
    yMin: number,
    yMax: number,
    xCells: number,
    yCells: number
  ): unknown;
  free?: () => void;
  predictBatchGpu?(lats: Float64Array, lons: Float64Array): Promise<unknown>;
  predictBatchGpuOrCpu?(
    lats: Float64Array,
    lons: Float64Array
  ): Promise<unknown>;
}

/** Shape passed to WASM (plain arrays for serde deserialization). */
export interface OrdinaryKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  values: number[];
  variogram: {
    variogramType: string;
    nugget: number;
    sill: number;
    range: number;
    shape?: number;
  };
}

export interface BinomialKrigingOptionsWasm {
  lats: number[];
  lons: number[];
  successes: number[];
  trials: number[];
  variogram: {
    variogramType: string;
    nugget: number;
    sill: number;
    range: number;
    shape?: number;
  };
}

export interface BinomialKrigingWithPriorOptionsWasm extends BinomialKrigingOptionsWasm {
  prior: { alpha: number; beta: number };
}

/** Shape of the raw `pkg/kriging_rs.js` glue module, typed for TS consumers. */
export type RawModule = {
  default: (input?: unknown) => Promise<unknown>;
  WasmOrdinaryKriging: {
    new (options: OrdinaryKrigingOptionsWasm): WasmOrdinaryInstance;
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      values: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmOrdinaryInstance;
  };
  WasmBinomialKriging: {
    new (options: BinomialKrigingOptionsWasm): WasmBinomialInstance;
    newWithPrior(
      options: BinomialKrigingWithPriorOptionsWasm
    ): WasmBinomialInstance;
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmBinomialInstance;
    fromPrecomputedLogits(
      lats: Float64Array,
      lons: Float64Array,
      logits: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmBinomialInstance;
  };
  WasmSimpleKriging?: {
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      values: Float64Array,
      mean: number,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmSimpleInstance;
  };
  WasmUniversalKriging?: {
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      values: Float64Array,
      trend: string,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape?: number
    ): WasmUniversalInstance;
  };
  WasmProjectedKriging?: {
    fromArrays(
      xs: Float64Array,
      ys: Float64Array,
      values: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number
    ): WasmProjectedInstance;
  };
  WasmBinomialProjectedKriging?: {
    fromArrays(
      xs: Float64Array,
      ys: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number
    ): WasmBinomialProjectedInstance;
    fromArraysWithPrior(
      xs: Float64Array,
      ys: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number,
      priorAlpha: number,
      priorBeta: number
    ): WasmBinomialProjectedInstance;
    fromPrecomputedLogits(
      xs: Float64Array,
      ys: Float64Array,
      logits: Float64Array,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      majorAngleDeg: number,
      rangeRatio: number
    ): WasmBinomialProjectedInstance;
  };
  WasmVariogramType: {
    readonly Spherical: number;
    readonly Exponential: number;
    readonly Gaussian: number;
    readonly Cubic: number;
    readonly Stable: number;
    readonly Matern: number;
    readonly Power: number;
    readonly HoleEffect: number;
    readonly ConfluentHypergeometric: number;
  };
  fitVariogram: (
    sampleLats: Float64Array,
    sampleLons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    variogramType: number,
    estimator?: string
  ) => unknown;
  computeEmpiricalVariogram: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    maxDistance: number | undefined,
    nBins: number,
    estimator?: string
  ) => unknown;
  computeDirectionalEmpiricalVariogram: (
    xs: Float64Array,
    ys: Float64Array,
    values: Float64Array,
    maxDistance: number,
    nBins: number,
    azimuthDeg: number,
    toleranceDeg: number
  ) => unknown;
  leaveOneOut: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  kFold: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    k: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  leaveOneOutSimple: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    mean: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  kFoldSimple: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    mean: number,
    k: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  leaveOneOutUniversal: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    trend: string,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  kFoldUniversal: (
    lats: Float64Array,
    lons: Float64Array,
    values: Float64Array,
    trend: string,
    k: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  leaveOneOutProjected: (
    xs: Float64Array,
    ys: Float64Array,
    values: Float64Array,
    majorAngleDeg: number,
    rangeRatio: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  kFoldProjected: (
    xs: Float64Array,
    ys: Float64Array,
    values: Float64Array,
    majorAngleDeg: number,
    rangeRatio: number,
    k: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number
  ) => unknown;
  leaveOneOutBinomial: (
    lats: Float64Array,
    lons: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number,
    priorAlpha?: number,
    priorBeta?: number
  ) => unknown;
  kFoldBinomial: (
    lats: Float64Array,
    lons: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    k: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number,
    priorAlpha?: number,
    priorBeta?: number
  ) => unknown;
  leaveOneOutBinomialProjected: (
    xs: Float64Array,
    ys: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    majorAngleDeg: number,
    rangeRatio: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number,
    priorAlpha?: number,
    priorBeta?: number
  ) => unknown;
  kFoldBinomialProjected: (
    xs: Float64Array,
    ys: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    majorAngleDeg: number,
    rangeRatio: number,
    k: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape?: number,
    priorAlpha?: number,
    priorBeta?: number
  ) => unknown;
  conditionalSimulate: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateSimple: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    mean: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateUniversal: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    trend: string,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateProjected: (
    conditioningXs: Float64Array,
    conditioningYs: Float64Array,
    conditioningValues: Float64Array,
    targetXs: Float64Array,
    targetYs: Float64Array,
    majorAngleDeg: number,
    rangeRatio: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateBinomial: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateMany: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    nRealizations: number,
    baseSeed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateManyBinomial: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    nRealizations: number,
    baseSeed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateBinomialProjected: (
    conditioningXs: Float64Array,
    conditioningYs: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    targetXs: Float64Array,
    targetYs: Float64Array,
    majorAngleDeg: number,
    rangeRatio: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateManyBinomialProjected: (
    conditioningXs: Float64Array,
    conditioningYs: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    targetXs: Float64Array,
    targetYs: Float64Array,
    majorAngleDeg: number,
    rangeRatio: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    shape2: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    nRealizations: number,
    baseSeed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  evaluateNestedVariogram: (
    components: unknown,
    distances: Float64Array
  ) => unknown;
  aggregatePolygonsOverEnsemble: (
    samples: Float64Array,
    nRealizations: number,
    nTargets: number,
    polygonIndices: Uint32Array,
    polygonWeights: Float64Array,
    polygonOffsets: Uint32Array,
    quantiles: Float64Array
  ) => unknown;
  webgpuAvailable?: () => Promise<unknown>;
  leaveOneOutSpaceTime: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined
  ) => unknown;
  kFoldSpaceTime: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    k: number,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined
  ) => unknown;
  leaveOneOutSpaceTimeSimple: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    mean: number,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined
  ) => unknown;
  kFoldSpaceTimeSimple: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    mean: number,
    k: number,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined
  ) => unknown;
  leaveOneOutSpaceTimeUniversal: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    trend: string,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined
  ) => unknown;
  kFoldSpaceTimeUniversal: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    trend: string,
    k: number,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined
  ) => unknown;
  leaveOneOutSpaceTimeBinomial: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined
  ) => unknown;
  kFoldSpaceTimeBinomial: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    k: number,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined
  ) => unknown;
  conditionalSimulateSpaceTime: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningTimes: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    targetTimes: Float64Array,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateSpaceTimeSimple: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningTimes: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    targetTimes: Float64Array,
    mean: number,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateSpaceTimeUniversal: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningTimes: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    targetTimes: Float64Array,
    trend: string,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateSpaceTimeBinomial: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningTimes: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    targetTimes: Float64Array,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    seed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateSpaceTimeMany: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningTimes: Float64Array,
    conditioningValues: Float64Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    targetTimes: Float64Array,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    nRealizations: number,
    baseSeed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  conditionalSimulateSpaceTimeManyBinomial: (
    conditioningLats: Float64Array,
    conditioningLons: Float64Array,
    conditioningTimes: Float64Array,
    successes: Uint32Array,
    trials: Uint32Array,
    targetLats: Float64Array,
    targetLons: Float64Array,
    targetTimes: Float64Array,
    family: string,
    spatialType: string,
    spatialNugget: number,
    spatialSill: number,
    spatialRange: number,
    spatialShape: number | undefined,
    spatialShape2: number | undefined,
    temporalType: string,
    temporalNugget: number,
    temporalSill: number,
    temporalRange: number,
    temporalShape: number | undefined,
    temporalShape2: number | undefined,
    k1: number | undefined,
    k2: number | undefined,
    k3: number | undefined,
    priorAlpha: number | undefined,
    priorBeta: number | undefined,
    nRealizations: number,
    baseSeed: bigint,
    targetOrder?: Uint32Array
  ) => unknown;
  WasmSpaceTimeOrdinaryKriging?: {
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      values: Float64Array,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      spatialShape2: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      temporalShape2: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmSpaceTimeInstance;
  };
  WasmSpaceTimeSimpleKriging?: {
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      values: Float64Array,
      mean: number,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      spatialShape2: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      temporalShape2: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmSpaceTimeInstance;
  };
  WasmSpaceTimeUniversalKriging?: {
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      values: Float64Array,
      trend: string,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      spatialShape2: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      temporalShape2: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmSpaceTimeInstance;
  };
  WasmSpaceTimeBinomialKriging?: {
    fromArrays(
      lats: Float64Array,
      lons: Float64Array,
      times: Float64Array,
      successes: Uint32Array,
      trials: Uint32Array,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      spatialShape2: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      temporalShape2: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmSpaceTimeBinomialInstance;
  };
  WasmSpaceTimeOrdinaryProjectedKriging?: {
    fromArrays(
      xs: Float64Array,
      ys: Float64Array,
      times: Float64Array,
      values: Float64Array,
      majorAngleDeg: number,
      rangeRatio: number,
      family: string,
      spatialType: string,
      spatialNugget: number,
      spatialSill: number,
      spatialRange: number,
      spatialShape: number | undefined,
      spatialShape2: number | undefined,
      temporalType: string,
      temporalNugget: number,
      temporalSill: number,
      temporalRange: number,
      temporalShape: number | undefined,
      temporalShape2: number | undefined,
      k1: number | undefined,
      k2: number | undefined,
      k3: number | undefined
    ): WasmSpaceTimeInstance;
  };
  wasmComputeEmpiricalSpaceTimeVariogram: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    maxSpatialDistance: number | undefined,
    maxTemporalLag: number | undefined,
    nSpatialBins: number,
    nTemporalBins: number,
    estimator: string
  ) => unknown;
  wasmFitSpaceTimeVariogram: (
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array,
    values: Float64Array,
    maxSpatialDistance: number | undefined,
    maxTemporalLag: number | undefined,
    nSpatialBins: number,
    nTemporalBins: number,
    estimator: string,
    family: string,
    spatialModel: string,
    temporalModel: string
  ) => unknown;
  WasmOrdinaryKriging3D?: {
    fromArrays(
      xs: Float64Array,
      ys: Float64Array,
      zs: Float64Array,
      values: Float64Array,
      ang1: number,
      ang2: number,
      ang3: number,
      anis1: number,
      anis2: number,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined,
      maxRadius: number | undefined,
      maxNeighbors: number | undefined
    ): WasmKriging3DInstance;
  };
  WasmSimpleKriging3D?: {
    fromArrays(
      xs: Float64Array,
      ys: Float64Array,
      zs: Float64Array,
      values: Float64Array,
      mean: number,
      ang1: number,
      ang2: number,
      ang3: number,
      anis1: number,
      anis2: number,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined
    ): WasmKriging3DInstance;
  };
  WasmUniversalKriging3D?: {
    fromArraysLinear(
      xs: Float64Array,
      ys: Float64Array,
      zs: Float64Array,
      values: Float64Array,
      ang1: number,
      ang2: number,
      ang3: number,
      anis1: number,
      anis2: number,
      variogramType: string,
      nugget: number,
      sill: number,
      range: number,
      shape: number | undefined
    ): WasmKriging3DInstance;
  };
  gaussianSimulation3D?: (
    sampleXs: Float64Array,
    sampleYs: Float64Array,
    sampleZs: Float64Array,
    sampleValues: Float64Array,
    ang1: number,
    ang2: number,
    ang3: number,
    anis1: number,
    anis2: number,
    variogramType: string,
    nugget: number,
    sill: number,
    range: number,
    shape: number | undefined,
    nx: number,
    ny: number,
    nz: number,
    originX: number,
    originY: number,
    originZ: number,
    spacingX: number,
    spacingY: number,
    spacingZ: number,
    seed: bigint,
    nRealizations: number,
    scoreSpace: boolean,
    onRealization: (idx: number, grid: Float64Array) => unknown
  ) => void;
  computeDirectionalVariogram3D?: (
    xs: Float64Array,
    ys: Float64Array,
    zs: Float64Array,
    values: Float64Array,
    xlag: number,
    xltol: number,
    nLags: number,
    azmDeg: number,
    atolDeg: number,
    bandh: number,
    dipDeg: number,
    dtolDeg: number,
    bandv: number
  ) => unknown;
  fitSpherical3DJoint?: (
    majorDistances: Float64Array,
    majorSemivariances: Float64Array,
    majorNPairs: Float64Array,
    minorDistances: Float64Array,
    minorSemivariances: Float64Array,
    minorNPairs: Float64Array,
    verticalDistances: Float64Array,
    verticalSemivariances: Float64Array,
    verticalNPairs: Float64Array
  ) => unknown;
  fitSpherical3DTwoStage?: (
    majorDistances: Float64Array,
    majorSemivariances: Float64Array,
    majorNPairs: Float64Array,
    minorDistances: Float64Array,
    minorSemivariances: Float64Array,
    minorNPairs: Float64Array,
    verticalDistances: Float64Array,
    verticalSemivariances: Float64Array,
    verticalNPairs: Float64Array,
    dataVariance: number
  ) => unknown;
  fitSpherical3DFixedNugget?: (
    majorDistances: Float64Array,
    majorSemivariances: Float64Array,
    majorNPairs: Float64Array,
    minorDistances: Float64Array,
    minorSemivariances: Float64Array,
    minorNPairs: Float64Array,
    verticalDistances: Float64Array,
    verticalSemivariances: Float64Array,
    verticalNPairs: Float64Array,
    nugget: number
  ) => unknown;
  fitSpherical1D?: (
    distances: Float64Array,
    semivariances: Float64Array,
    nPairs: Float64Array
  ) => unknown;
};

/**
 * WASM 3-D kriging instance shape, shared by the ordinary / simple / universal
 * classes (they expose identical predict surfaces).
 */
export interface WasmKriging3DInstance {
  predict(x: number, y: number, z: number): unknown;
  predictBatch(xs: Float64Array, ys: Float64Array, zs: Float64Array): unknown;
  free?: () => void;
}

/** WASM space-time continuous kriging instance shape (ordinary / simple / universal / projected). */
export interface WasmSpaceTimeInstance {
  predict(a: number, b: number, time: number): unknown;
  predictBatch?(a: Float64Array, b: Float64Array, times: Float64Array): unknown;
  predictBatchArrays(
    a: Float64Array,
    b: Float64Array,
    times: Float64Array
  ): unknown;
  free?: () => void;
}

/** WASM space-time binomial kriging instance shape. */
export interface WasmSpaceTimeBinomialInstance {
  predict(lat: number, lon: number, time: number): unknown;
  predictBatch(
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array
  ): unknown;
  predictBatchArrays(
    lats: Float64Array,
    lons: Float64Array,
    times: Float64Array
  ): unknown;
  getBuildNotes(): unknown;
  free?: () => void;
}
