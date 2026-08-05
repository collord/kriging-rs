import {
  BinomialKriging,
  OrdinaryKriging,
  OrdinaryKriging3D,
  SimpleKriging3D,
  UniversalKriging3D,
  computeDirectionalVariogram3D,
  fitSpherical1D,
  fitSpherical3DFixedNugget,
  fitSpherical3DJoint,
  fitSpherical3DTwoStage,
  gaussianSimulation3D,
  SpaceTimeBinomialKriging,
  SpaceTimeOrdinaryKriging,
  SpaceTimeProjectedOrdinaryKriging,
  SpaceTimeSimpleKriging,
  SpaceTimeUniversalKriging,
  computeEmpiricalSpaceTimeVariogram,
  conditionalSimulateSpaceTime,
  conditionalSimulateSpaceTimeBinomial,
  conditionalSimulateSpaceTimeSimple,
  conditionalSimulateSpaceTimeUniversal,
  fitSpaceTimeVariogram,
  fitVariogram,
  init,
  interpolateOrdinaryToGrid,
  interpolateBinomialToGrid,
  kFoldSpaceTime,
  kFoldSpaceTimeBinomial,
  kFoldSpaceTimeSimple,
  kFoldSpaceTimeUniversal,
  leaveOneOutSpaceTime,
  leaveOneOutSpaceTimeBinomial,
  leaveOneOutSpaceTimeSimple,
  leaveOneOutSpaceTimeUniversal,
  VariogramType,
  type BinomialBatchArrayOutput,
  type BinomialBuildNotes,
  type BinomialCvResult,
  type BinomialPrediction,
  type BinomialGridOutput,
  type InterpolateBinomialToGridResult,
  type BinomialSimulationResult,
  type CvResult,
  type EmpiricalSpaceTimeVariogramResult,
  type FitSpaceTimeVariogramResult,
  type Batch3DArrayOutput,
  type DirectionalVariogram3DResult,
  type FittedSpherical1D,
  type FittedSpherical3D,
  type GaussianSimulation3DResult,
  type OrdinaryBatchArrayOutput,
  type OrdinaryPrediction,
  type OrdinaryGridOutput,
  type Prediction3D,
  type PredictGridOptions,
  type SpaceTimeVariogramParams,
  type VariogramTypeName,
} from "../src/index.js";

// Contract: init() must return Promise<void>
type _InitReturn = ReturnType<typeof init>;
const _initReturnsVoid: _InitReturn extends Promise<void> ? true : false = true;

type IsAny<T> = 0 extends 1 & T ? true : false;
type AssertNotAny<T> = IsAny<T> extends true ? never : true;

const variogram: VariogramTypeName = "gaussian";
const lats = new Float64Array([0, 1, 2]);
const lons = new Float64Array([0, 1, 2]);
const values = new Float64Array([3, 4, 5]);

const ordinary = new OrdinaryKriging({
  lats,
  lons,
  values,
  variogram: { variogramType: variogram, nugget: 0.01, sill: 1.0, range: 100 },
});
const pred = ordinary.predict(0.5, 0.5);
const batch = ordinary.predictBatch(lats, lons);
const batchArrays = ordinary.predictBatchArrays(lats, lons);

const _predNotAny: AssertNotAny<typeof pred> = true;
const _batchNotAny: AssertNotAny<typeof batch> = true;
const _batchArraysNotAny: AssertNotAny<typeof batchArrays> = true;
const _predType: OrdinaryPrediction = pred;
const _batchItemType: OrdinaryPrediction = batch[0];
const _batchArraysType: OrdinaryBatchArrayOutput = batchArrays;

const fit = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: VariogramType.Gaussian,
  nBins: 12,
});
const fitWithString = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: "exponential",
});
const _fitWithStringType: VariogramTypeName = fitWithString.variogramType;
const _fitVariogramType: VariogramTypeName = fit.variogramType;
const fittedOrdinary = new OrdinaryKriging({
  lats,
  lons,
  values,
  variogram: {
    variogramType: fit.variogramType,
    nugget: fit.nugget,
    sill: fit.sill,
    range: fit.range,
    shape: fit.shape,
  },
});
const fromFittedOrdinary = OrdinaryKriging.fromFitted({
  lats,
  lons,
  values,
  fittedVariogram: fit,
  nuggetOverride: 0.05,
});
const _fromFittedPred: OrdinaryPrediction = fromFittedOrdinary.predict(
  0.5,
  0.5
);
const gridOpts: PredictGridOptions = {
  west: 0,
  south: 0,
  east: 1,
  north: 1,
  xCells: 5,
  yCells: 4,
};
const ordinaryGrid: OrdinaryGridOutput =
  fromFittedOrdinary.predictGrid(gridOpts);
const _ordinaryGridType: OrdinaryGridOutput = ordinaryGrid;
const fittedBatch = fittedOrdinary.predictBatch(lats, lons);
const _fittedBatchItemType: OrdinaryPrediction = fittedBatch[0];
const fittedBatchArrays = fittedOrdinary.predictBatchArrays(lats, lons);
const _fittedBatchArrayType: OrdinaryBatchArrayOutput = fittedBatchArrays;

const successes = new Uint32Array([2, 4, 6]);
const trials = new Uint32Array([10, 10, 10]);
const binomial = new BinomialKriging({
  lats,
  lons,
  successes,
  trials,
  variogram: {
    variogramType: "exponential",
    nugget: 0.01,
    sill: 1.0,
    range: 100,
  },
});
const bPred = binomial.predict(0.4, 0.4);
const _bPredType: BinomialPrediction = bPred;
const _bPredNotAny: AssertNotAny<typeof bPred> = true;
const bArrayOut = binomial.predictBatchArrays(lats, lons);
const _bArrayType: BinomialBatchArrayOutput = bArrayOut;

const binomialFromFitted = BinomialKriging.fromFittedVariogram({
  lats,
  lons,
  successes,
  trials,
  fittedVariogram: fit,
});
const _binomialFromFittedPred: BinomialPrediction = binomialFromFitted.predict(
  0.4,
  0.4
);

const binomialFromFittedPrior = BinomialKriging.fromFittedVariogramWithPrior({
  lats,
  lons,
  successes,
  trials,
  fittedVariogram: fit,
  prior: { alpha: 1, beta: 1 },
  nuggetOverride: 0.02,
});
const _binomialFromFittedPriorPred: BinomialPrediction =
  binomialFromFittedPrior.predict(0.4, 0.4);
const binomialGrid: BinomialGridOutput =
  binomialFromFittedPrior.predictGrid(gridOpts);
const _binomialGridType: BinomialGridOutput = binomialGrid;

const _oneShotOrdinary: OrdinaryGridOutput = interpolateOrdinaryToGrid({
  lats: Array.from(lats),
  lons: Array.from(lons),
  values: Array.from(values),
  west: 0,
  south: 0,
  east: 2,
  north: 2,
  xCells: 3,
  yCells: 3,
  variogramType: "exponential",
  nBins: 12,
});

// ---------- Space-time contracts ----------

const times = new Float64Array([0, 1, 2]);

const stVariogram = {
  family: "separable" as const,
  spatial: {
    variogramType: "exponential" as const,
    nugget: 0.01,
    sill: 1.0,
    range: 100,
  },
  temporal: {
    variogramType: "exponential" as const,
    nugget: 0.01,
    sill: 1.0,
    range: 5,
  },
};

const stOrdinary = new SpaceTimeOrdinaryKriging({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
});
const _stPred: OrdinaryPrediction = stOrdinary.predict(0.5, 0.5, 1.0);
const _stBatchArrays: OrdinaryBatchArrayOutput = stOrdinary.predictBatchArrays(
  lats,
  lons,
  times
);

const stSimple = new SpaceTimeSimpleKriging({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  mean: 4.0,
});
const _stSimplePred: OrdinaryPrediction = stSimple.predict(0.5, 0.5, 1.0);

const stUniversal = new SpaceTimeUniversalKriging({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  trend: "linearInSpaceAndTime",
});
const _stUniversalPred: OrdinaryPrediction = stUniversal.predict(0.5, 0.5, 1.0);

const stBinomial = new SpaceTimeBinomialKriging({
  lats,
  lons,
  times,
  successes,
  trials,
  variogram: stVariogram,
});
const _stBinomialPred: BinomialPrediction = stBinomial.predict(0.5, 0.5, 1.0);
const _stBinNotes: BinomialBuildNotes = stBinomial.getBuildNotes();
const _stBinomialBatchArrays: BinomialBatchArrayOutput =
  stBinomial.predictBatchArrays(lats, lons, times);

const stProjected = new SpaceTimeProjectedOrdinaryKriging({
  xs: lats,
  ys: lons,
  times,
  values,
  variogram: stVariogram,
  majorAngleDeg: 0,
  rangeRatio: 1.0,
});
const _stProjectedPred: OrdinaryPrediction = stProjected.predict(0.5, 0.5, 1.0);

const _stEmpirical: EmpiricalSpaceTimeVariogramResult =
  computeEmpiricalSpaceTimeVariogram({
    lats,
    lons,
    times,
    values,
    nSpatialBins: 4,
    nTemporalBins: 3,
  });

const _stFit: FitSpaceTimeVariogramResult = fitSpaceTimeVariogram({
  lats,
  lons,
  times,
  values,
  nSpatialBins: 4,
  nTemporalBins: 3,
  family: "separable",
  spatialModel: "exponential",
  temporalModel: "exponential",
});

// Space-time CV / SGS contracts
const _stCv: CvResult = leaveOneOutSpaceTime({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
});
const _stCvKf: CvResult = kFoldSpaceTime({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  k: 3,
});
const _stCvSimple: CvResult = leaveOneOutSpaceTimeSimple({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  mean: 4,
});
const _stCvSimpleKf: CvResult = kFoldSpaceTimeSimple({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  mean: 4,
  k: 3,
});
const _stCvUniv: CvResult = leaveOneOutSpaceTimeUniversal({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  trend: "linearInSpaceAndTime",
});
const _stCvUnivKf: CvResult = kFoldSpaceTimeUniversal({
  lats,
  lons,
  times,
  values,
  variogram: stVariogram,
  trend: "linearInSpaceAndTime",
  k: 3,
});
const _stCvBin: BinomialCvResult = leaveOneOutSpaceTimeBinomial({
  lats,
  lons,
  times,
  successes,
  trials,
  variogram: stVariogram,
});
const _stCvBinKf: BinomialCvResult = kFoldSpaceTimeBinomial({
  lats,
  lons,
  times,
  successes,
  trials,
  variogram: stVariogram,
  k: 3,
});

const _stSgs: Float64Array = conditionalSimulateSpaceTime({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats: new Float64Array([0.5]),
  targetLons: new Float64Array([0.5]),
  targetTimes: new Float64Array([0.5]),
  variogram: stVariogram,
  seed: 1n,
});
const _stSgsSimple: Float64Array = conditionalSimulateSpaceTimeSimple({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats: new Float64Array([0.5]),
  targetLons: new Float64Array([0.5]),
  targetTimes: new Float64Array([0.5]),
  variogram: stVariogram,
  mean: 4,
  seed: 1n,
});
const _stSgsUniv: Float64Array = conditionalSimulateSpaceTimeUniversal({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats: new Float64Array([0.5]),
  targetLons: new Float64Array([0.5]),
  targetTimes: new Float64Array([0.5]),
  variogram: stVariogram,
  trend: "constant",
  seed: 1n,
});
const _stSgsBin: BinomialSimulationResult =
  conditionalSimulateSpaceTimeBinomial({
    conditioningLats: lats,
    conditioningLons: lons,
    conditioningTimes: times,
    successes,
    trials,
    targetLats: new Float64Array([0.5]),
    targetLons: new Float64Array([0.5]),
    targetTimes: new Float64Array([0.5]),
    variogram: stVariogram,
    seed: 1n,
  });

const _oneShotBinomial: InterpolateBinomialToGridResult = interpolateBinomialToGrid({
  lats: Array.from(lats),
  lons: Array.from(lons),
  successes: Array.from(successes),
  trials: Array.from(trials),
  west: 0,
  south: 0,
  east: 2,
  north: 2,
  xCells: 3,
  yCells: 3,
  variogramType: "exponential",
  prior: { alpha: 1, beta: 1 },
});

// ---------- SpaceTimeVariogramParams discriminated-union contracts ----------

const _separableOk: SpaceTimeVariogramParams = {
  family: "separable",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
};

const _productSumOk: SpaceTimeVariogramParams = {
  family: "productSum",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
  k1: 1,
  k2: 0,
  k3: 0,
};

// Separable MUST NOT accept k1/k2/k3.
const _separableRejectsK: SpaceTimeVariogramParams = {
  family: "separable",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
  // @ts-expect-error separable has no k coefficients
  k1: 1,
};

// productSum MUST require k1/k2/k3.
// @ts-expect-error productSum requires k1, k2, k3
const _productSumRequiresK: SpaceTimeVariogramParams = {
  family: "productSum",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
};

// snake_case "product_sum" must be rejected at the TS boundary.
const _snakeCaseRejected: SpaceTimeVariogramParams = {
  // @ts-expect-error snake_case family name is not part of the public API
  family: "product_sum",
  spatial: stVariogram.spatial,
  temporal: stVariogram.temporal,
  k1: 1,
  k2: 0,
  k3: 0,
};

// ---------- 3-D kriging / variography / SGS contracts ----------

const xs3 = new Float64Array([0, 1, 2]);
const ys3 = new Float64Array([0, 1, 2]);
const zs3 = new Float64Array([0, 1, 2]);
const values3 = new Float64Array([3, 4, 5]);
const variogram3d = {
  variogramType: variogram,
  nugget: 0.01,
  sill: 1.0,
  range: 100,
};

// Contract: the 3-D options accept a two-shape (confluent-hypergeometric)
// variogram spec, including the tail-decay shape2. This is what the
// VariogramSpec boundary threads through to the 3-D surface.
const _ch3d = new OrdinaryKriging3D({
  xs: xs3,
  ys: ys3,
  zs: zs3,
  values: values3,
  variogram: {
    variogramType: "confluenthypergeometric",
    nugget: 0.01,
    sill: 1.0,
    range: 100,
    shape: 1.0,
    shape2: 2.0,
  },
});
_ch3d.free();

const ok3d = new OrdinaryKriging3D({
  xs: xs3,
  ys: ys3,
  zs: zs3,
  values: values3,
  variogram: variogram3d,
  anisotropy: { ang1: 0, ang2: 0, ang3: 0, anis1: 1, anis2: 1 },
  neighborhood: { maxNeighbors: 8, maxRadius: 50 },
});
const pred3d = ok3d.predict(0.5, 0.5, 0.5);
const batch3d = ok3d.predictBatch(xs3, ys3, zs3);

const _pred3dNotAny: AssertNotAny<typeof pred3d> = true;
const _batch3dNotAny: AssertNotAny<typeof batch3d> = true;
const _pred3dType: Prediction3D = pred3d;
const _pred3dInflation: boolean = pred3d.usedNuggetInflation;
const _batch3dType: Batch3DArrayOutput = batch3d;
const _batch3dCond: Float64Array = batch3d.conditionNumbers;

const sk3d = new SimpleKriging3D({
  xs: xs3,
  ys: ys3,
  zs: zs3,
  values: values3,
  variogram: variogram3d,
  mean: 4,
});
const _sk3dPred: Prediction3D = sk3d.predict(0.5, 0.5, 0.5);

const uk3d = new UniversalKriging3D({
  xs: xs3,
  ys: ys3,
  zs: zs3,
  values: values3,
  variogram: variogram3d,
});
const _uk3dPred: Prediction3D = uk3d.predict(0.5, 0.5, 0.5);

const dir3d = computeDirectionalVariogram3D({
  xs: xs3,
  ys: ys3,
  zs: zs3,
  values: values3,
  lagDistance: 1,
  nLags: 10,
  azimuthDeg: 0,
  dipDeg: 0,
});
const _dir3dNotAny: AssertNotAny<typeof dir3d> = true;
const _dir3dType: DirectionalVariogram3DResult = dir3d;
const _dir3dPairs: Uint32Array = dir3d.nPairs;

const axis = {
  distances: dir3d.distances,
  semivariances: dir3d.semivariances,
  nPairs: dir3d.nPairs,
};
const fit3dJoint = fitSpherical3DJoint({
  major: axis,
  minor: axis,
  vertical: axis,
});
const _fit3dNotAny: AssertNotAny<typeof fit3dJoint> = true;
const _fit3dType: FittedSpherical3D = fit3dJoint;
const _fit3dTwoStage: FittedSpherical3D = fitSpherical3DTwoStage({
  major: axis,
  minor: axis,
  vertical: axis,
  dataVariance: 1.0,
});
const _fit3dFixed: FittedSpherical3D = fitSpherical3DFixedNugget({
  major: axis,
  minor: axis,
  vertical: axis,
  nugget: 0.05,
});
const _fit1d: FittedSpherical1D = fitSpherical1D(axis);

const sgs3dResult = gaussianSimulation3D({
  xs: xs3,
  ys: ys3,
  zs: zs3,
  values: values3,
  variogram: variogram3d,
  grid: {
    nx: 4,
    ny: 4,
    nz: 2,
    originX: 0,
    originY: 0,
    originZ: 0,
    spacingX: 1,
    spacingY: 1,
    spacingZ: 1,
  },
  seed: 42n,
  nRealizations: 2,
  onRealization: (idx: number, grid: Float64Array) => {
    void idx;
    return grid.length === 0;
  },
});
const _sgs3dNotAny: AssertNotAny<typeof sgs3dResult> = true;
const _sgs3dType: GaussianSimulation3DResult = sgs3dResult;
