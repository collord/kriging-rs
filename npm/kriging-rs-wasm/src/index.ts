/**
 * Entry point for `kriging-rs-wasm`. This module only re-exports the public API;
 * implementation lives in sibling modules (see `types.ts`, `errors.ts`, `kriging/*`,
 * `variogram.ts`, `cv.ts`, `simulation.ts`, `interpolate.ts`).
 *
 * Typical usage:
 *
 * ```ts
 * import init, { OrdinaryKriging, fitVariogram } from "kriging-rs-wasm";
 *
 * await init();
 * const fitted = fitVariogram({ sampleLats, sampleLons, values, variogramType: "exponential" });
 * const model = OrdinaryKriging.fromFitted({ lats, lons, values, fittedVariogram: fitted });
 * const { value, variance } = model.predict(37.7, -122.4);
 * model.free();
 * ```
 *
 * @module
 */

export { init, VariogramType, webgpuAvailable } from "./internal/module.js";
export { KrigingError } from "./errors.js";
export { OrdinaryKriging } from "./kriging/ordinary.js";
export { SimpleKriging } from "./kriging/simple.js";
export { UniversalKriging } from "./kriging/universal.js";
export { ProjectedKriging } from "./kriging/projected.js";
export { BinomialKriging } from "./kriging/binomial.js";
export { BinomialProjectedKriging } from "./kriging/binomial-projected.js";
export { OrdinaryKriging3D } from "./kriging/ordinary-3d.js";
export { SimpleKriging3D } from "./kriging/simple-3d.js";
export { UniversalKriging3D } from "./kriging/universal-3d.js";
export {
  computeDirectionalVariogram3D,
  fitSpherical1D,
  fitSpherical3DFixedNugget,
  fitSpherical3DJoint,
  fitSpherical3DTwoStage,
} from "./variogram-3d.js";
export { gaussianSimulation3D } from "./simulate-3d.js";
export {
  computeDirectionalEmpiricalVariogram,
  computeEmpiricalVariogram,
  evaluateNestedVariogram,
  fitVariogram,
} from "./variogram.js";
export {
  kFold,
  kFoldBinomial,
  kFoldBinomialProjected,
  kFoldProjected,
  kFoldSimple,
  kFoldSpaceTime,
  kFoldSpaceTimeBinomial,
  kFoldSpaceTimeSimple,
  kFoldSpaceTimeUniversal,
  kFoldUniversal,
  leaveOneOut,
  leaveOneOutBinomial,
  leaveOneOutBinomialProjected,
  leaveOneOutProjected,
  leaveOneOutSimple,
  leaveOneOutSpaceTime,
  leaveOneOutSpaceTimeBinomial,
  leaveOneOutSpaceTimeSimple,
  leaveOneOutSpaceTimeUniversal,
  leaveOneOutUniversal,
} from "./cv.js";
export {
  conditionalSimulate,
  conditionalSimulateBinomial,
  conditionalSimulateBinomialProjected,
  conditionalSimulateMany,
  conditionalSimulateManyBinomial,
  conditionalSimulateManyBinomialProjected,
  conditionalSimulateManySpaceTime,
  conditionalSimulateManySpaceTimeBinomial,
  conditionalSimulateProjected,
  conditionalSimulateSimple,
  conditionalSimulateSpaceTime,
  conditionalSimulateSpaceTimeBinomial,
  conditionalSimulateSpaceTimeSimple,
  conditionalSimulateSpaceTimeUniversal,
  conditionalSimulateUniversal,
} from "./simulation.js";
export {
  ensembleExceedanceProbability,
  ensembleMean,
  ensembleQuantiles,
  ensembleVariance,
} from "./aggregate.js";
export {
  gridCellCenters,
  reshapeGridRow,
  simulateBinomialGrid,
  simulateBinomialGridEnsemble,
  simulateBinomialGridSummary,
} from "./simulate-grid.js";
export {
  simulateBinomialSpaceTimeGrid,
  simulateBinomialSpaceTimeGridAtDate,
  simulateBinomialSpaceTimeGridEnsemble,
  simulateBinomialSpaceTimeGridEnsembleAtDate,
  simulateBinomialSpaceTimeGridSummary,
  simulateBinomialSpaceTimeGridSummaryAtDate,
} from "./simulate-grid-spacetime.js";
export {
  aggregatePrevalenceByPolygon,
  polygonCellsFromMask,
} from "./polygons.js";
export {
  interpolateBinomialToGrid,
  interpolateOrdinaryToGrid,
} from "./interpolate.js";
export {
  SpaceTimeBinomialKriging,
  SpaceTimeOrdinaryKriging,
  SpaceTimeProjectedOrdinaryKriging,
  SpaceTimeSimpleKriging,
  SpaceTimeUniversalKriging,
  computeEmpiricalSpaceTimeVariogram,
  fitSpaceTimeVariogram,
} from "./spacetime/index.js";
export { datesFromTimes, timesFromDates } from "./time.js";
export {
  cokrige,
  cokrigeCollocated,
  collocatedCosimulate,
  collocatedSecondaryFromPaired,
  computeCrossVariogram,
  cosimulateMultiVariable,
  fitLmc,
} from "./cokriging.js";

export type {
  CokrigeCollocatedOptions,
  CokrigeOptions,
  CokrigingKindSpec,
  CollocatedCosimulateOptions,
  CollocatedSecondary,
  CoregionalizationSpec,
  CoregionalizationStructureSpec,
  CorrelationBasisSpec,
  CosimulateMultiVariableOptions,
  CosimulationResult,
  CrossVariogramOptions,
  CrossVariogramResult,
  FitLmcOptions,
  HeterotopicMultiVariableData,
  IsotopicMultiVariableData,
  LmcFitResult,
  MultiVariableData,
  VariableSamples,
  AggregatePrevalenceByPolygonOptions,
  Anisotropy3DParams,
  AxisVariogramInput,
  Batch3DArrayOutput,
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResidual,
  BinomialCvResult,
  BinomialCvSummary,
  BinomialFromPrecomputedLogitsOptions,
  BinomialGridEnsemble,
  BinomialGridOutput,
  BinomialGridSimulation,
  BinomialGridSummary,
  BinomialKrigingFromFittedVariogramOptions,
  BinomialKrigingFromFittedVariogramWithPriorOptions,
  BinomialKrigingOptions,
  BinomialKrigingWithPriorOptions,
  BinomialPrediction,
  BinomialPriorParams,
  BinomialProjectedFromPrecomputedLogitsOptions,
  BinomialProjectedKrigingOptions,
  BinomialProjectedKrigingWithPriorOptions,
  BinomialSimulationManyResult,
  BinomialSimulationResult,
  ComputeDirectionalEmpiricalVariogramOptions,
  ComputeEmpiricalVariogramOptions,
  ConditionalSimulateBinomialOptions,
  ConditionalSimulateBinomialProjectedOptions,
  ConditionalSimulateManyBinomialOptions,
  ConditionalSimulateManyBinomialProjectedOptions,
  ConditionalSimulateManyOptions,
  ConditionalSimulateManySpaceTimeBinomialOptions,
  ConditionalSimulateManySpaceTimeOptions,
  ConditionalSimulateOptions,
  ConditionalSimulateProjectedOptions,
  ConditionalSimulateSimpleOptions,
  ConditionalSimulateSpaceTimeBinomialOptions,
  ConditionalSimulateSpaceTimeOptions,
  ConditionalSimulateSpaceTimeSimpleOptions,
  ConditionalSimulateSpaceTimeUniversalOptions,
  ConditionalSimulateUniversalOptions,
  CvResidual,
  DateAxisOptions,
  CvResult,
  CvSummary,
  DirectionalVariogram3DOptions,
  DirectionalVariogram3DResult,
  EmpiricalEstimator,
  EmpiricalVariogramResult,
  FitSpherical1DOptions,
  FitSpherical3DFixedNuggetOptions,
  FitSpherical3DOptions,
  FitSpherical3DTwoStageOptions,
  FitVariogramOptions,
  FittedSpherical1D,
  FittedSpherical3D,
  FittedVariogram,
  GaussianSimulation3DOptions,
  GaussianSimulation3DResult,
  GeoGridBounds,
  Grid3DOptions,
  IntegerArrayInput,
  InterpolateBinomialToGridOptions,
  InterpolateBinomialToGridResult,
  InterpolateOrdinaryToGridOptions,
  KFoldBinomialOptions,
  KFoldBinomialProjectedOptions,
  KFoldOptions,
  KFoldProjectedOptions,
  KFoldSimpleOptions,
  KFoldSpaceTimeBinomialOptions,
  KFoldSpaceTimeOptions,
  KFoldSpaceTimeSimpleOptions,
  KFoldSpaceTimeUniversalOptions,
  KFoldUniversalOptions,
  KrigingErrorCode,
  LeaveOneOutBinomialOptions,
  LeaveOneOutBinomialProjectedOptions,
  LeaveOneOutOptions,
  LeaveOneOutProjectedOptions,
  LeaveOneOutSimpleOptions,
  LeaveOneOutSpaceTimeBinomialOptions,
  LeaveOneOutSpaceTimeOptions,
  LeaveOneOutSpaceTimeSimpleOptions,
  LeaveOneOutSpaceTimeUniversalOptions,
  LeaveOneOutUniversalOptions,
  Neighborhood3DOptions,
  NeighborhoodOptions,
  NestedVariogramComponent,
  NestedVariogramEvaluation,
  NumericArrayInput,
  OrdinaryBatchArrayOutput,
  OrdinaryGridOutput,
  OrdinaryKriging3DOptions,
  OrdinaryKrigingFromFittedOptions,
  OrdinaryKrigingOptions,
  OrdinaryPrediction,
  Prediction3D,
  PolygonAggregateResult,
  PolygonCells,
  PolygonCellsFromMaskOptions,
  PredictGridAtDateOptions,
  PredictGridAtTimeOptions,
  PredictGridOptions,
  ProjectedKrigingOptions,
  Realization3DCallback,
  SimpleKriging3DOptions,
  SimpleKrigingOptions,
  SimulateBinomialGridEnsembleOptions,
  SimulateBinomialGridOptions,
  SimulateBinomialGridSummaryOptions,
  SimulateBinomialSpaceTimeGridAtDateOptions,
  SimulateBinomialSpaceTimeGridEnsembleAtDateOptions,
  SimulateBinomialSpaceTimeGridEnsembleOptions,
  SimulateBinomialSpaceTimeGridOptions,
  SimulateBinomialSpaceTimeGridSummaryAtDateOptions,
  SimulateBinomialSpaceTimeGridSummaryOptions,
  UniversalKriging3DOptions,
  UniversalKrigingOptions,
  UniversalTrend,
  VariogramParams,
  VariogramTypeName,
  ComputeEmpiricalSpaceTimeVariogramOptions,
  EmpiricalSpaceTimeVariogramResult,
  FitSpaceTimeVariogramOptions,
  FitSpaceTimeVariogramResult,
  FittedSpaceTimeVariogram,
  SpaceTimeBinomialKrigingFromFittedOptions,
  SpaceTimeBinomialKrigingOptions,
  SpaceTimeOrdinaryKrigingFromFittedOptions,
  SpaceTimeOrdinaryKrigingOptions,
  SpaceTimeProjectedOrdinaryKrigingFromFittedOptions,
  SpaceTimeProjectedOrdinaryKrigingOptions,
  SpaceTimeSimpleKrigingFromFittedOptions,
  SpaceTimeSimpleKrigingOptions,
  SpaceTimeUniversalKrigingFromFittedOptions,
  SpaceTimeUniversalKrigingOptions,
  SpaceTimeUniversalTrend,
  SpaceTimeVariogramFamily,
  SpaceTimeVariogramParams,
} from "./types.js";
export type { TimeUnit } from "./time.js";

import { init } from "./internal/module.js";

export default init;
