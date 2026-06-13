/**
 * Internal: value converters that turn raw WASM outputs into the strongly-typed shapes
 * exposed by the public API.
 *
 * @module
 */

import type {
  Batch3DArrayOutput,
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialCvResidual,
  BinomialCvResult,
  BinomialCvSummary,
  BinomialPrediction,
  BinomialPriorParams,
  CvResidual,
  CvResult,
  CvSummary,
  DirectionalVariogram3DResult,
  EmpiricalSpaceTimeVariogramResult,
  EmpiricalVariogramResult,
  FitSpaceTimeVariogramResult,
  FittedSpaceTimeVariogram,
  FittedSpherical1D,
  FittedSpherical3D,
  OrdinaryBatchArrayOutput,
  OrdinaryPrediction,
  Prediction3D,
  VariogramParams,
  VariogramTypeName,
} from "../types.js";
import {
  asRecord,
  requireFiniteOrNaN,
  requireFloat64Array,
  requireNumber,
  requireUint32Array,
  requireVariogramType,
} from "./convert.js";

export function mapOrdinaryPrediction(value: unknown): OrdinaryPrediction {
  const item = asRecord(value);
  return {
    value: requireNumber(item.value),
    variance: requireNumber(item.variance),
  };
}

export function mapOrdinaryPredictionArray(
  value: unknown
): OrdinaryPrediction[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected ordinary prediction array output");
  }
  return value.map((item) => mapOrdinaryPrediction(item));
}

export function mapBinomialPrediction(value: unknown): BinomialPrediction {
  const item = asRecord(value);
  const maybeSnake = item.logit_value;
  const maybeCamel = item.logitValue;
  const logit = maybeCamel ?? maybeSnake;
  const maybePrevalenceVariance =
    item.prevalenceVariance ?? item.prevalence_variance;
  const variance = requireNumber(item.variance);
  const prevalence = requireNumber(item.prevalence);
  const prevalenceVariance =
    typeof maybePrevalenceVariance === "number" &&
    Number.isFinite(maybePrevalenceVariance)
      ? maybePrevalenceVariance
      : deltaPrevalenceVariance(prevalence, variance);
  return {
    prevalence,
    logitValue: requireNumber(logit),
    variance,
    prevalenceVariance,
  };
}

export function mapBinomialPredictionArray(
  value: unknown
): BinomialPrediction[] {
  if (!Array.isArray(value)) {
    throw new Error("Expected binomial prediction array output");
  }
  return value.map((item) => mapBinomialPrediction(item));
}

function mapBinomialPriorFromNotes(value: unknown): BinomialPriorParams {
  const rec = asRecord(value);
  return {
    alpha: requireNumber(rec.alpha),
    beta: requireNumber(rec.beta),
  };
}

function mapZeroTrialDroppedIndices(value: unknown): number[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) {
    throw new Error("Expected zeroTrialDroppedIndices array from WASM");
  }
  return value.map((x) => requireNumber(x));
}

/** Map raw `getBuildNotes()` JSON from WASM to {@link BinomialBuildNotes}. */
export function mapBinomialBuildNotes(value: unknown): BinomialBuildNotes {
  const rec = asRecord(value);
  const priorRaw = rec.prior ?? rec.Prior;
  const calVer =
    rec.calibrationVersion ?? rec.calibration_version;
  const logitInfl =
    rec.logitInflation ?? rec.logit_inflation;
  const nAttempts = rec.nBuildAttempts ?? rec.n_build_attempts;
  const dropped =
    rec.zeroTrialDroppedIndices ?? rec.zero_trial_dropped_indices;
  const fromLogitsOnly =
    rec.fromPrecomputedLogitsOnly ?? rec.from_precomputed_logits_only;
  if (typeof fromLogitsOnly !== "boolean") {
    throw new Error("Expected boolean fromPrecomputedLogitsOnly from WASM");
  }
  return {
    calibrationVersion: requireNumber(calVer),
    logitInflation: requireNumber(logitInfl),
    nBuildAttempts: requireNumber(nAttempts),
    prior: mapBinomialPriorFromNotes(priorRaw),
    zeroTrialDroppedIndices: mapZeroTrialDroppedIndices(dropped),
    fromPrecomputedLogitsOnly: fromLogitsOnly,
  };
}

export function mapOrdinaryBatchArrayOutput(
  value: unknown
): OrdinaryBatchArrayOutput {
  const out = asRecord(value);
  return {
    values: requireFloat64Array(out.values),
    variances: requireFloat64Array(out.variances),
  };
}

export function mapBinomialBatchArrayOutput(
  value: unknown
): BinomialBatchArrayOutput {
  const out = asRecord(value);
  const prevalences = requireFloat64Array(out.prevalences);
  const variances = requireFloat64Array(out.variances);
  const explicit = out.prevalenceVariances;
  const prevalenceVariances =
    explicit instanceof Float64Array
      ? explicit
      : computeDeltaPrevalenceVariances(prevalences, variances);
  return {
    prevalences,
    logitValues: requireFloat64Array(out.logitValues),
    variances,
    prevalenceVariances,
  };
}

export function deltaPrevalenceVariance(
  prevalence: number,
  logitVariance: number
): number {
  const factor = prevalence * (1 - prevalence);
  return factor * factor * Math.max(0, logitVariance);
}

function computeDeltaPrevalenceVariances(
  prevalences: Float64Array,
  variances: Float64Array
): Float64Array {
  const n = prevalences.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = deltaPrevalenceVariance(prevalences[i], variances[i]);
  }
  return out;
}

export function mapEmpiricalVariogram(
  value: unknown
): EmpiricalVariogramResult {
  const rec = asRecord(value);
  return {
    distances: requireFloat64Array(rec.distances),
    semivariances: requireFloat64Array(rec.semivariances),
    counts: requireUint32Array(rec.counts),
  };
}

export function mapCvOutput(value: unknown): CvResult {
  const rec = asRecord(value);
  const indices = requireUint32Array(rec.indices);
  const observed = requireFloat64Array(rec.observed);
  const predicted = requireFloat64Array(rec.predicted);
  const variances = requireFloat64Array(rec.variances);
  const summaryRec = asRecord(rec.summary);
  const summary: CvSummary = {
    n: requireNumber(summaryRec.n),
    meanError: requireFiniteOrNaN(summaryRec.meanError),
    rmse: requireFiniteOrNaN(summaryRec.rmse),
    msdr: requireFiniteOrNaN(summaryRec.msdr),
  };
  const residuals: CvResidual[] = [];
  for (let i = 0; i < indices.length; i++) {
    residuals.push({
      index: indices[i],
      observed: observed[i],
      predicted: predicted[i],
      variance: variances[i],
      error: observed[i] - predicted[i],
    });
  }
  return {
    residuals,
    summary,
    arrays: { indices, observed, predicted, variances },
  };
}

function mapCvSummary(value: unknown): CvSummary {
  const rec = asRecord(value);
  return {
    n: requireNumber(rec.n),
    meanError: requireFiniteOrNaN(rec.meanError),
    rmse: requireFiniteOrNaN(rec.rmse),
    msdr: requireFiniteOrNaN(rec.msdr),
  };
}

export function mapBinomialCvOutput(value: unknown): BinomialCvResult {
  const rec = asRecord(value);
  const indices = requireUint32Array(rec.indices);
  const successes = requireUint32Array(rec.successes);
  const trials = requireUint32Array(rec.trials);
  const observedLogit = requireFloat64Array(rec.observedLogit);
  const predictedLogit = requireFloat64Array(rec.predictedLogit);
  const logitVariance = requireFloat64Array(rec.logitVariance);
  const observedPrevalence = requireFloat64Array(rec.observedPrevalence);
  const predictedPrevalence = requireFloat64Array(rec.predictedPrevalence);
  const prevalenceVariance = requireFloat64Array(rec.prevalenceVariance);
  const summaryRec = asRecord(rec.summary);
  const summary: BinomialCvSummary = {
    n: requireNumber(summaryRec.n),
    nEvaluated: requireNumber(summaryRec.nEvaluated),
    logit: mapCvSummary(summaryRec.logit),
    prevalence: mapCvSummary(summaryRec.prevalence),
  };
  const residuals: BinomialCvResidual[] = [];
  for (let i = 0; i < indices.length; i++) {
    residuals.push({
      index: indices[i],
      successes: successes[i],
      trials: trials[i],
      observedLogit: observedLogit[i],
      predictedLogit: predictedLogit[i],
      logitVariance: logitVariance[i],
      observedPrevalence: observedPrevalence[i],
      predictedPrevalence: predictedPrevalence[i],
      prevalenceVariance: prevalenceVariance[i],
      logitError: observedLogit[i] - predictedLogit[i],
      prevalenceError: observedPrevalence[i] - predictedPrevalence[i],
    });
  }
  return {
    residuals,
    summary,
    arrays: {
      indices,
      successes,
      trials,
      observedLogit,
      predictedLogit,
      logitVariance,
      observedPrevalence,
      predictedPrevalence,
      prevalenceVariance,
    },
  };
}

// ---------- Space-time mappers ----------

function mapVariogramParams(value: unknown): VariogramParams {
  const rec = asRecord(value);
  const vt = requireVariogramType(rec.variogramType);
  const out: VariogramParams = {
    variogramType: vt as VariogramTypeName,
    nugget: requireNumber(rec.nugget),
    sill: requireNumber(rec.sill),
    range: requireNumber(rec.range),
  };
  if (typeof rec.shape === "number" && Number.isFinite(rec.shape)) {
    out.shape = rec.shape;
  }
  return out;
}

export function mapFittedSpaceTimeVariogram(
  value: unknown
): FittedSpaceTimeVariogram {
  const rec = asRecord(value);
  const family = rec.family;
  if (family !== "separable" && family !== "productSum") {
    throw new Error("Unknown space-time variogram family from WASM");
  }
  const spatial = mapVariogramParams(rec.spatial);
  const temporal = mapVariogramParams(rec.temporal);
  const residuals = requireFiniteOrNaN(rec.residuals);
  if (family === "productSum") {
    return {
      family: "productSum",
      spatial,
      temporal,
      k1: requireNumber(rec.k1),
      k2: requireNumber(rec.k2),
      k3: requireNumber(rec.k3),
      residuals,
    };
  }
  return {
    family: "separable",
    spatial,
    temporal,
    residuals,
  };
}

export function mapEmpiricalSpaceTimeVariogram(
  value: unknown
): EmpiricalSpaceTimeVariogramResult {
  const rec = asRecord(value);
  return {
    nSpatialBins: requireNumber(rec.nSpatialBins),
    nTemporalBins: requireNumber(rec.nTemporalBins),
    spatialLags: requireFloat64Array(rec.spatialLags),
    temporalLags: requireFloat64Array(rec.temporalLags),
    semivariances: requireFloat64Array(rec.semivariances),
    nPairs: requireFloat64Array(rec.nPairs),
  };
}

export function mapFitSpaceTimeVariogramResult(
  value: unknown
): FitSpaceTimeVariogramResult {
  const rec = asRecord(value);
  return {
    empirical: mapEmpiricalSpaceTimeVariogram(rec.empirical),
    fit: mapFittedSpaceTimeVariogram(rec.fit),
  };
}

export function mapPrediction3D(value: unknown): Prediction3D {
  const item = asRecord(value);
  if (typeof item.usedNuggetInflation !== "boolean") {
    throw new Error("Expected boolean usedNuggetInflation from WASM");
  }
  return {
    value: requireNumber(item.value),
    variance: requireNumber(item.variance),
    conditionNumber: requireNumber(item.conditionNumber),
    usedNuggetInflation: item.usedNuggetInflation,
  };
}

export function mapBatch3DArrayOutput(value: unknown): Batch3DArrayOutput {
  const rec = asRecord(value);
  return {
    values: requireFloat64Array(rec.values),
    variances: requireFloat64Array(rec.variances),
    conditionNumbers: requireFloat64Array(rec.conditionNumbers),
    usedNuggetInflation: requireFloat64Array(rec.usedNuggetInflation),
  };
}

export function mapDirectionalVariogram3DResult(
  value: unknown
): DirectionalVariogram3DResult {
  const rec = asRecord(value);
  return {
    distances: requireFloat64Array(rec.distances),
    semivariances: requireFloat64Array(rec.semivariances),
    // The WASM side ships pair counts as a Float64Array; counts are integral.
    nPairs: Uint32Array.from(requireFloat64Array(rec.nPairs)),
  };
}

export function mapFittedSpherical3D(value: unknown): FittedSpherical3D {
  const rec = asRecord(value);
  return {
    nugget: requireNumber(rec.nugget),
    sill: requireNumber(rec.sill),
    rangeMajor: requireNumber(rec.rangeMajor),
    rangeMinor: requireNumber(rec.rangeMinor),
    rangeVertical: requireNumber(rec.rangeVertical),
    residuals: requireNumber(rec.residuals),
  };
}

export function mapFittedSpherical1D(value: unknown): FittedSpherical1D {
  const rec = asRecord(value);
  return {
    nugget: requireNumber(rec.nugget),
    sill: requireNumber(rec.sill),
    range: requireNumber(rec.range),
    residuals: requireNumber(rec.residuals),
  };
}
