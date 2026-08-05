import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { beforeAll, describe, test, expect } from "vitest";
import {
  init,
  KrigingError,
  OrdinaryKriging,
  BinomialKriging,
  SimpleKriging,
  UniversalKriging,
  ProjectedKriging,
  BinomialProjectedKriging,
  SpaceTimeOrdinaryKriging,
  SpaceTimeSimpleKriging,
  SpaceTimeUniversalKriging,
  SpaceTimeBinomialKriging,
  SpaceTimeProjectedOrdinaryKriging,
  computeEmpiricalSpaceTimeVariogram,
  fitSpaceTimeVariogram,
  fitVariogram,
  interpolateOrdinaryToGrid,
  interpolateBinomialToGrid,
  computeEmpiricalVariogram,
  computeDirectionalEmpiricalVariogram,
  leaveOneOut,
  kFold,
  leaveOneOutSimple,
  kFoldSimple,
  leaveOneOutUniversal,
  kFoldUniversal,
  leaveOneOutProjected,
  kFoldProjected,
  leaveOneOutBinomial,
  kFoldBinomial,
  leaveOneOutBinomialProjected,
  kFoldBinomialProjected,
  conditionalSimulate,
  conditionalSimulateMany,
  conditionalSimulateManyBinomial,
  conditionalSimulateManySpaceTime,
  conditionalSimulateManySpaceTimeBinomial,
  conditionalSimulateSimple,
  conditionalSimulateUniversal,
  conditionalSimulateProjected,
  conditionalSimulateBinomial,
  conditionalSimulateBinomialProjected,
  conditionalSimulateManyBinomialProjected,
  leaveOneOutSpaceTime,
  kFoldSpaceTime,
  leaveOneOutSpaceTimeSimple,
  kFoldSpaceTimeSimple,
  leaveOneOutSpaceTimeUniversal,
  kFoldSpaceTimeUniversal,
  leaveOneOutSpaceTimeBinomial,
  kFoldSpaceTimeBinomial,
  conditionalSimulateSpaceTime,
  conditionalSimulateSpaceTimeSimple,
  conditionalSimulateSpaceTimeUniversal,
  conditionalSimulateSpaceTimeBinomial,
  ensembleExceedanceProbability,
  ensembleMean,
  ensembleQuantiles,
  ensembleVariance,
  gridCellCenters,
  reshapeGridRow,
  simulateBinomialGrid,
  simulateBinomialGridEnsemble,
  simulateBinomialGridSummary,
  simulateBinomialSpaceTimeGrid,
  simulateBinomialSpaceTimeGridAtDate,
  simulateBinomialSpaceTimeGridEnsemble,
  simulateBinomialSpaceTimeGridEnsembleAtDate,
  simulateBinomialSpaceTimeGridSummary,
  simulateBinomialSpaceTimeGridSummaryAtDate,
  aggregatePrevalenceByPolygon,
  polygonCellsFromMask,
  evaluateNestedVariogram,
  VariogramType,
  OrdinaryKriging3D,
  SimpleKriging3D,
  UniversalKriging3D,
  computeDirectionalVariogram3D,
  fitSpherical1D,
  fitSpherical3DFixedNugget,
  fitSpherical3DJoint,
  fitSpherical3DTwoStage,
  gaussianSimulation3D,
  type OrdinaryPrediction,
  type BinomialPrediction,
  type VariogramTypeName,
  type Prediction3D,
} from "../dist/index.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const pkgDir = resolve(__dirname, "..");
const wasmPath = resolve(pkgDir, "pkg/kriging_rs_bg.wasm");

async function loadWasm(): Promise<Buffer> {
  return readFile(wasmPath);
}

beforeAll(async () => {
  const wasmBytes = await loadWasm();
  await init(wasmBytes);
});

describe("Ordinary kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const values = [10, 12, 11, 13];
  const variogramType: VariogramTypeName = "gaussian";
  const nugget = 0.01;
  const sill = 1.5;
  const range = 5.0;

  test("using block disposes an OrdinaryKriging model and a later predict throws model_freed", () => {
    let captured: OrdinaryKriging;
    {
      using model = new OrdinaryKriging({
        lats,
        lons,
        values,
        variogram: { variogramType, nugget, sill, range },
      });
      captured = model;
      const { value } = model.predict(0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
    }
    let err: unknown;
    try {
      captured.predict(0.5, 0.5);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(KrigingError);
    expect((err as KrigingError).code).toBe("model_freed");
  });

  test("fitVariogram returns valid FittedVariogram (enum)", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Exponential,
      nBins: 12,
    });
    expect(fit).toMatchObject({
      variogramType: "exponential",
      nugget: expect.any(Number),
      sill: expect.any(Number),
      range: expect.any(Number),
      residuals: expect.any(Number),
    });
    expect(fit.nugget).toBeGreaterThanOrEqual(0);
    expect(fit.sill).toBeGreaterThanOrEqual(0);
    expect(fit.range).toBeGreaterThanOrEqual(0);
  });

  test("fitVariogram accepts string variogramType", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: "gaussian",
    });
    expect(fit.variogramType).toBe("gaussian");
    expect(fit.nugget).toBeGreaterThanOrEqual(0);
    expect(fit.sill).toBeGreaterThanOrEqual(0);
    expect(fit.range).toBeGreaterThanOrEqual(0);
  });

  test("OrdinaryKriging predict returns value and variance", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const pred = model.predict(0.5, 0.5);
    expect(pred).toMatchObject({
      value: expect.any(Number),
      variance: expect.any(Number),
    });
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("OrdinaryKriging predictBatch returns array of predictions", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const batchLats = [0.25, 0.5, 0.75];
    const batchLons = [0.25, 0.5, 0.75];
    const out = model.predictBatch(batchLats, batchLons);
    expect(Array.isArray(out)).toBe(true);
    expect(out.length).toBe(3);
    out.forEach((p: OrdinaryPrediction) => {
      expect(p).toMatchObject({
        value: expect.any(Number),
        variance: expect.any(Number),
      });
      expect(p.variance).toBeGreaterThanOrEqual(0);
    });
    model.free();
  });

  test("OrdinaryKriging predictBatchArrays returns Float64Arrays", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const batchLats = [0.25, 0.5];
    const batchLons = [0.25, 0.5];
    const out = model.predictBatchArrays(batchLats, batchLons);
    expect(out.values).toBeInstanceOf(Float64Array);
    expect(out.variances).toBeInstanceOf(Float64Array);
    expect(out.values.length).toBe(2);
    expect(out.variances.length).toBe(2);
    out.variances.forEach((v) => expect(v).toBeGreaterThanOrEqual(0));
    model.free();
  });

  test("model from fitVariogram produces consistent predictions", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Gaussian,
      nBins: 12,
    });
    const model = new OrdinaryKriging({
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
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.value)).toBe(true);
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("OrdinaryKriging.fromFitted produces same predictions as constructor with fitted variogram", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Gaussian,
      nBins: 12,
    });
    const model = OrdinaryKriging.fromFitted({
      lats,
      lons,
      values,
      fittedVariogram: fit,
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.value)).toBe(true);
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("OrdinaryKriging.fromFitted with nuggetOverride uses override", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: VariogramType.Gaussian,
      nBins: 12,
    });
    const modelNoOverride = OrdinaryKriging.fromFitted({
      lats,
      lons,
      values,
      fittedVariogram: fit,
    });
    const predNoOverride = modelNoOverride.predict(0.5, 0.5);
    modelNoOverride.free();
    const modelOverride = OrdinaryKriging.fromFitted({
      lats,
      lons,
      values,
      fittedVariogram: fit,
      nuggetOverride: 0,
    });
    const predOverride = modelOverride.predict(0.5, 0.5);
    modelOverride.free();
    expect(Number.isFinite(predOverride.value)).toBe(true);
    expect(Number.isFinite(predOverride.variance)).toBe(true);
    expect(predOverride.variance).not.toBe(predNoOverride.variance);
  });

  test("OrdinaryKriging free() is idempotent", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    model.free();
    model.free();
    expect(() => model.predict(0.5, 0.5)).toThrow(KrigingError);
  });

  test("KrigingError has code model_freed after free()", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    model.free();
    try {
      model.predict(0.5, 0.5);
    } catch (e) {
      expect(e).toBeInstanceOf(KrigingError);
      expect((e as KrigingError).code).toBe("model_freed");
      return;
    }
    expect.fail("should have thrown");
  });

  test("OrdinaryKriging predictGrid returns 2D arrays matching predictBatchArrays", () => {
    const model = new OrdinaryKriging({
      lats,
      lons,
      values,
      variogram: { variogramType, nugget, sill, range },
    });
    const gridOpts = {
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 3,
    };
    const out = model.predictGrid(gridOpts);
    expect(out.values.length).toBe(3);
    expect(out.values[0].length).toBe(2);
    expect(out.variances.length).toBe(3);
    expect(out.variances[0].length).toBe(2);
    const { lats: flatLats, lons: flatLons } = (() => {
      const nRows = 3;
      const nCols = 2;
      const latsArr = new Float64Array(nRows * nCols);
      const lonsArr = new Float64Array(nRows * nCols);
      let k = 0;
      for (let j = 0; j < nRows; j++) {
        const lat = 0 + (j + 0.5) * (1 / nRows);
        for (let i = 0; i < nCols; i++) {
          latsArr[k] = lat;
          lonsArr[k] = 0 + (i + 0.5) * (1 / nCols);
          k++;
        }
      }
      return { lats: latsArr, lons: lonsArr };
    })();
    const batch = model.predictBatchArrays(flatLats, flatLons);
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 2; i++) {
        expect(out.values[j][i]).toBe(batch.values[j * 2 + i]);
        expect(out.variances[j][i]).toBe(batch.variances[j * 2 + i]);
      }
    }
    model.free();
  });

  test("interpolateOrdinaryToGrid returns grid and frees model", () => {
    const out = interpolateOrdinaryToGrid({
      lats,
      lons,
      values,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
    });
    expect(out.values.length).toBe(2);
    expect(out.values[0].length).toBe(2);
    expect(out.variances.length).toBe(2);
    expect(Number.isFinite(out.values[0][0])).toBe(true);
  });
});

describe("Binomial kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const successes = [2, 4, 3, 5];
  const trials = [10, 10, 10, 10];
  const variogramType: VariogramTypeName = "exponential";
  const nugget = 0.01;
  const sill = 1.0;
  const range = 100;

  test("BinomialKriging predict returns prevalence, logitValue, variance", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const pred = model.predict(0.5, 0.5);
    expect(pred).toMatchObject({
      prevalence: expect.any(Number),
      logitValue: expect.any(Number),
      variance: expect.any(Number),
    });
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    expect(pred.variance).toBeGreaterThanOrEqual(0);
    model.free();
  });

  test("BinomialKriging predictBatch returns array of predictions", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const out = model.predictBatch([0.25, 0.5], [0.25, 0.5]);
    expect(Array.isArray(out)).toBe(true);
    expect(out.length).toBe(2);
    out.forEach((p: BinomialPrediction) => {
      expect(p.prevalence).toBeGreaterThanOrEqual(0);
      expect(p.prevalence).toBeLessThanOrEqual(1);
      expect(p.variance).toBeGreaterThanOrEqual(0);
    });
    model.free();
  });

  test("BinomialKriging predictBatchArrays returns Float64Arrays", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const out = model.predictBatchArrays([0.25, 0.5], [0.25, 0.5]);
    expect(out.prevalences).toBeInstanceOf(Float64Array);
    expect(out.logitValues).toBeInstanceOf(Float64Array);
    expect(out.variances).toBeInstanceOf(Float64Array);
    expect(out.prevalences.length).toBe(2);
    model.free();
  });

  test("BinomialKriging.newWithPrior produces valid predictions", () => {
    const alpha = 1;
    const beta = 1;
    const model = BinomialKriging.newWithPrior({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
      prior: { alpha, beta },
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.prevalence)).toBe(true);
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    model.free();
  });

  test("BinomialKriging free() is idempotent", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    model.free();
    model.free();
    expect(() => model.predict(0.5, 0.5)).toThrow(KrigingError);
  });

  test("BinomialKriging predictGrid returns 2D arrays", () => {
    const model = new BinomialKriging({
      lats,
      lons,
      successes,
      trials,
      variogram: { variogramType, nugget, sill, range },
    });
    const out = model.predictGrid({
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
    });
    expect(out.prevalences.length).toBe(2);
    expect(out.prevalences[0].length).toBe(2);
    expect(out.variances.length).toBe(2);
    model.free();
  });

  test("interpolateBinomialToGrid returns grid and frees model", () => {
    const out = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
    });
    expect(out.prevalences.length).toBe(2);
    expect(out.prevalences[0].length).toBe(2);
    expect(out.variances.length).toBe(2);
    expect(Number.isFinite(out.prevalences[0][0])).toBe(true);
    expect(out.fittedVariogram.variogramType).toBe("exponential");
    expect(Number.isFinite(out.fittedVariogram.range)).toBe(true);
    expect(out.cv).toBeUndefined();
    expect(out.buildNotes.calibrationVersion).toBeGreaterThanOrEqual(1);
    expect(out.buildNotes.prior.alpha).toBe(1);
    expect(out.buildNotes.prior.beta).toBe(1);
  });

  test("interpolateBinomialToGrid fits on smoothed logits (matches manual pipeline)", () => {
    const out = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
    });

    const ALPHA = 1;
    const BETA = 1;
    const manualLogits = successes.map((s, i) => {
      const p = (s + ALPHA) / (trials[i] + ALPHA + BETA);
      return Math.log(p / (1 - p));
    });
    const manualFit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values: manualLogits,
      variogramType: "exponential",
      nBins: 12,
    });
    const manualModel = BinomialKriging.fromFittedVariogram({
      lats,
      lons,
      successes,
      trials,
      fittedVariogram: manualFit,
    });
    try {
      const manualGrid = manualModel.predictGrid({
        west: 0,
        south: 0,
        east: 1,
        north: 1,
        xCells: 2,
        yCells: 2,
      });
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          expect(out.prevalences[j][i]).toBeCloseTo(
            manualGrid.prevalences[j][i],
            12
          );
          expect(out.logitValues[j][i]).toBeCloseTo(
            manualGrid.logitValues[j][i],
            12
          );
        }
      }
      expect(out.fittedVariogram.nugget).toBeCloseTo(manualFit.nugget, 12);
      expect(out.fittedVariogram.sill).toBeCloseTo(manualFit.sill, 12);
      expect(out.fittedVariogram.range).toBeCloseTo(manualFit.range, 12);
      expect(out.buildNotes.prior.alpha).toBe(1);
      expect(out.buildNotes.prior.beta).toBe(1);
    } finally {
      manualModel.free();
    }
  });

  test("interpolateBinomialToGrid honors estimator option", () => {
    const classical = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
    });
    const robust = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
      estimator: "cressie-hawkins",
    });
    expect(Number.isFinite(robust.fittedVariogram.range)).toBe(true);
    expect(Number.isFinite(classical.fittedVariogram.range)).toBe(true);
  });

  test("interpolateBinomialToGrid with withCv returns CV summary on both scales", () => {
    const out = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
      withCv: true,
    });
    expect(out.cv).toBeDefined();
    expect(out.cv?.n).toBe(lats.length);
    expect(out.cv?.nEvaluated).toBeLessThanOrEqual(lats.length);
    expect(Number.isFinite(out.cv?.logit.rmse)).toBe(true);
    expect(Number.isFinite(out.cv?.prevalence.rmse)).toBe(true);
  });

  test("interpolateBinomialToGrid with withCv {k} runs k-fold CV", () => {
    const out = interpolateBinomialToGrid({
      lats,
      lons,
      successes,
      trials,
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      variogramType: "exponential",
      nBins: 12,
      withCv: { k: 2 },
    });
    expect(out.cv).toBeDefined();
    expect(out.cv?.n).toBe(lats.length);
  });

  test("BinomialKriging.fromFittedVariogram produces valid predictions", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values: lats.map((_, i) => successes[i] / trials[i]),
      variogramType: "exponential",
      nBins: 12,
    });
    const model = BinomialKriging.fromFittedVariogram({
      lats,
      lons,
      successes,
      trials,
      fittedVariogram: fit,
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.prevalence)).toBe(true);
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    model.free();
  });

  test("BinomialKriging.fromFittedVariogramWithPrior produces valid predictions", () => {
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values: lats.map((_, i) => successes[i] / trials[i]),
      variogramType: "exponential",
      nBins: 12,
    });
    const model = BinomialKriging.fromFittedVariogramWithPrior({
      lats,
      lons,
      successes,
      trials,
      fittedVariogram: fit,
      prior: { alpha: 1, beta: 1 },
    });
    const pred = model.predict(0.5, 0.5);
    expect(Number.isFinite(pred.prevalence)).toBe(true);
    expect(pred.prevalence).toBeGreaterThanOrEqual(0);
    expect(pred.prevalence).toBeLessThanOrEqual(1);
    model.free();
  });
});

describe("Error handling", () => {
  test("fitVariogram with mismatched array lengths throws KrigingError with code", () => {
    try {
      fitVariogram({
        sampleLats: [0, 1],
        sampleLons: [0, 1, 2],
        values: [1, 2, 3],
        variogramType: VariogramType.Gaussian,
      });
    } catch (e) {
      expect(e).toBeInstanceOf(KrigingError);
      expect((e as KrigingError).code).toBe("mismatched_arrays");
      return;
    }
    expect.fail("should have thrown");
  });

  test("OrdinaryKriging with mismatched lats/lons throws KrigingError", () => {
    expect(
      () =>
        new OrdinaryKriging({
          lats: [0, 1],
          lons: [0, 1, 2],
          values: [1, 2, 3],
          variogram: {
            variogramType: "gaussian",
            nugget: 0.01,
            sill: 1,
            range: 100,
          },
        })
    ).toThrow(KrigingError);
  });

  test("BinomialKriging with mismatched arrays throws KrigingError", () => {
    expect(
      () =>
        new BinomialKriging({
          lats: [0, 1],
          lons: [0, 1],
          successes: [1, 2, 3],
          trials: [10, 10, 10],
          variogram: {
            variogramType: "gaussian",
            nugget: 0.01,
            sill: 1,
            range: 100,
          },
        })
    ).toThrow(KrigingError);
  });
});

describe("Search neighborhood (ordinary kriging)", () => {
  const lats = [0, 0, 1, 1, 2, 2];
  const lons = [0, 1, 0, 1, 0, 1];
  const values = [10, 12, 11, 13, 14, 15];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.01,
    sill: 1.0,
    range: 500.0,
  };

  test("setNeighborhood limits to k nearest and is reversible", () => {
    const model = new OrdinaryKriging({ lats, lons, values, variogram });
    try {
      expect(model.neighborhood()).toBeNull();
      model.setNeighborhood({ maxNeighbors: 3 });
      expect(model.neighborhood()).toEqual({ maxNeighbors: 3 });
      const pred = model.predict(0.5, 0.5);
      expect(Number.isFinite(pred.value)).toBe(true);
      model.setNeighborhood();
      expect(model.neighborhood()).toBeNull();
    } finally {
      model.free();
    }
  });

  test("setNeighborhood with invalid radius throws", () => {
    const model = new OrdinaryKriging({ lats, lons, values, variogram });
    try {
      expect(() => model.setNeighborhood({ maxRadius: -1 })).toThrow(
        KrigingError
      );
    } finally {
      model.free();
    }
  });
});

describe("Simple kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const values = [10, 12, 11, 13];

  test("SimpleKriging predicts with known mean", () => {
    const model = new SimpleKriging({
      lats,
      lons,
      values,
      mean: 11.5,
      variogram: {
        variogramType: "exponential",
        nugget: 0.01,
        sill: 1,
        range: 500,
      },
    });
    try {
      expect(model.mean).toBeCloseTo(11.5, 6);
      const pred = model.predict(0.5, 0.5);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
      const batch = model.predictBatchArrays([0.5, 0.25], [0.5, 0.25]);
      expect(batch.values.length).toBe(2);
      expect(batch.variances.length).toBe(2);
    } finally {
      model.free();
    }
  });
});

describe("Universal kriging", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const values = [10, 12, 11, 13];

  test("UniversalKriging with linear trend predicts finite values", () => {
    const model = new UniversalKriging({
      lats,
      lons,
      values,
      trend: "linear",
      variogram: {
        variogramType: "exponential",
        nugget: 0.01,
        sill: 1,
        range: 500,
      },
    });
    try {
      const pred = model.predict(0.5, 0.5);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("UniversalKriging rejects unknown trend", () => {
    expect(
      () =>
        new UniversalKriging({
          lats,
          lons,
          values,
          trend: "cubic" as unknown as "linear",
          variogram: {
            variogramType: "exponential",
            nugget: 0.01,
            sill: 1,
            range: 500,
          },
        })
    ).toThrow(KrigingError);
  });
});

describe("Projected kriging", () => {
  test("ProjectedKriging with 2D anisotropy predicts finite values", () => {
    const xs = [0, 1000, 0, 1000];
    const ys = [0, 0, 1000, 1000];
    const values = [10, 11, 12, 13];
    const model = new ProjectedKriging({
      xs,
      ys,
      values,
      variogram: {
        variogramType: "spherical",
        nugget: 0.1,
        sill: 1.0,
        range: 5000,
      },
      majorAngleDeg: 0,
      rangeRatio: 0.5,
    });
    try {
      const pred = model.predict(500, 500);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("ProjectedKriging rejects range ratio outside (0, 1]", () => {
    expect(
      () =>
        new ProjectedKriging({
          xs: [0, 1, 0, 1],
          ys: [0, 0, 1, 1],
          values: [10, 11, 12, 13],
          variogram: {
            variogramType: "spherical",
            nugget: 0.1,
            sill: 1,
            range: 5,
          },
          majorAngleDeg: 0,
          rangeRatio: 2.0,
        })
    ).toThrow(KrigingError);
  });
});

describe("Projected binomial kriging", () => {
  // 4 corner stations of a 1km square in projected (meter) coordinates with a
  // mild positive logit gradient toward (1, 1).
  const xs = [0, 1000, 0, 1000];
  const ys = [0, 0, 1000, 1000];
  const successes = [3, 7, 4, 9];
  const trials = [10, 12, 9, 15];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.1,
    sill: 1.0,
    range: 1500,
  };

  test("BinomialProjectedKriging.predict returns prevalence in [0, 1] with non-negative variance", () => {
    const model = new BinomialProjectedKriging({
      xs,
      ys,
      successes,
      trials,
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
    });
    try {
      const pred = model.predict(500, 500);
      expect(Number.isFinite(pred.prevalence)).toBe(true);
      expect(Number.isFinite(pred.logitValue)).toBe(true);
      expect(pred.prevalence).toBeGreaterThanOrEqual(0);
      expect(pred.prevalence).toBeLessThanOrEqual(1);
      expect(pred.variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("BinomialProjectedKriging.predictBatch and predictBatchArrays agree", () => {
    const model = new BinomialProjectedKriging({
      xs,
      ys,
      successes,
      trials,
      variogram,
      majorAngleDeg: 30,
      rangeRatio: 0.5,
    });
    try {
      const targets = { xs: [250, 750], ys: [250, 750] };
      const batch = model.predictBatch(targets.xs, targets.ys);
      const arrays = model.predictBatchArrays(targets.xs, targets.ys);
      expect(batch).toHaveLength(2);
      expect(arrays.prevalences).toBeInstanceOf(Float64Array);
      expect(arrays.logitValues).toBeInstanceOf(Float64Array);
      expect(arrays.variances).toBeInstanceOf(Float64Array);
      for (let i = 0; i < 2; i++) {
        expect(arrays.prevalences[i]).toBeCloseTo(batch[i].prevalence, 6);
        expect(arrays.logitValues[i]).toBeCloseTo(batch[i].logitValue, 6);
        expect(arrays.variances[i]).toBeCloseTo(batch[i].variance, 6);
      }
    } finally {
      model.free();
    }
  });

  test("BinomialProjectedKriging.newWithPrior and fromPrecomputedLogits both predict", () => {
    const withPrior = BinomialProjectedKriging.newWithPrior({
      xs,
      ys,
      successes,
      trials,
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
      prior: { alpha: 1, beta: 1 },
    });
    const fromLogits = BinomialProjectedKriging.fromPrecomputedLogits({
      xs,
      ys,
      logits: [-1.2, 0.3, -0.9, 0.6],
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
    });
    try {
      for (const m of [withPrior, fromLogits]) {
        const pred = m.predict(500, 500);
        expect(pred.prevalence).toBeGreaterThanOrEqual(0);
        expect(pred.prevalence).toBeLessThanOrEqual(1);
      }
    } finally {
      withPrior.free();
      fromLogits.free();
    }
  });

  test("BinomialProjectedKriging rejects rangeRatio outside (0, 1]", () => {
    expect(
      () =>
        new BinomialProjectedKriging({
          xs,
          ys,
          successes,
          trials,
          variogram,
          majorAngleDeg: 0,
          rangeRatio: 1.5,
        })
    ).toThrow(KrigingError);
  });

  test("BinomialProjectedKriging.free is idempotent and predict throws after free", () => {
    const model = new BinomialProjectedKriging({
      xs,
      ys,
      successes,
      trials,
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
    });
    model.free();
    expect(() => model.free()).not.toThrow();
    expect(() => model.predict(0, 0)).toThrow(KrigingError);
  });

  test("leaveOneOutBinomialProjected returns one residual per station with both scales", () => {
    const out = leaveOneOutBinomialProjected({
      xs,
      ys,
      successes,
      trials,
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
    });
    expect(out.residuals).toHaveLength(xs.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.observedLogit)).toBe(true);
      expect(Number.isFinite(r.predictedLogit)).toBe(true);
      expect(Number.isFinite(r.observedPrevalence)).toBe(true);
      expect(Number.isFinite(r.predictedPrevalence)).toBe(true);
      expect(r.predictedPrevalence).toBeGreaterThanOrEqual(0);
      expect(r.predictedPrevalence).toBeLessThanOrEqual(1);
    }
    expect(out.summary.n).toBe(xs.length);
    expect(out.summary.nEvaluated).toBe(xs.length);
    expect(Number.isFinite(out.summary.logit.rmse)).toBe(true);
    expect(Number.isFinite(out.summary.prevalence.rmse)).toBe(true);
  });

  test("kFoldBinomialProjected covers every station exactly once", () => {
    const out = kFoldBinomialProjected({
      xs,
      ys,
      successes,
      trials,
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
      k: 2,
    });
    const seen = new Set(out.residuals.map((r) => r.index));
    expect(seen.size).toBe(xs.length);
    for (let i = 0; i < xs.length; i++) expect(seen.has(i)).toBe(true);
  });

  test("conditionalSimulateBinomialProjected is reproducible for a fixed seed", () => {
    const opts = {
      conditioningXs: xs,
      conditioningYs: ys,
      successes,
      trials,
      targetXs: [250, 500, 750],
      targetYs: [250, 500, 750],
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
      seed: 42n,
    } as const;
    const a = conditionalSimulateBinomialProjected(opts);
    const b = conditionalSimulateBinomialProjected(opts);
    expect(a.prevalenceSamples).toEqual(b.prevalenceSamples);
    expect(a.logitSamples).toEqual(b.logitSamples);
    expect(a.prevalenceSamples.length).toBe(3);
    for (const v of a.prevalenceSamples) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  test("conditionalSimulateManyBinomialProjected matches per-realization seeds", () => {
    const targetXs = [250, 750];
    const targetYs = [250, 750];
    const baseSeed = 7n;
    const nRealizations = 3;
    const many = conditionalSimulateManyBinomialProjected({
      conditioningXs: xs,
      conditioningYs: ys,
      successes,
      trials,
      targetXs,
      targetYs,
      variogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
      nRealizations,
      baseSeed,
    });
    expect(many.nRealizations).toBe(nRealizations);
    expect(many.nTargets).toBe(targetXs.length);
    expect(many.prevalenceSamples.length).toBe(nRealizations * targetXs.length);
    for (let k = 0; k < nRealizations; k++) {
      const single = conditionalSimulateBinomialProjected({
        conditioningXs: xs,
        conditioningYs: ys,
        successes,
        trials,
        targetXs,
        targetYs,
        variogram,
        majorAngleDeg: 0,
        rangeRatio: 1,
        seed: baseSeed + BigInt(k),
      });
      const off = k * targetXs.length;
      for (let j = 0; j < targetXs.length; j++) {
        expect(many.prevalenceSamples[off + j]).toBeCloseTo(
          single.prevalenceSamples[j],
          6
        );
        expect(many.logitSamples[off + j]).toBeCloseTo(
          single.logitSamples[j],
          6
        );
      }
    }
  });
});

describe("Empirical variogram (direct)", () => {
  const lats = [0, 0, 1, 1, 2, 2];
  const lons = [0, 1, 0, 1, 0, 1];
  const values = [10, 12, 11, 13, 14, 15];

  test("computeEmpiricalVariogram returns non-empty bins", () => {
    const out = computeEmpiricalVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      nBins: 5,
    });
    expect(out.distances.length).toBe(out.semivariances.length);
    expect(out.counts.length).toBe(out.distances.length);
    expect(out.distances.length).toBeGreaterThan(0);
  });

  test("Cressie-Hawkins estimator produces non-negative semivariances", () => {
    const out = computeEmpiricalVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      nBins: 5,
      estimator: "cressie-hawkins",
    });
    for (const s of out.semivariances) {
      if (Number.isFinite(s)) expect(s).toBeGreaterThanOrEqual(0);
    }
  });

  test("computeDirectionalEmpiricalVariogram respects tolerance", () => {
    const xs = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    const ys = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    const values = xs.map((_, i) => i * 0.5 + ys[i] * 2);
    const out = computeDirectionalEmpiricalVariogram({
      xs,
      ys,
      values,
      maxDistance: 5,
      nBins: 4,
      azimuthDeg: 0,
      toleranceDeg: 15,
    });
    expect(out.distances.length).toBe(4);
  });
});

describe("Cross-validation", () => {
  const lats = [0, 0, 1, 1, 2, 2];
  const lons = [0, 1, 0, 1, 0, 1];
  const values = [10, 12, 11, 13, 14, 15];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("leaveOneOut returns one residual per station", () => {
    const out = leaveOneOut({ lats, lons, values, variogram });
    expect(out.residuals.length).toBe(values.length);
    expect(out.summary.n).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
      expect(r.error).toBeCloseTo(r.observed - r.predicted, 6);
    }
    expect(Number.isFinite(out.summary.rmse)).toBe(true);
  });

  test("kFold returns residuals for all stations", () => {
    const out = kFold({ lats, lons, values, variogram, k: 3 });
    expect(out.residuals.length).toBe(values.length);
    expect(out.summary.n).toBe(values.length);
  });

  test("kFold rejects k < 2", () => {
    expect(() => kFold({ lats, lons, values, variogram, k: 1 })).toThrow(
      KrigingError
    );
  });
});

describe("Cross-validation (per-variant)", () => {
  const lats = [0, 0, 1, 1, 2, 2, 3, 3];
  const lons = [0, 1, 0, 1, 0, 1, 0, 1];
  const values = lats.map((lat, i) => 2 * lat + 3 * lons[i] + 1);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("leaveOneOutSimple returns one residual per station with known mean", () => {
    const out = leaveOneOutSimple({ lats, lons, values, variogram, mean });
    expect(out.residuals.length).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
      expect(Number.isFinite(r.variance)).toBe(true);
    }
  });

  test("kFoldSimple covers every station exactly once", () => {
    const out = kFoldSimple({ lats, lons, values, variogram, mean, k: 4 });
    expect(out.residuals.length).toBe(values.length);
    const seen = new Set<number>();
    for (const r of out.residuals) {
      expect(seen.has(r.index)).toBe(false);
      seen.add(r.index);
    }
    expect(seen.size).toBe(values.length);
  });

  test("leaveOneOutUniversal with constant trend matches leaveOneOut (ordinary)", () => {
    const ok = leaveOneOut({ lats, lons, values, variogram });
    const uk = leaveOneOutUniversal({
      lats,
      lons,
      values,
      variogram,
      trend: "constant",
    });
    expect(uk.residuals.length).toBe(ok.residuals.length);
    for (let i = 0; i < ok.residuals.length; i++) {
      expect(uk.residuals[i].predicted).toBeCloseTo(
        ok.residuals[i].predicted,
        6
      );
    }
  });

  test("kFoldUniversal with linear trend returns finite residuals", () => {
    const out = kFoldUniversal({
      lats,
      lons,
      values,
      variogram,
      trend: "linear",
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
    }
  });

  test("leaveOneOutUniversal rejects unknown trend", () => {
    expect(() =>
      leaveOneOutUniversal({
        lats,
        lons,
        values,
        variogram,
        // @ts-expect-error — deliberately invalid
        trend: "cubic",
      })
    ).toThrow(KrigingError);
  });

  test("leaveOneOutProjected (isotropic) returns residuals in input order", () => {
    // Planar grid; reuse lats/lons as xs/ys for structure.
    const xs = lats;
    const ys = lons;
    const projectedVariogram = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 5.0,
    };
    const out = leaveOneOutProjected({
      xs,
      ys,
      values,
      variogram: projectedVariogram,
      majorAngleDeg: 0,
      rangeRatio: 1,
    });
    expect(out.residuals.length).toBe(values.length);
    for (let i = 0; i < out.residuals.length; i++) {
      expect(out.residuals[i].index).toBe(i);
      expect(Number.isFinite(out.residuals[i].predicted)).toBe(true);
    }
  });

  test("kFoldProjected rejects invalid rangeRatio", () => {
    const xs = lats;
    const ys = lons;
    const projectedVariogram = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 5.0,
    };
    expect(() =>
      kFoldProjected({
        xs,
        ys,
        values,
        variogram: projectedVariogram,
        majorAngleDeg: 0,
        rangeRatio: 2,
        k: 4,
      })
    ).toThrow(KrigingError);
  });
});

describe("Cross-validation (binomial, both scales)", () => {
  // Smooth logit gradient; prevalences ~ 0.1..0.9 across a 4x4 grid.
  const coords: { lat: number; lon: number }[] = [];
  const successes: number[] = [];
  const trials: number[] = [];
  const logistic = (x: number) => 1 / (1 + Math.exp(-x));
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      coords.push({ lat: i, lon: j });
      const p = logistic(-2 + 0.5 * i + 0.5 * j);
      const n = 40;
      successes.push(Math.round(p * n));
      trials.push(n);
    }
  }
  const lats = coords.map((c) => c.lat);
  const lons = coords.map((c) => c.lon);
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 2.0,
    range: 5.0,
  };

  test("leaveOneOutBinomial reports both logit and prevalence scales in input order", () => {
    const out = leaveOneOutBinomial({
      lats,
      lons,
      successes,
      trials,
      variogram,
    });
    expect(out.residuals.length).toBe(lats.length);
    for (let i = 0; i < out.residuals.length; i++) {
      const r = out.residuals[i];
      expect(r.index).toBe(i);
      expect(r.trials).toBe(trials[i]);
      expect(r.successes).toBe(successes[i]);
      expect(Number.isFinite(r.observedLogit)).toBe(true);
      expect(Number.isFinite(r.predictedLogit)).toBe(true);
      expect(Number.isFinite(r.logitVariance)).toBe(true);
      expect(Number.isFinite(r.observedPrevalence)).toBe(true);
      expect(Number.isFinite(r.predictedPrevalence)).toBe(true);
      expect(Number.isFinite(r.prevalenceVariance)).toBe(true);
      expect(r.predictedPrevalence).toBeGreaterThanOrEqual(0);
      expect(r.predictedPrevalence).toBeLessThanOrEqual(1);
      expect(r.logitError).toBeCloseTo(r.observedLogit - r.predictedLogit, 9);
      expect(r.prevalenceError).toBeCloseTo(
        r.observedPrevalence - r.predictedPrevalence,
        9
      );
    }
    expect(out.summary.n).toBe(lats.length);
    expect(out.summary.nEvaluated).toBe(lats.length);
    expect(Number.isFinite(out.summary.logit.rmse)).toBe(true);
    expect(Number.isFinite(out.summary.prevalence.rmse)).toBe(true);
  });

  test("leaveOneOutBinomial handles trials==0 with NaN observations and excluded summary", () => {
    const successesZ = successes.slice();
    const trialsZ = trials.slice();
    successesZ[0] = 0;
    trialsZ[0] = 0;
    const out = leaveOneOutBinomial({
      lats,
      lons,
      successes: successesZ,
      trials: trialsZ,
      variogram,
    });
    const r0 = out.residuals[0];
    expect(r0.trials).toBe(0);
    expect(Number.isNaN(r0.observedLogit)).toBe(true);
    expect(Number.isNaN(r0.observedPrevalence)).toBe(true);
    expect(Number.isNaN(r0.logitError)).toBe(true);
    expect(Number.isNaN(r0.prevalenceError)).toBe(true);
    // Prediction still populated.
    expect(Number.isFinite(r0.predictedLogit)).toBe(true);
    expect(Number.isFinite(r0.predictedPrevalence)).toBe(true);
    // Summary excludes the zero-trials station.
    expect(out.summary.n).toBe(lats.length);
    expect(out.summary.nEvaluated).toBe(lats.length - 1);
    expect(out.summary.logit.n).toBe(lats.length - 1);
    expect(out.summary.prevalence.n).toBe(lats.length - 1);
  });

  test("kFoldBinomial covers every station exactly once", () => {
    const out = kFoldBinomial({
      lats,
      lons,
      successes,
      trials,
      variogram,
      k: 4,
    });
    expect(out.residuals.length).toBe(lats.length);
    const seen = new Set<number>();
    for (const r of out.residuals) {
      expect(seen.has(r.index)).toBe(false);
      seen.add(r.index);
    }
    expect(seen.size).toBe(lats.length);
  });

  test("leaveOneOutBinomial accepts custom Beta(alpha, beta) prior", () => {
    const out = leaveOneOutBinomial({
      lats,
      lons,
      successes,
      trials,
      variogram,
      prior: { alpha: 1, beta: 1 },
    });
    expect(out.residuals.length).toBe(lats.length);
    expect(Number.isFinite(out.summary.prevalence.rmse)).toBe(true);
  });
});

describe("Conditional simulation", () => {
  const condLats = [0, 0, 1, 1];
  const condLons = [0, 1, 0, 1];
  const condValues = [10, 12, 11, 13];
  const targetLats = [0.5, 0.25, 0.75];
  const targetLons = [0.5, 0.25, 0.75];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("conditionalSimulate is deterministic under fixed seed", () => {
    const a = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 42,
    });
    const b = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 42,
    });
    expect(a.length).toBe(targetLats.length);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  test("different seeds produce different realizations", () => {
    const a = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 1,
    });
    const b = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 2,
    });
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  test("conditionalSimulateMany equals manual loop with baseSeed + k", () => {
    const many = conditionalSimulateMany({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      nRealizations: 3,
      baseSeed: 7,
    });
    expect(many).toBeInstanceOf(Float64Array);
    expect(many.length).toBe(3 * targetLats.length);
    const nTargets = targetLats.length;
    for (let k = 0; k < 3; k++) {
      const row = conditionalSimulate({
        conditioningLats: condLats,
        conditioningLons: condLons,
        conditioningValues: condValues,
        targetLats,
        targetLons,
        variogram,
        seed: BigInt(7) + BigInt(k),
      });
      for (let j = 0; j < nTargets; j++) {
        expect(many[k * nTargets + j]).toBe(row[j]);
      }
    }
  });

  test("conditionalSimulateMany rejects non-positive nRealizations", () => {
    expect(() =>
      conditionalSimulateMany({
        conditioningLats: condLats,
        conditioningLons: condLons,
        conditioningValues: condValues,
        targetLats,
        targetLons,
        variogram,
        nRealizations: 0,
      }),
    ).toThrow(KrigingError);
  });

  test("conditionalSimulate accepts the confluent-hypergeometric kernel with two shapes", () => {
    // nu = shape (smoothness), alpha = shape2 (tail decay).
    const chVariogram = {
      variogramType: "confluenthypergeometric" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 500,
      shape: 1.5,
      shape2: 2.0,
    };
    const a = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram: chVariogram,
      seed: 99,
    });
    const b = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram: chVariogram,
      seed: 99,
    });
    expect(a.length).toBe(targetLats.length);
    // Every sample is finite, and the run is reproducible under a fixed seed.
    expect(Array.from(a).every((v) => Number.isFinite(v))).toBe(true);
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  test("confluent-hypergeometric defaults the tail-decay (alpha) when shape2 is omitted", () => {
    // Providing only `shape` (nu) is valid; `alpha` defaults, mirroring how stable/matern
    // default their shape. The run must still produce finite, reproducible samples.
    const opts = {
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram: {
        variogramType: "confluenthypergeometric" as VariogramTypeName,
        nugget: 0.05,
        sill: 1.0,
        range: 500,
        shape: 1.0,
      },
      seed: 3,
    };
    const out = conditionalSimulate(opts);
    expect(out.length).toBe(targetLats.length);
    expect(Array.from(out).every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("Conditional simulation (per-variant)", () => {
  const condLats = [0, 0, 1, 1];
  const condLons = [0, 1, 0, 1];
  const condValues = [10, 12, 11, 13];
  const targetLats = [0.5, 0.25, 0.75];
  const targetLons = [0.5, 0.25, 0.75];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("conditionalSimulateSimple is deterministic and returns finite samples", () => {
    const a = conditionalSimulateSimple({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      mean: 11.5,
      seed: 101,
    });
    const b = conditionalSimulateSimple({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      mean: 11.5,
      seed: 101,
    });
    expect(a.length).toBe(targetLats.length);
    expect(Array.from(a)).toEqual(Array.from(b));
    for (const v of a) expect(Number.isFinite(v)).toBe(true);
  });

  test("conditionalSimulateUniversal with constant trend ≈ conditionalSimulate", () => {
    // Constant universal trend is mathematically equivalent to ordinary kriging.
    // Both use the same seeded RNG path, so realizations should match exactly.
    const ord = conditionalSimulate({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      seed: 7,
    });
    const uni = conditionalSimulateUniversal({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      trend: "constant",
      seed: 7,
    });
    expect(uni.length).toBe(targetLats.length);
    for (let i = 0; i < ord.length; i += 1) {
      expect(Math.abs(uni[i] - ord[i])).toBeLessThan(1e-6);
    }
  });

  test("conditionalSimulateUniversal rejects too-few conditioning points for linear trend", () => {
    expect(() =>
      conditionalSimulateUniversal({
        conditioningLats: [0, 0, 1],
        conditioningLons: [0, 1, 0],
        conditioningValues: [10, 12, 11],
        targetLats: [0.5],
        targetLons: [0.5],
        variogram,
        trend: "linear",
        seed: 0,
      })
    ).toThrow(KrigingError);
  });

  test("conditionalSimulateProjected is deterministic on planar coordinates", () => {
    const xs = [0, 0, 1, 1];
    const ys = [0, 1, 0, 1];
    const values = [10, 12, 11, 13];
    const targetXs = [0.5, 0.25];
    const targetYs = [0.5, 0.75];
    const vg = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 2.0,
    };
    const a = conditionalSimulateProjected({
      conditioningXs: xs,
      conditioningYs: ys,
      conditioningValues: values,
      targetXs,
      targetYs,
      variogram: vg,
      majorAngleDeg: 0,
      rangeRatio: 1,
      seed: 13,
    });
    const b = conditionalSimulateProjected({
      conditioningXs: xs,
      conditioningYs: ys,
      conditioningValues: values,
      targetXs,
      targetYs,
      variogram: vg,
      majorAngleDeg: 0,
      rangeRatio: 1,
      seed: 13,
    });
    expect(a.length).toBe(targetXs.length);
    expect(Array.from(a)).toEqual(Array.from(b));
    for (const v of a) expect(Number.isFinite(v)).toBe(true);
  });
});

describe("Conditional simulation (binomial, both scales)", () => {
  const condLats = [0, 0, 1, 1];
  const condLons = [0, 1, 0, 1];
  const successes = [3, 7, 4, 9];
  const trials = [10, 12, 9, 15];
  const targetLats = [0.5, 0.25];
  const targetLons = [0.5, 0.75];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };

  test("conditionalSimulateBinomial reports both scales consistently", () => {
    const result = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 42,
    });
    expect(result.logitSamples.length).toBe(targetLats.length);
    expect(result.prevalenceSamples.length).toBe(targetLats.length);
    for (let i = 0; i < result.logitSamples.length; i += 1) {
      const logit = result.logitSamples[i];
      const prev = result.prevalenceSamples[i];
      expect(Number.isFinite(logit)).toBe(true);
      expect(prev).toBeGreaterThan(0);
      expect(prev).toBeLessThan(1);
      const expected = 1 / (1 + Math.exp(-logit));
      expect(Math.abs(prev - expected)).toBeLessThan(1e-6);
    }
  });

  test("conditionalSimulateBinomial is deterministic for same seed", () => {
    const a = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 99,
    });
    const b = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 99,
    });
    expect(Array.from(a.logitSamples)).toEqual(Array.from(b.logitSamples));
    expect(Array.from(a.prevalenceSamples)).toEqual(
      Array.from(b.prevalenceSamples)
    );
  });

  test("conditionalSimulateBinomial drops trials === 0 stations from conditioning", () => {
    const withZero = conditionalSimulateBinomial({
      conditioningLats: [...condLats, 0.5],
      conditioningLons: [...condLons, 0.5],
      successes: [...successes, 0],
      trials: [...trials, 0],
      targetLats,
      targetLons,
      variogram,
      seed: 4,
    });
    const without = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 4,
    });
    expect(Array.from(withZero.logitSamples)).toEqual(
      Array.from(without.logitSamples)
    );
    expect(Array.from(withZero.prevalenceSamples)).toEqual(
      Array.from(without.prevalenceSamples)
    );
  });

  test("conditionalSimulateBinomial with custom prior differs from default", () => {
    const defaultPrior = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      seed: 55,
    });
    const customPrior = conditionalSimulateBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      prior: { alpha: 2, beta: 5 },
      seed: 55,
    });
    expect(
      Math.abs(defaultPrior.logitSamples[0] - customPrior.logitSamples[0])
    ).toBeGreaterThan(1e-9);
  });

  test("conditionalSimulateManyBinomial row k matches single call with seed = baseSeed + k", () => {
    const N = 4;
    const baseSeed = 21n;
    const many = conditionalSimulateManyBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      nRealizations: N,
      baseSeed,
    });
    expect(many.nRealizations).toBe(N);
    expect(many.nTargets).toBe(targetLats.length);
    expect(many.logitSamples.length).toBe(N * targetLats.length);
    expect(many.prevalenceSamples.length).toBe(N * targetLats.length);
    for (let k = 0; k < N; k++) {
      const single = conditionalSimulateBinomial({
        conditioningLats: condLats,
        conditioningLons: condLons,
        successes,
        trials,
        targetLats,
        targetLons,
        variogram,
        seed: baseSeed + BigInt(k),
      });
      const off = k * targetLats.length;
      for (let j = 0; j < targetLats.length; j++) {
        expect(many.logitSamples[off + j]).toBe(single.logitSamples[j]);
        expect(many.prevalenceSamples[off + j]).toBe(
          single.prevalenceSamples[j]
        );
      }
    }
  });

  test("conditionalSimulateManyBinomial honors custom prior", () => {
    const defaultPrior = conditionalSimulateManyBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      nRealizations: 2,
      baseSeed: 13,
    });
    const customPrior = conditionalSimulateManyBinomial({
      conditioningLats: condLats,
      conditioningLons: condLons,
      successes,
      trials,
      targetLats,
      targetLons,
      variogram,
      prior: { alpha: 2, beta: 5 },
      nRealizations: 2,
      baseSeed: 13,
    });
    expect(
      Math.abs(defaultPrior.logitSamples[0] - customPrior.logitSamples[0])
    ).toBeGreaterThan(1e-9);
  });

  test("conditionalSimulateManyBinomial rejects non-positive nRealizations", () => {
    expect(() =>
      conditionalSimulateManyBinomial({
        conditioningLats: condLats,
        conditioningLons: condLons,
        successes,
        trials,
        targetLats,
        targetLons,
        variogram,
        nRealizations: 0,
      })
    ).toThrow(KrigingError);
  });
});

describe("Binomial kriging: fromPrecomputedLogits", () => {
  test("builds from externally supplied logits", () => {
    const lats = [0, 0, 1, 1];
    const lons = [0, 1, 0, 1];
    const logits = [-1.5, 0.0, 0.5, 1.5];
    const model = BinomialKriging.fromPrecomputedLogits({
      lats,
      lons,
      logits,
      variogram: {
        variogramType: "exponential",
        nugget: 0.05,
        sill: 1,
        range: 500,
      },
    });
    try {
      const pred = model.predict(0.5, 0.5);
      expect(pred.prevalence).toBeGreaterThanOrEqual(0);
      expect(pred.prevalence).toBeLessThanOrEqual(1);
    } finally {
      model.free();
    }
  });

  test("rejects non-finite logits", () => {
    expect(() =>
      BinomialKriging.fromPrecomputedLogits({
        lats: [0, 0, 1],
        lons: [0, 1, 0],
        logits: [0, Number.POSITIVE_INFINITY, 1],
        variogram: {
          variogramType: "exponential",
          nugget: 0.05,
          sill: 1,
          range: 500,
        },
      })
    ).toThrow(KrigingError);
  });
});

describe("Nested variograms", () => {
  test("evaluateNestedVariogram sums components on the semivariance scale", () => {
    const distances = [0, 100, 500, 1000];
    const out = evaluateNestedVariogram(
      [
        { variogramType: "exponential", nugget: 0.1, sill: 0.5, range: 200 },
        { variogramType: "spherical", nugget: 0.0, sill: 0.5, range: 800 },
      ],
      distances
    );
    expect(out.distances.length).toBe(distances.length);
    expect(out.semivariances.length).toBe(distances.length);
    expect(out.covariances.length).toBe(distances.length);
    for (let i = 0; i < out.semivariances.length; i++) {
      expect(Number.isFinite(out.semivariances[i])).toBe(true);
      expect(out.semivariances[i]).toBeGreaterThanOrEqual(0);
    }
    expect(out.semivariances[out.semivariances.length - 1]).toBeGreaterThan(
      out.semivariances[0]
    );
  });

  test("evaluateNestedVariogram rejects empty components", () => {
    expect(() => evaluateNestedVariogram([], [1, 2, 3])).toThrow(KrigingError);
  });
});

describe("fitVariogram estimator option", () => {
  test("Cressie-Hawkins estimator runs and returns finite params", () => {
    const lats = [0, 0, 1, 1, 2, 2];
    const lons = [0, 1, 0, 1, 0, 1];
    const values = [10, 12, 11, 13, 14, 15];
    const fit = fitVariogram({
      sampleLats: lats,
      sampleLons: lons,
      values,
      variogramType: "exponential",
      estimator: "cressie-hawkins",
      nBins: 5,
    });
    expect(Number.isFinite(fit.nugget)).toBe(true);
    expect(Number.isFinite(fit.sill)).toBe(true);
    expect(Number.isFinite(fit.range)).toBe(true);
  });
});

describe("Uninitialized API", () => {
  test("using API before init throws (run in subprocess)", () => {
    const scriptPath = resolve(
      pkgDir,
      "scripts/assert-uninitialized-throws.mjs"
    );
    const result = spawnSync(process.execPath, [scriptPath], {
      cwd: pkgDir,
      encoding: "utf8",
    });
    expect(result.status).toBe(0);
  });
});

describe("Space-time kriging", () => {
  const lats = [0, 0, 0, 0, 1, 1, 1, 1];
  const lons = [0, 0, 1, 1, 0, 0, 1, 1];
  const times = [0, 1, 0, 1, 0, 1, 0, 1];
  const values = [1, 2, 1.5, 2.5, 1.2, 2.1, 1.8, 2.7];
  const successes = [2, 4, 3, 5, 3, 5, 4, 6];
  const trials = [10, 10, 10, 10, 10, 10, 10, 10];
  const variogram = {
    family: "separable" as const,
    spatial: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 300,
    },
    temporal: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 2,
    },
  };

  test("ordinary model predicts finite value and non-negative variance", () => {
    const model = new SpaceTimeOrdinaryKriging({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    try {
      const { value, variance } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
      expect(variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("ordinary batch-arrays returns typed arrays sized by targets", () => {
    const model = new SpaceTimeOrdinaryKriging({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    try {
      const out = model.predictBatchArrays(
        [0.25, 0.75],
        [0.25, 0.75],
        [0.25, 0.75]
      );
      expect(out.values).toBeInstanceOf(Float64Array);
      expect(out.values.length).toBe(2);
      expect(out.variances.length).toBe(2);
    } finally {
      model.free();
    }
  });

  test("simple model reverts to the supplied mean far outside the data", () => {
    const mean = 100;
    const model = new SpaceTimeSimpleKriging({
      lats,
      lons,
      times,
      values,
      variogram: {
        ...variogram,
        spatial: { ...variogram.spatial, range: 0.5 },
        temporal: { ...variogram.temporal, range: 0.5 },
      },
      mean,
    });
    try {
      const { value } = model.predict(50, 50, 500);
      expect(Math.abs(value - mean)).toBeLessThan(1);
    } finally {
      model.free();
    }
  });

  test("universal model with constant trend predicts finite value", () => {
    const model = new SpaceTimeUniversalKriging({
      lats,
      lons,
      times,
      values,
      variogram,
      trend: "constant",
    });
    try {
      const { value } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
    } finally {
      model.free();
    }
  });

  test("binomial model returns prevalence in the unit interval", () => {
    const model = new SpaceTimeBinomialKriging({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    try {
      const { prevalence, logitValue, variance, prevalenceVariance } =
        model.predict(0.5, 0.5, 0.5);
      expect(prevalence).toBeGreaterThan(0);
      expect(prevalence).toBeLessThan(1);
      expect(Number.isFinite(logitValue)).toBe(true);
      expect(variance).toBeGreaterThanOrEqual(0);
      expect(prevalenceVariance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("projected ordinary model predicts finite value", () => {
    const model = new SpaceTimeProjectedOrdinaryKriging({
      xs: lats,
      ys: lons,
      times,
      values,
      variogram: {
        ...variogram,
        spatial: { ...variogram.spatial, range: 2.0 },
      },
      majorAngleDeg: 0,
      rangeRatio: 1.0,
    });
    try {
      const { value } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
    } finally {
      model.free();
    }
  });

  test("computeEmpiricalSpaceTimeVariogram produces bin arrays of the expected size", () => {
    const emp = computeEmpiricalSpaceTimeVariogram({
      lats,
      lons,
      times,
      values,
      nSpatialBins: 3,
      nTemporalBins: 2,
    });
    expect(emp.nSpatialBins).toBe(3);
    expect(emp.nTemporalBins).toBe(2);
    const expected = 3 * 2;
    expect(emp.semivariances.length).toBe(expected);
    expect(emp.spatialLags.length).toBe(expected);
    expect(emp.temporalLags.length).toBe(expected);
    expect(emp.nPairs.length).toBe(expected);
  });

  test("SpaceTimeOrdinaryKriging.fromFitted matches manual construction", () => {
    const denseLats: number[] = [];
    const denseLons: number[] = [];
    const denseTimes: number[] = [];
    const denseValues: number[] = [];
    for (let i = 0; i < 6; i++) {
      for (let t = 0; t < 5; t++) {
        denseLats.push(i * 0.1);
        denseLons.push(0.05 * t);
        denseTimes.push(t);
        denseValues.push((i + t) * 0.5);
      }
    }
    for (const family of ["separable", "productSum"] as const) {
      const fit = fitSpaceTimeVariogram({
        lats: denseLats,
        lons: denseLons,
        times: denseTimes,
        values: denseValues,
        nSpatialBins: 5,
        nTemporalBins: 5,
        family,
        spatialModel: "exponential",
        temporalModel: "exponential",
      });
      const m1 = SpaceTimeOrdinaryKriging.fromFitted({
        lats: denseLats,
        lons: denseLons,
        times: denseTimes,
        values: denseValues,
        fittedVariogram: fit.fit,
      });
      const m2 = new SpaceTimeOrdinaryKriging({
        lats: denseLats,
        lons: denseLons,
        times: denseTimes,
        values: denseValues,
        variogram: fit.fit,
      });
      try {
        const p1 = m1.predict(0.25, 0.1, 2);
        const p2 = m2.predict(0.25, 0.1, 2);
        expect(p1.value).toBeCloseTo(p2.value, 12);
        expect(p1.variance).toBeCloseTo(p2.variance, 12);
      } finally {
        m1.free();
        m2.free();
      }
    }
  });

  test("camelCase productSum family round-trips through fit and predict", () => {
    const denseLats: number[] = [];
    const denseLons: number[] = [];
    const denseTimes: number[] = [];
    const denseValues: number[] = [];
    for (let i = 0; i < 6; i++) {
      for (let t = 0; t < 5; t++) {
        denseLats.push(i * 0.1);
        denseLons.push(0.05 * t);
        denseTimes.push(t);
        denseValues.push((i + t) * 0.5);
      }
    }
    const fit = fitSpaceTimeVariogram({
      lats: denseLats,
      lons: denseLons,
      times: denseTimes,
      values: denseValues,
      nSpatialBins: 5,
      nTemporalBins: 5,
      family: "productSum",
      spatialModel: "exponential",
      temporalModel: "exponential",
    });
    expect(fit.fit.family).toBe("productSum");
    if (fit.fit.family !== "productSum") throw new Error("unreachable");
    expect(Number.isFinite(fit.fit.k1)).toBe(true);
    expect(Number.isFinite(fit.fit.k2)).toBe(true);
    expect(Number.isFinite(fit.fit.k3)).toBe(true);

    const model = new SpaceTimeOrdinaryKriging({
      lats: denseLats,
      lons: denseLons,
      times: denseTimes,
      values: denseValues,
      variogram: fit.fit,
    });
    try {
      const { value, variance } = model.predict(0.25, 0.1, 2);
      expect(Number.isFinite(value)).toBe(true);
      expect(variance).toBeGreaterThanOrEqual(0);
    } finally {
      model.free();
    }
  });

  test("SpaceTimeOrdinaryKriging.predictGridAtTime matches manual batch predict", () => {
    const model = new SpaceTimeOrdinaryKriging({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    try {
      const grid = model.predictGridAtTime({
        west: 0,
        south: 0,
        east: 1,
        north: 1,
        xCells: 3,
        yCells: 2,
        time: 0.5,
      });
      expect(grid.values.length).toBe(2);
      expect(grid.values[0].length).toBe(3);
      expect(grid.variances.length).toBe(2);
      expect(grid.variances[0].length).toBe(3);
      const manualLats: number[] = [];
      const manualLons: number[] = [];
      const manualTimes: number[] = [];
      const latStep = 0.5;
      const lonStep = 1 / 3;
      for (let j = 0; j < 2; j++) {
        const lat = 0 + (j + 0.5) * latStep;
        for (let i = 0; i < 3; i++) {
          manualLats.push(lat);
          manualLons.push(0 + (i + 0.5) * lonStep);
          manualTimes.push(0.5);
        }
      }
      const out = model.predictBatchArrays(manualLats, manualLons, manualTimes);
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 3; i++) {
          const flatIdx = j * 3 + i;
          expect(grid.values[j][i]).toBeCloseTo(out.values[flatIdx], 12);
          expect(grid.variances[j][i]).toBeCloseTo(out.variances[flatIdx], 12);
        }
      }
    } finally {
      model.free();
    }
  });

  test("SpaceTimeBinomialKriging.predictGridAtTime returns 2D prevalence grids", () => {
    const model = new SpaceTimeBinomialKriging({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    try {
      const grid = model.predictGridAtTime({
        west: 0,
        south: 0,
        east: 1,
        north: 1,
        xCells: 2,
        yCells: 2,
        time: 0.5,
      });
      expect(grid.prevalences.length).toBe(2);
      expect(grid.prevalences[0].length).toBe(2);
      expect(grid.logitValues.length).toBe(2);
      expect(grid.variances.length).toBe(2);
      expect(grid.prevalenceVariances.length).toBe(2);
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          expect(grid.prevalences[j][i]).toBeGreaterThanOrEqual(0);
          expect(grid.prevalences[j][i]).toBeLessThanOrEqual(1);
          expect(grid.variances[j][i]).toBeGreaterThanOrEqual(0);
        }
      }
    } finally {
      model.free();
    }
  });

  test("SpaceTimeBinomialKriging.predictAtDate matches predict with timesFromDates", () => {
    // Model was built with `times = [0, 1, 0, 1, ...]` interpreted as days;
    // pick a calendar date 12 hours past the Unix epoch -> time = 0.5 days.
    const model = new SpaceTimeBinomialKriging({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    try {
      const date = new Date(12 * 3_600_000); // 0.5 days after epoch
      const fromDate = model.predictAtDate(0.5, 0.5, date);
      const fromTime = model.predict(0.5, 0.5, 0.5);
      expect(fromDate.prevalence).toBeCloseTo(fromTime.prevalence, 12);
      expect(fromDate.logitValue).toBeCloseTo(fromTime.logitValue, 12);
      expect(fromDate.variance).toBeCloseTo(fromTime.variance, 12);
    } finally {
      model.free();
    }
  });

  test("SpaceTimeBinomialKriging date-aware predict batch helpers honor timeUnit", () => {
    const model = new SpaceTimeBinomialKriging({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    try {
      // The date helpers project Date -> numeric time using `timeUnit`/`epoch`;
      // they DO NOT know what unit the model was trained in. Here we deliberately
      // request "hours" so the conversion emits 12 (hours past epoch), and verify
      // the same numeric value reaches `predictBatchArrays` directly. Callers in
      // production would pass the same unit they built the model with.
      const date = new Date(12 * 3_600_000);
      const dates = [date, date];
      const opts = { timeUnit: "hours" as const };
      const fromDates = model.predictBatchAtDates([0.5, 0.5], [0.5, 0.5], dates, opts);
      const fromArrays = model.predictBatchArraysAtDates(
        [0.5, 0.5],
        [0.5, 0.5],
        dates,
        opts
      );
      const direct = model.predictBatchArrays([0.5, 0.5], [0.5, 0.5], [12, 12]);
      for (let i = 0; i < 2; i++) {
        expect(fromDates[i].prevalence).toBeCloseTo(direct.prevalences[i], 12);
        expect(fromArrays.prevalences[i]).toBeCloseTo(
          direct.prevalences[i],
          12
        );
        expect(fromArrays.logitValues[i]).toBeCloseTo(
          direct.logitValues[i],
          12
        );
      }
    } finally {
      model.free();
    }
  });

  test("SpaceTimeBinomialKriging.predictGridAtDate equals predictGridAtTime", () => {
    const model = new SpaceTimeBinomialKriging({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    try {
      const date = new Date(12 * 3_600_000); // 0.5 days
      const fromDate = model.predictGridAtDate({
        west: 0,
        south: 0,
        east: 1,
        north: 1,
        xCells: 2,
        yCells: 2,
        date,
      });
      const fromTime = model.predictGridAtTime({
        west: 0,
        south: 0,
        east: 1,
        north: 1,
        xCells: 2,
        yCells: 2,
        time: 0.5,
      });
      for (let j = 0; j < 2; j++) {
        for (let i = 0; i < 2; i++) {
          expect(fromDate.prevalences[j][i]).toBeCloseTo(
            fromTime.prevalences[j][i],
            12
          );
          expect(fromDate.logitValues[j][i]).toBeCloseTo(
            fromTime.logitValues[j][i],
            12
          );
        }
      }
    } finally {
      model.free();
    }
  });

  test("simulateBinomialSpaceTimeGrid* date-aware helpers match numeric versions", () => {
    const epoch = new Date(Date.UTC(2024, 0, 1));
    // condDates for `times = [0, 1, 0, 1, ...]` (days from epoch)
    const condDates = times.map(
      (t) => new Date(epoch.getTime() + t * 86_400_000)
    );
    const targetDate = new Date(epoch.getTime() + 0.5 * 86_400_000); // 0.5 days
    const numericOpts = {
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
      time: 0.5,
      seed: 7n,
    } as const;
    const dateOpts = {
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 2,
      yCells: 2,
      lats,
      lons,
      dates: condDates,
      successes,
      trials,
      variogram,
      date: targetDate,
      timeUnit: "days" as const,
      epoch,
      seed: 7n,
    } as const;
    const a = simulateBinomialSpaceTimeGrid(numericOpts);
    const b = simulateBinomialSpaceTimeGridAtDate(dateOpts);
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        expect(b.prevalences[j][i]).toBeCloseTo(a.prevalences[j][i], 12);
        expect(b.logitSamples[j][i]).toBeCloseTo(a.logitSamples[j][i], 12);
      }
    }

    const ensembleA = simulateBinomialSpaceTimeGridEnsemble({
      ...numericOpts,
      seed: undefined,
      nRealizations: 3,
      baseSeed: 11n,
    });
    const ensembleB = simulateBinomialSpaceTimeGridEnsembleAtDate({
      ...dateOpts,
      seed: undefined,
      nRealizations: 3,
      baseSeed: 11n,
    });
    expect(ensembleB.prevalenceSamples).toEqual(ensembleA.prevalenceSamples);
    expect(ensembleB.logitSamples).toEqual(ensembleA.logitSamples);
    expect(ensembleB.nRealizations).toBe(3);
    expect(ensembleB.nTargets).toBe(4);

    const summaryA = simulateBinomialSpaceTimeGridSummary({
      ...numericOpts,
      seed: undefined,
      nRealizations: 3,
      baseSeed: 11n,
      quantiles: [0.5],
    });
    const summaryB = simulateBinomialSpaceTimeGridSummaryAtDate({
      ...dateOpts,
      seed: undefined,
      nRealizations: 3,
      baseSeed: 11n,
      quantiles: [0.5],
    });
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i < 2; i++) {
        expect(summaryB.meanPrevalence[j][i]).toBeCloseTo(
          summaryA.meanPrevalence[j][i],
          12
        );
        expect(summaryB.quantiles[0].values[j][i]).toBeCloseTo(
          summaryA.quantiles[0].values[j][i],
          12
        );
      }
    }
  });

  test("fitSpaceTimeVariogram returns a separable fit with finite residuals", () => {
    const denseLats: number[] = [];
    const denseLons: number[] = [];
    const denseTimes: number[] = [];
    const denseValues: number[] = [];
    for (let i = 0; i < 6; i++) {
      for (let t = 0; t < 5; t++) {
        denseLats.push(i * 0.1);
        denseLons.push(0.05 * t);
        denseTimes.push(t);
        denseValues.push((i + t) * 0.5);
      }
    }
    const result = fitSpaceTimeVariogram({
      lats: denseLats,
      lons: denseLons,
      times: denseTimes,
      values: denseValues,
      nSpatialBins: 5,
      nTemporalBins: 5,
      family: "separable",
      spatialModel: "exponential",
      temporalModel: "exponential",
    });
    expect(result.fit.family).toBe("separable");
    expect(result.fit.spatial.variogramType).toBe("exponential");
    expect(result.fit.temporal.variogramType).toBe("exponential");
    expect(Number.isFinite(result.fit.residuals)).toBe(true);
  });

  test("mismatched arrays raise a KrigingError with a code", () => {
    expect(
      () =>
        new SpaceTimeOrdinaryKriging({
          lats: [0, 0],
          lons: [0, 1, 2],
          times: [0, 1],
          values: [1, 2],
          variogram,
        })
    ).toThrow(KrigingError);
  });

  test("using block disposes ST models and subsequent predict throws model_freed", () => {
    let freedAddress: SpaceTimeOrdinaryKriging;
    {
      using model = new SpaceTimeOrdinaryKriging({
        lats,
        lons,
        times,
        values,
        variogram,
      });
      freedAddress = model;
      const { value } = model.predict(0.5, 0.5, 0.5);
      expect(Number.isFinite(value)).toBe(true);
    }
    let err: unknown;
    try {
      freedAddress.predict(0.5, 0.5, 0.5);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(KrigingError);
    expect((err as KrigingError).code).toBe("model_freed");
  });
});

describe("Space-time cross-validation", () => {
  const lats = [0, 0, 0, 0, 1, 1, 1, 1];
  const lons = [0, 0, 1, 1, 0, 0, 1, 1];
  const times = [0, 1, 0, 1, 0, 1, 0, 1];
  const values = [1, 2, 1.5, 2.5, 1.2, 2.1, 1.8, 2.7];
  const successes = [2, 4, 3, 5, 3, 5, 4, 6];
  const trials = [10, 10, 10, 10, 10, 10, 10, 10];
  const variogram = {
    family: "separable" as const,
    spatial: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 300,
    },
    temporal: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 2,
    },
  };

  test("leaveOneOutSpaceTime returns one residual per station", () => {
    const out = leaveOneOutSpaceTime({ lats, lons, times, values, variogram });
    expect(out.residuals.length).toBe(values.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predicted)).toBe(true);
      expect(r.variance).toBeGreaterThanOrEqual(0);
    }
    expect(out.summary.n).toBe(values.length);
  });

  test("kFoldSpaceTime returns residuals for all stations", () => {
    const out = kFoldSpaceTime({
      lats,
      lons,
      times,
      values,
      variogram,
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("kFoldSpaceTime rejects k < 2", () => {
    expect(() =>
      kFoldSpaceTime({ lats, lons, times, values, variogram, k: 1 })
    ).toThrow(KrigingError);
  });

  test("leaveOneOutSpaceTimeSimple honors known mean", () => {
    const out = leaveOneOutSpaceTimeSimple({
      lats,
      lons,
      times,
      values,
      variogram,
      mean: 2,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("kFoldSpaceTimeSimple returns residuals for all stations", () => {
    const out = kFoldSpaceTimeSimple({
      lats,
      lons,
      times,
      values,
      variogram,
      mean: 2,
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("leaveOneOutSpaceTimeUniversal with constant trend matches ordinary CV", () => {
    const ok = leaveOneOutSpaceTime({
      lats,
      lons,
      times,
      values,
      variogram,
    });
    const uk = leaveOneOutSpaceTimeUniversal({
      lats,
      lons,
      times,
      values,
      variogram,
      trend: "constant",
    });
    expect(uk.residuals.length).toBe(ok.residuals.length);
    for (let i = 0; i < ok.residuals.length; i++) {
      expect(uk.residuals[i].predicted).toBeCloseTo(
        ok.residuals[i].predicted,
        6
      );
    }
  });

  test("kFoldSpaceTimeUniversal returns residuals for all stations", () => {
    const out = kFoldSpaceTimeUniversal({
      lats,
      lons,
      times,
      values,
      variogram,
      trend: "linearInTime",
      k: 4,
    });
    expect(out.residuals.length).toBe(values.length);
  });

  test("leaveOneOutSpaceTimeBinomial returns dual-scale residuals", () => {
    const out = leaveOneOutSpaceTimeBinomial({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
    });
    expect(out.residuals.length).toBe(successes.length);
    for (const r of out.residuals) {
      expect(Number.isFinite(r.predictedLogit)).toBe(true);
      expect(r.predictedPrevalence).toBeGreaterThan(0);
      expect(r.predictedPrevalence).toBeLessThan(1);
    }
  });

  test("kFoldSpaceTimeBinomial returns residuals for all stations", () => {
    const out = kFoldSpaceTimeBinomial({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
      k: 4,
    });
    expect(out.residuals.length).toBe(successes.length);
  });

  test("binomial CV accepts custom Beta prior via `prior`", () => {
    const out = leaveOneOutSpaceTimeBinomial({
      lats,
      lons,
      times,
      successes,
      trials,
      variogram,
      prior: { alpha: 2, beta: 5 },
    });
    expect(out.residuals.length).toBe(successes.length);
  });
});

describe("Space-time conditional simulation", () => {
  const conditioningLats = [0, 0, 0, 0, 1, 1, 1, 1];
  const conditioningLons = [0, 0, 1, 1, 0, 0, 1, 1];
  const conditioningTimes = [0, 1, 0, 1, 0, 1, 0, 1];
  const conditioningValues = [1, 2, 1.5, 2.5, 1.2, 2.1, 1.8, 2.7];
  const successes = [2, 4, 3, 5, 3, 5, 4, 6];
  const trials = [10, 10, 10, 10, 10, 10, 10, 10];
  const targetLats = [0.25, 0.75];
  const targetLons = [0.25, 0.75];
  const targetTimes = [0.5, 0.5];
  const variogram = {
    family: "separable" as const,
    spatial: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 300,
    },
    temporal: {
      variogramType: "exponential" as const,
      nugget: 0.01,
      sill: 1.0,
      range: 2,
    },
  };

  test("conditionalSimulateSpaceTime is deterministic under fixed seed", () => {
    const a = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 42n,
    });
    const b = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 42n,
    });
    expect(Array.from(a)).toEqual(Array.from(b));
  });

  test("conditionalSimulateSpaceTime differs across seeds", () => {
    const a = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 1n,
    });
    const b = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 2n,
    });
    expect(Array.from(a)).not.toEqual(Array.from(b));
  });

  test("conditionalSimulateManySpaceTime equals manual loop with baseSeed + k", () => {
    const many = conditionalSimulateManySpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      nRealizations: 3,
      baseSeed: 5,
    });
    expect(many).toBeInstanceOf(Float64Array);
    const nTargets = targetLats.length;
    expect(many.length).toBe(3 * nTargets);
    for (let k = 0; k < 3; k++) {
      const row = conditionalSimulateSpaceTime({
        conditioningLats,
        conditioningLons,
        conditioningTimes,
        conditioningValues,
        targetLats,
        targetLons,
        targetTimes,
        variogram,
        seed: BigInt(5) + BigInt(k),
      });
      for (let j = 0; j < nTargets; j++) {
        expect(many[k * nTargets + j]).toBe(row[j]);
      }
    }
  });

  test("conditionalSimulateSpaceTimeSimple honors known mean", () => {
    const out = conditionalSimulateSpaceTimeSimple({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      mean: 2,
      seed: 7n,
    });
    expect(out.length).toBe(targetLats.length);
    for (const v of out) expect(Number.isFinite(v)).toBe(true);
  });

  test("conditionalSimulateSpaceTimeUniversal with constant trend ≈ ordinary", () => {
    const ord = conditionalSimulateSpaceTime({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 11n,
    });
    const uk = conditionalSimulateSpaceTimeUniversal({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      conditioningValues,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      trend: "constant",
      seed: 11n,
    });
    expect(uk.length).toBe(ord.length);
    for (let i = 0; i < ord.length; i++) {
      expect(uk[i]).toBeCloseTo(ord[i], 5);
    }
  });

  test("conditionalSimulateSpaceTimeBinomial returns logit and prevalence samples", () => {
    const out = conditionalSimulateSpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 13n,
    });
    expect(out.logitSamples.length).toBe(targetLats.length);
    expect(out.prevalenceSamples.length).toBe(targetLats.length);
    for (let i = 0; i < targetLats.length; i++) {
      expect(Number.isFinite(out.logitSamples[i])).toBe(true);
      expect(out.prevalenceSamples[i]).toBeGreaterThan(0);
      expect(out.prevalenceSamples[i]).toBeLessThan(1);
    }
  });

  test("conditionalSimulateSpaceTimeBinomial is deterministic under fixed seed", () => {
    const a = conditionalSimulateSpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 99n,
    });
    const b = conditionalSimulateSpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      seed: 99n,
    });
    expect(Array.from(a.logitSamples)).toEqual(Array.from(b.logitSamples));
    expect(Array.from(a.prevalenceSamples)).toEqual(
      Array.from(b.prevalenceSamples)
    );
  });

  test("conditionalSimulateManySpaceTimeBinomial row k matches single call", () => {
    const N = 3;
    const baseSeed = 41n;
    const many = conditionalSimulateManySpaceTimeBinomial({
      conditioningLats,
      conditioningLons,
      conditioningTimes,
      successes,
      trials,
      targetLats,
      targetLons,
      targetTimes,
      variogram,
      nRealizations: N,
      baseSeed,
    });
    expect(many.nRealizations).toBe(N);
    expect(many.nTargets).toBe(targetLats.length);
    for (let k = 0; k < N; k++) {
      const single = conditionalSimulateSpaceTimeBinomial({
        conditioningLats,
        conditioningLons,
        conditioningTimes,
        successes,
        trials,
        targetLats,
        targetLons,
        targetTimes,
        variogram,
        seed: baseSeed + BigInt(k),
      });
      const off = k * targetLats.length;
      for (let j = 0; j < targetLats.length; j++) {
        expect(many.logitSamples[off + j]).toBe(single.logitSamples[j]);
        expect(many.prevalenceSamples[off + j]).toBe(
          single.prevalenceSamples[j]
        );
      }
    }
  });
});

describe("Ensemble aggregators", () => {
  // Hand-built 3 realizations × 4 targets matrix:
  //   row 0: [1, 2, 3, 4]
  //   row 1: [2, 4, 6, 8]
  //   row 2: [3, 6, 9, 12]
  const samples = new Float64Array([1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12]);
  const N = 3;
  const T = 4;

  test("ensembleMean computes per-target column mean", () => {
    const m = ensembleMean(samples, N, T);
    expect(Array.from(m)).toEqual([2, 4, 6, 8]);
  });

  test("ensembleVariance computes unbiased per-target variance (n-1 denom)", () => {
    const v = ensembleVariance(samples, N, T);
    expect(Array.from(v)).toEqual([1, 4, 9, 16]);
  });

  test("ensembleVariance rejects nRealizations < 2", () => {
    expect(() =>
      ensembleVariance(new Float64Array([1, 2, 3, 4]), 1, T)
    ).toThrow(KrigingError);
  });

  test("ensembleQuantiles returns linearly interpolated quantiles per target", () => {
    const q = ensembleQuantiles(samples, N, T, [0, 0.5, 1]);
    expect(q.length).toBe(3 * T);
    expect(Array.from(q.subarray(0, T))).toEqual([1, 2, 3, 4]);
    expect(Array.from(q.subarray(T, 2 * T))).toEqual([2, 4, 6, 8]);
    expect(Array.from(q.subarray(2 * T, 3 * T))).toEqual([3, 6, 9, 12]);
  });

  test("ensembleQuantiles rejects probabilities outside [0,1]", () => {
    expect(() => ensembleQuantiles(samples, N, T, [-0.1])).toThrow(
      KrigingError
    );
    expect(() => ensembleQuantiles(samples, N, T, [1.1])).toThrow(KrigingError);
  });

  test("ensembleQuantiles rejects empty quantile list", () => {
    expect(() => ensembleQuantiles(samples, N, T, [])).toThrow(KrigingError);
  });

  test("ensembleExceedanceProbability counts strict exceedances per target", () => {
    // threshold 2 → cells > 2 by column:
    //   col 0: {1,2,3} → 1/3
    //   col 1: {2,4,6} → 2/3
    //   col 2: {3,6,9} → 3/3
    //   col 3: {4,8,12} → 3/3
    const p = ensembleExceedanceProbability(samples, N, T, 2);
    expect(p[0]).toBeCloseTo(1 / 3, 12);
    expect(p[1]).toBeCloseTo(2 / 3, 12);
    expect(p[2]).toBe(1);
    expect(p[3]).toBe(1);
  });

  test("aggregator dimension mismatch throws KrigingError", () => {
    expect(() => ensembleMean(new Float64Array(11), N, T)).toThrow(
      KrigingError
    );
  });

  test("aggregators integrate with conditionalSimulateMany output", () => {
    const condLats = [0, 0, 1, 1];
    const condLons = [0, 1, 0, 1];
    const condValues = [10, 12, 11, 13];
    const targetLats = [0.5, 0.25, 0.75];
    const targetLons = [0.5, 0.25, 0.75];
    const variogram = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 500,
    };
    const Nr = 8;
    const buf = conditionalSimulateMany({
      conditioningLats: condLats,
      conditioningLons: condLons,
      conditioningValues: condValues,
      targetLats,
      targetLons,
      variogram,
      nRealizations: Nr,
      baseSeed: 314159,
    });
    const m = ensembleMean(buf, Nr, targetLats.length);
    const v = ensembleVariance(buf, Nr, targetLats.length);
    expect(m.length).toBe(targetLats.length);
    expect(v.length).toBe(targetLats.length);
    for (let j = 0; j < targetLats.length; j++) {
      expect(Number.isFinite(m[j])).toBe(true);
      expect(v[j]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("Binomial grid simulation", () => {
  const lats = [0, 0, 1, 1];
  const lons = [0, 1, 0, 1];
  const successes = [3, 7, 4, 9];
  const trials = [10, 12, 9, 15];
  const variogram = {
    variogramType: "exponential" as VariogramTypeName,
    nugget: 0.05,
    sill: 1.0,
    range: 500,
  };
  const bounds = {
    west: 0,
    south: 0,
    east: 1,
    north: 1,
    xCells: 3,
    yCells: 2,
  };

  test("gridCellCenters lays out (j, i) with i fastest, j running south->north", () => {
    const { lats: gLats, lons: gLons } = gridCellCenters(bounds);
    expect(gLats.length).toBe(bounds.xCells * bounds.yCells);
    expect(gLons.length).toBe(bounds.xCells * bounds.yCells);
    const dx = (bounds.east - bounds.west) / bounds.xCells;
    const dy = (bounds.north - bounds.south) / bounds.yCells;
    for (let j = 0; j < bounds.yCells; j++) {
      for (let i = 0; i < bounds.xCells; i++) {
        const idx = j * bounds.xCells + i;
        expect(gLats[idx]).toBeCloseTo(bounds.south + (j + 0.5) * dy, 12);
        expect(gLons[idx]).toBeCloseTo(bounds.west + (i + 0.5) * dx, 12);
      }
    }
  });

  test("reshapeGridRow inverts gridCellCenters layout", () => {
    const flat = new Float64Array(bounds.xCells * bounds.yCells);
    for (let i = 0; i < flat.length; i++) flat[i] = i;
    const grid = reshapeGridRow(flat, bounds.xCells, bounds.yCells);
    expect(grid.length).toBe(bounds.yCells);
    expect(grid[0].length).toBe(bounds.xCells);
    for (let j = 0; j < bounds.yCells; j++) {
      for (let i = 0; i < bounds.xCells; i++) {
        expect(grid[j][i]).toBe(j * bounds.xCells + i);
      }
    }
  });

  test("simulateBinomialGrid returns dual-scale 2D arrays consistent with logistic()", () => {
    const grid = simulateBinomialGrid({
      ...bounds,
      lats,
      lons,
      successes,
      trials,
      variogram,
      seed: 17,
    });
    expect(grid.logitSamples.length).toBe(bounds.yCells);
    expect(grid.prevalences.length).toBe(bounds.yCells);
    for (let j = 0; j < bounds.yCells; j++) {
      expect(grid.logitSamples[j].length).toBe(bounds.xCells);
      expect(grid.prevalences[j].length).toBe(bounds.xCells);
      for (let i = 0; i < bounds.xCells; i++) {
        const l = grid.logitSamples[j][i];
        const p = grid.prevalences[j][i];
        expect(Number.isFinite(l)).toBe(true);
        expect(p).toBeGreaterThan(0);
        expect(p).toBeLessThan(1);
        const expected = 1 / (1 + Math.exp(-l));
        expect(Math.abs(p - expected)).toBeLessThan(1e-6);
      }
    }
  });

  test("simulateBinomialGrid is deterministic for same seed", () => {
    const a = simulateBinomialGrid({
      ...bounds,
      lats,
      lons,
      successes,
      trials,
      variogram,
      seed: 123,
    });
    const b = simulateBinomialGrid({
      ...bounds,
      lats,
      lons,
      successes,
      trials,
      variogram,
      seed: 123,
    });
    for (let j = 0; j < bounds.yCells; j++) {
      for (let i = 0; i < bounds.xCells; i++) {
        expect(a.prevalences[j][i]).toBe(b.prevalences[j][i]);
      }
    }
  });

  test("simulateBinomialGridEnsemble row k matches simulateBinomialGrid(seed = baseSeed + k)", () => {
    const N = 3;
    const baseSeed = 42n;
    const ens = simulateBinomialGridEnsemble({
      ...bounds,
      lats,
      lons,
      successes,
      trials,
      variogram,
      nRealizations: N,
      baseSeed,
    });
    const nTargets = bounds.xCells * bounds.yCells;
    expect(ens.nRealizations).toBe(N);
    expect(ens.nTargets).toBe(nTargets);
    expect(ens.logitSamples.length).toBe(N * nTargets);
    expect(ens.prevalenceSamples.length).toBe(N * nTargets);
    for (let k = 0; k < N; k++) {
      const single = simulateBinomialGrid({
        ...bounds,
        lats,
        lons,
        successes,
        trials,
        variogram,
        seed: baseSeed + BigInt(k),
      });
      for (let j = 0; j < bounds.yCells; j++) {
        for (let i = 0; i < bounds.xCells; i++) {
          const off = k * nTargets + j * bounds.xCells + i;
          expect(ens.prevalenceSamples[off]).toBe(single.prevalences[j][i]);
          expect(ens.logitSamples[off]).toBe(single.logitSamples[j][i]);
        }
      }
    }
  });

  test("simulateBinomialGridSummary returns mean/variance maps with correct shape", () => {
    const N = 8;
    const summary = simulateBinomialGridSummary({
      ...bounds,
      lats,
      lons,
      successes,
      trials,
      variogram,
      nRealizations: N,
      baseSeed: 7,
      quantiles: [0.025, 0.5, 0.975],
      exceedanceThresholds: [0.5],
    });
    expect(summary.nRealizations).toBe(N);
    expect(summary.summarizeOn).toBe("prevalence");
    expect(summary.meanLogit.length).toBe(bounds.yCells);
    expect(summary.meanLogit[0].length).toBe(bounds.xCells);
    expect(summary.meanPrevalence.length).toBe(bounds.yCells);
    expect(summary.varianceLogit.length).toBe(bounds.yCells);
    expect(summary.quantiles).toHaveLength(3);
    for (const q of summary.quantiles) {
      expect(q.values.length).toBe(bounds.yCells);
      expect(q.values[0].length).toBe(bounds.xCells);
    }
    expect(summary.exceedances).toHaveLength(1);
    expect(summary.exceedances[0].threshold).toBe(0.5);
    for (let j = 0; j < bounds.yCells; j++) {
      for (let i = 0; i < bounds.xCells; i++) {
        const m = summary.meanPrevalence[j][i];
        expect(m).toBeGreaterThanOrEqual(0);
        expect(m).toBeLessThanOrEqual(1);
        expect(summary.varianceLogit[j][i]).toBeGreaterThanOrEqual(0);
        const ex = summary.exceedances[0].values[j][i];
        expect(ex).toBeGreaterThanOrEqual(0);
        expect(ex).toBeLessThanOrEqual(1);
      }
    }
  });

  test("simulateBinomialGridSummary respects summarizeOn = logit", () => {
    const summary = simulateBinomialGridSummary({
      ...bounds,
      lats,
      lons,
      successes,
      trials,
      variogram,
      nRealizations: 4,
      baseSeed: 11,
      quantiles: [0.5],
      summarizeOn: "logit",
    });
    expect(summary.summarizeOn).toBe("logit");
    for (let j = 0; j < bounds.yCells; j++) {
      for (let i = 0; i < bounds.xCells; i++) {
        // Logit-scale quantile must be a finite real (not bounded to [0,1]).
        expect(Number.isFinite(summary.quantiles[0].values[j][i])).toBe(true);
      }
    }
  });

  test("simulateBinomialGrid rejects degenerate grid bounds", () => {
    expect(() =>
      simulateBinomialGrid({
        ...bounds,
        west: 1,
        east: 1,
        lats,
        lons,
        successes,
        trials,
        variogram,
        seed: 0,
      })
    ).toThrow(KrigingError);
    expect(() =>
      simulateBinomialGrid({
        ...bounds,
        xCells: 0,
        lats,
        lons,
        successes,
        trials,
        variogram,
        seed: 0,
      })
    ).toThrow(KrigingError);
  });
});

describe("Polygon aggregation over ensembles", () => {
  // Synthetic ensemble: 5 realizations × 6 targets, prevalence = (k * 0.1) + j * 0.01
  // and a logit field = -1 + 0.2 * k + 0.05 * j.
  function buildEnsemble(
    nRealizations: number,
    nTargets: number,
    p: (k: number, j: number) => number,
    l: (k: number, j: number) => number
  ) {
    const prev = new Float64Array(nRealizations * nTargets);
    const logit = new Float64Array(nRealizations * nTargets);
    for (let k = 0; k < nRealizations; k++) {
      const off = k * nTargets;
      for (let j = 0; j < nTargets; j++) {
        prev[off + j] = p(k, j);
        logit[off + j] = l(k, j);
      }
    }
    return {
      nRealizations,
      nTargets,
      logitSamples: logit,
      prevalenceSamples: prev,
    };
  }

  test("aggregatePrevalenceByPolygon: simple area mean matches hand calculation", () => {
    const ens = buildEnsemble(
      5,
      6,
      (k, j) => k * 0.1 + j * 0.01,
      (k, j) => -1 + 0.2 * k + 0.05 * j
    );
    const polygons = [
      { id: "all", indices: [0, 1, 2, 3, 4, 5], weights: [1, 1, 1, 1, 1, 1] },
      { id: "right-half", indices: [3, 4, 5], weights: [1, 1, 1] },
    ];
    const result = aggregatePrevalenceByPolygon({
      ensemble: ens,
      polygons,
      quantiles: [0.0, 0.5, 1.0],
    });
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("all");
    expect(result[0].summarizeOn).toBe("prevalence");
    expect(result[0].nRealizations).toBe(5);
    expect(result[0].totalWeight).toBe(6);

    // For polygon "all": realization k area-mean = k*0.1 + 0.025
    // -> values 0.025, 0.125, 0.225, 0.325, 0.425; mean = 0.225, var = 0.025
    // Tolerances reflect the crate-default f32 precision of the WASM aggregator
    // (about 7 decimal digits for O(1) magnitudes).
    expect(result[0].mean).toBeCloseTo(0.225, 6);
    expect(result[0].variance).not.toBeNull();
    expect(result[0].variance as number).toBeCloseTo(0.025, 6);
    expect(result[0].quantiles[0].value).toBeCloseTo(0.025, 6);
    expect(result[0].quantiles[1].value).toBeCloseTo(0.225, 6);
    expect(result[0].quantiles[2].value).toBeCloseTo(0.425, 6);

    // For polygon "right-half": realization k area-mean = k*0.1 + 0.04
    // -> mean = 0.24, var = 0.025
    expect(result[1].mean).toBeCloseTo(0.24, 6);
    expect((result[1].variance as number)).toBeCloseTo(0.025, 6);
  });

  test("aggregatePrevalenceByPolygon: weighted mean honors per-cell weights", () => {
    // 2 realizations × 2 targets, prevalence row 0 = [0, 1], row 1 = [0.4, 0.8]
    const ens = {
      nRealizations: 2,
      nTargets: 2,
      logitSamples: new Float64Array([-2, 2, -2, 2]),
      prevalenceSamples: new Float64Array([0, 1, 0.4, 0.8]),
    };
    // Weights (1, 3) -> 0.25 * x_0 + 0.75 * x_1
    const result = aggregatePrevalenceByPolygon({
      ensemble: ens,
      polygons: [{ indices: [0, 1], weights: [1, 3] }],
    });
    // Row 0 weighted mean: 0.25*0 + 0.75*1 = 0.75
    // Row 1 weighted mean: 0.25*0.4 + 0.75*0.8 = 0.7
    // Mean = 0.725 (f32-precision aggregator; see "simple area mean" test).
    expect(result[0].mean).toBeCloseTo(0.725, 6);
    expect(result[0].totalWeight).toBe(4);
  });

  test("aggregatePrevalenceByPolygon: summarizeOn='logit' summarizes the logit buffer", () => {
    const ens = buildEnsemble(
      4,
      3,
      (k) => 0.1 + k * 0.01,
      (k, j) => k - j
    );
    const result = aggregatePrevalenceByPolygon({
      ensemble: ens,
      polygons: [{ id: "p", indices: [0, 1, 2], weights: [1, 1, 1] }],
      summarizeOn: "logit",
      quantiles: [0.5],
    });
    // Per realization k, mean of (k, k-1, k-2) = k - 1.
    // Across k = 0..3 -> values [-1, 0, 1, 2], mean = 0.5 (f32 aggregator).
    expect(result[0].summarizeOn).toBe("logit");
    expect(result[0].mean).toBeCloseTo(0.5, 6);
    expect(result[0].quantiles[0].value).toBeCloseTo(0.5, 6);
  });

  test("aggregatePrevalenceByPolygon integrates with simulateBinomialGridEnsemble", () => {
    const lats = [0, 0, 1, 1];
    const lons = [0, 1, 0, 1];
    const successes = [3, 7, 4, 9];
    const trials = [10, 12, 9, 15];
    const variogram = {
      variogramType: "exponential" as VariogramTypeName,
      nugget: 0.05,
      sill: 1.0,
      range: 500,
    };
    const ens = simulateBinomialGridEnsemble({
      west: 0,
      south: 0,
      east: 1,
      north: 1,
      xCells: 4,
      yCells: 3,
      lats,
      lons,
      successes,
      trials,
      variogram,
      nRealizations: 6,
      baseSeed: 21,
    });
    // Polygon = upper-left 2×2 quadrant
    const upperLeft = polygonCellsFromMask({
      xCells: ens.xCells,
      yCells: ens.yCells,
      mask: [
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
      ],
      id: "upper-left",
    });
    const result = aggregatePrevalenceByPolygon({
      ensemble: ens,
      polygons: [upperLeft],
      quantiles: [0.025, 0.5, 0.975],
    });
    expect(result[0].id).toBe("upper-left");
    expect(result[0].nRealizations).toBe(6);
    expect(result[0].totalWeight).toBe(4);
    expect(result[0].mean).toBeGreaterThan(0);
    expect(result[0].mean).toBeLessThan(1);
    const [lo, med, hi] = result[0].quantiles.map((q) => q.value);
    expect(lo).toBeLessThanOrEqual(med);
    expect(med).toBeLessThanOrEqual(hi);
    expect(lo).toBeGreaterThanOrEqual(0);
    expect(hi).toBeLessThanOrEqual(1);
  });

  test("aggregatePrevalenceByPolygon: single realization yields null variance", () => {
    const ens = {
      nRealizations: 1,
      nTargets: 3,
      logitSamples: new Float64Array([0, 0, 0]),
      prevalenceSamples: new Float64Array([0.2, 0.3, 0.4]),
    };
    const result = aggregatePrevalenceByPolygon({
      ensemble: ens,
      polygons: [{ indices: [0, 1, 2], weights: [1, 1, 1] }],
      quantiles: [0.5],
    });
    expect(result[0].variance).toBeNull();
    expect(result[0].mean).toBeCloseTo(0.3, 6);
  });

  test("aggregatePrevalenceByPolygon validates inputs", () => {
    const ens = {
      nRealizations: 2,
      nTargets: 3,
      logitSamples: new Float64Array(6),
      prevalenceSamples: new Float64Array(6),
    };
    expect(() =>
      aggregatePrevalenceByPolygon({ ensemble: ens, polygons: [] })
    ).toThrow(KrigingError);
    expect(() =>
      aggregatePrevalenceByPolygon({
        ensemble: ens,
        polygons: [{ indices: [], weights: [] }],
      })
    ).toThrow(KrigingError);
    expect(() =>
      aggregatePrevalenceByPolygon({
        ensemble: ens,
        polygons: [{ indices: [0, 1], weights: [1] }],
      })
    ).toThrow(KrigingError);
    expect(() =>
      aggregatePrevalenceByPolygon({
        ensemble: ens,
        polygons: [{ indices: [3], weights: [1] }],
      })
    ).toThrow(KrigingError);
    expect(() =>
      aggregatePrevalenceByPolygon({
        ensemble: ens,
        polygons: [{ indices: [0], weights: [-1] }],
      })
    ).toThrow(KrigingError);
    expect(() =>
      aggregatePrevalenceByPolygon({
        ensemble: ens,
        polygons: [{ indices: [0, 1], weights: [0, 0] }],
      })
    ).toThrow(KrigingError);
    expect(() =>
      aggregatePrevalenceByPolygon({
        ensemble: ens,
        polygons: [{ indices: [0], weights: [1] }],
        quantiles: [-0.1],
      })
    ).toThrow(KrigingError);
  });

  test("polygonCellsFromMask: indicator and weighted masks both work", () => {
    const indicator = polygonCellsFromMask({
      xCells: 3,
      yCells: 2,
      mask: [
        [1, 0, 1],
        [0, 1, 0],
      ],
    });
    expect(Array.from(indicator.indices)).toEqual([0, 2, 4]);
    expect(Array.from(indicator.weights)).toEqual([1, 1, 1]);

    const weighted = polygonCellsFromMask({
      id: "pop",
      xCells: 3,
      yCells: 2,
      mask: [
        [10, 0, 5],
        [0, 20, 0],
      ],
    });
    expect(weighted.id).toBe("pop");
    expect(Array.from(weighted.indices)).toEqual([0, 2, 4]);
    expect(Array.from(weighted.weights)).toEqual([10, 5, 20]);

    // Booleans + nullish entries
    const boolMask = polygonCellsFromMask({
      xCells: 2,
      yCells: 2,
      mask: [
        [true, false],
        [null, true],
      ],
    });
    expect(Array.from(boolMask.indices)).toEqual([0, 3]);
    expect(Array.from(boolMask.weights)).toEqual([1, 1]);
  });

  test("polygonCellsFromMask validates shape and emptiness", () => {
    expect(() =>
      polygonCellsFromMask({
        xCells: 2,
        yCells: 2,
        mask: [[1, 1]],
      })
    ).toThrow(KrigingError);
    expect(() =>
      polygonCellsFromMask({
        xCells: 2,
        yCells: 2,
        mask: [
          [1, 1, 1],
          [1, 1, 1],
        ],
      })
    ).toThrow(KrigingError);
    expect(() =>
      polygonCellsFromMask({
        xCells: 2,
        yCells: 2,
        mask: [
          [0, 0],
          [0, 0],
        ],
      })
    ).toThrow(KrigingError);
  });
});

// ---------------------------------------------------------------------------
// 3-D kriging, variography, and SGS
// ---------------------------------------------------------------------------

/** 8 corners of a 10-unit cube plus its center, with a linear field. */
function cubeSamples3D() {
  const xs = [0, 10, 0, 10, 0, 10, 0, 10, 5];
  const ys = [0, 0, 10, 10, 0, 0, 10, 10, 5];
  const zs = [0, 0, 0, 0, 10, 10, 10, 10, 5];
  const values = xs.map((x, i) => 1 + 2 * x + 3 * ys[i] + 4 * zs[i]);
  return { xs, ys, zs, values };
}

/** Regular n^3 grid with a smooth correlated field plus deterministic jitter. */
function gridSamples3D(n: number) {
  const xs: number[] = [];
  const ys: number[] = [];
  const zs: number[] = [];
  const values: number[] = [];
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = i * 2;
        const y = j * 2;
        const z = k * 2;
        xs.push(x);
        ys.push(y);
        zs.push(z);
        values.push(
          Math.sin(x / 5) + Math.cos(y / 5) + 0.5 * Math.sin(z / 3) +
            0.1 * Math.sin(7.3 * i + 11.9 * j + 17.7 * k)
        );
      }
    }
  }
  return { xs, ys, zs, values };
}

const VARIOGRAM_3D = {
  variogramType: "exponential" as VariogramTypeName,
  nugget: 0.01,
  sill: 100.0,
  range: 15.0,
};

describe("3-D ordinary kriging", () => {
  test("predict returns finite value, variance, and diagnostics", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    const model = new OrdinaryKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
    });
    try {
      const pred: Prediction3D = model.predict(5, 5, 5);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(Number.isFinite(pred.variance)).toBe(true);
      expect(Number.isFinite(pred.conditionNumber)).toBe(true);
      expect(typeof pred.usedNuggetInflation).toBe("boolean");
    } finally {
      model.free();
    }
  });

  test("confluent-hypergeometric variogram works and shape2 (alpha) is threaded", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    // Two CH models differing only in the tail-decay alpha (shape2).
    // If shape2 reaches the engine, the predictions must differ; this
    // is what the VariogramSpec boundary buys the 3-D surface (a bare
    // scalar API could only carry the first shape, nu).
    const lowAlpha = new OrdinaryKriging3D({
      xs, ys, zs, values,
      variogram: {
        variogramType: "confluenthypergeometric",
        nugget: 0.01, sill: 100, range: 15, shape: 1.0, shape2: 0.5,
      },
    });
    const highAlpha = new OrdinaryKriging3D({
      xs, ys, zs, values,
      variogram: {
        variogramType: "confluenthypergeometric",
        nugget: 0.01, sill: 100, range: 15, shape: 1.0, shape2: 5.0,
      },
    });
    try {
      // Off the symmetry axes so the tail shape is detectable.
      const a = lowAlpha.predict(2, 4, 1.5);
      const b = highAlpha.predict(2, 4, 1.5);
      expect(Number.isFinite(a.value)).toBe(true);
      expect(Number.isFinite(b.value)).toBe(true);
      expect(Math.abs(a.value - b.value)).toBeGreaterThan(1e-6);
    } finally {
      lowAlpha.free();
      highAlpha.free();
    }
  });

  test("predictBatch returns parallel typed arrays", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    const model = new OrdinaryKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
    });
    try {
      const out = model.predictBatch([5, 2, 8], [5, 3, 7], [5, 4, 6]);
      expect(out.values).toBeInstanceOf(Float64Array);
      expect(out.values.length).toBe(3);
      expect(out.variances.length).toBe(3);
      expect(out.conditionNumbers.length).toBe(3);
      expect(out.usedNuggetInflation.length).toBe(3);
      for (const v of out.values) expect(Number.isFinite(v)).toBe(true);
    } finally {
      model.free();
    }
  });

  test("anisotropy changes the prediction", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    const iso = new OrdinaryKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
    });
    const aniso = new OrdinaryKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
      anisotropy: { ang1: 0, ang2: 0, ang3: 0, anis1: 1, anis2: 0.1 },
    });
    try {
      // Off the symmetry axes so the stretch is detectable.
      const a = iso.predict(2, 4, 1.5);
      const b = aniso.predict(2, 4, 1.5);
      expect(Math.abs(a.value - b.value)).toBeGreaterThan(1e-3);
    } finally {
      iso.free();
      aniso.free();
    }
  });

  test("neighborhood restriction still predicts finite values", () => {
    const { xs, ys, zs, values } = gridSamples3D(4);
    const model = new OrdinaryKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: { ...VARIOGRAM_3D, sill: 1.0 },
      neighborhood: { maxNeighbors: 8 },
    });
    try {
      const pred = model.predict(3, 3, 3);
      expect(Number.isFinite(pred.value)).toBe(true);
    } finally {
      model.free();
    }
  });

  test("mismatched array lengths throw KrigingError with stable code", () => {
    const { xs, ys, zs } = cubeSamples3D();
    expect(() => {
      new OrdinaryKriging3D({
        xs,
        ys,
        zs,
        values: [1, 2],
        variogram: VARIOGRAM_3D,
      });
    }).toThrow(KrigingError);
    try {
      new OrdinaryKriging3D({
        xs,
        ys,
        zs,
        values: [1, 2],
        variogram: VARIOGRAM_3D,
      });
    } catch (e) {
      expect((e as KrigingError).code).toBe("mismatched_arrays");
    }
  });

  test("free() makes subsequent use throw model_freed", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    const model = new OrdinaryKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
    });
    model.free();
    model.free(); // second free is a no-op
    try {
      model.predict(5, 5, 5);
      expect.unreachable("predict after free must throw");
    } catch (e) {
      expect(e).toBeInstanceOf(KrigingError);
      expect((e as KrigingError).code).toBe("model_freed");
    }
  });
});

describe("3-D simple kriging", () => {
  test("predict with known mean returns finite values", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const model = new SimpleKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
      mean,
    });
    try {
      const pred = model.predict(5, 5, 5);
      expect(Number.isFinite(pred.value)).toBe(true);
      expect(Number.isFinite(pred.variance)).toBe(true);
      const batch = model.predictBatch([5], [5], [5]);
      expect(batch.values.length).toBe(1);
    } finally {
      model.free();
    }
  });
});

describe("3-D universal kriging", () => {
  test("recovers a linear trend at an interior point", () => {
    const { xs, ys, zs, values } = cubeSamples3D();
    const model = new UniversalKriging3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
    });
    try {
      const pred = model.predict(5, 5, 5);
      const expected = 1 + 2 * 5 + 3 * 5 + 4 * 5;
      expect(Math.abs(pred.value - expected)).toBeLessThan(1.0);
    } finally {
      model.free();
    }
  });
});

describe("3-D directional variogram and spherical fits", () => {
  const { xs, ys, zs, values } = gridSamples3D(5);

  test("directional variogram returns aligned non-empty bins", () => {
    const result = computeDirectionalVariogram3D({
      xs,
      ys,
      zs,
      values,
      lagDistance: 2,
      nLags: 5,
      azimuthDeg: 0,
      dipDeg: 0,
    });
    expect(result.distances.length).toBeGreaterThan(0);
    expect(result.semivariances.length).toBe(result.distances.length);
    expect(result.nPairs.length).toBe(result.distances.length);
    expect(result.nPairs).toBeInstanceOf(Uint32Array);
    for (const g of result.semivariances) {
      expect(Number.isFinite(g)).toBe(true);
      expect(g).toBeGreaterThanOrEqual(0);
    }
  });

  test("omnidirectional collects more pairs than a narrow cone", () => {
    const narrow = computeDirectionalVariogram3D({
      xs,
      ys,
      zs,
      values,
      lagDistance: 2,
      nLags: 5,
      azimuthDeg: 0,
      dipDeg: 0,
    });
    const omni = computeDirectionalVariogram3D({
      xs,
      ys,
      zs,
      values,
      lagDistance: 2,
      nLags: 5,
      azimuthDeg: 0,
      azimuthToleranceDeg: 90,
      dipDeg: 0,
      dipToleranceDeg: 90,
    });
    const total = (r: { nPairs: Uint32Array }) =>
      [...r.nPairs].reduce((a, b) => a + b, 0);
    expect(total(omni)).toBeGreaterThan(total(narrow));
  });

  test("spherical fits return positive parameters on all entry points", () => {
    const axis = computeDirectionalVariogram3D({
      xs,
      ys,
      zs,
      values,
      lagDistance: 2,
      nLags: 5,
      azimuthDeg: 0,
      azimuthToleranceDeg: 90,
      dipDeg: 0,
      dipToleranceDeg: 90,
    });
    const input = { major: axis, minor: axis, vertical: axis };

    const joint = fitSpherical3DJoint(input);
    expect(joint.sill).toBeGreaterThan(0);
    expect(joint.rangeMajor).toBeGreaterThan(0);
    expect(joint.rangeMinor).toBeGreaterThan(0);
    expect(joint.rangeVertical).toBeGreaterThan(0);
    expect(Number.isFinite(joint.residuals)).toBe(true);

    const variance =
      values.reduce((a, b) => a + (b - 0) * (b - 0), 0) / values.length;
    const twoStage = fitSpherical3DTwoStage({
      ...input,
      dataVariance: variance,
    });
    expect(twoStage.sill).toBeGreaterThan(0);

    const fixed = fitSpherical3DFixedNugget({ ...input, nugget: 0.05 });
    // The engine stores parameters as f32, so the held nugget round-trips
    // with single precision.
    expect(fixed.nugget).toBeCloseTo(0.05, 6);

    const oneD = fitSpherical1D(axis);
    expect(oneD.sill).toBeGreaterThan(0);
    expect(oneD.range).toBeGreaterThan(0);
  });
});

describe("3-D sequential Gaussian simulation", () => {
  const { xs, ys, zs, values } = cubeSamples3D();
  const GRID = {
    nx: 4,
    ny: 4,
    nz: 2,
    originX: 0,
    originY: 0,
    originZ: 0,
    spacingX: 2.5,
    spacingY: 2.5,
    spacingZ: 5,
  };

  test("streams realizations in order as fresh Float64Arrays", () => {
    const seen: number[] = [];
    const grids: Float64Array[] = [];
    const result = gaussianSimulation3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
      grid: GRID,
      seed: 42,
      nRealizations: 3,
      onRealization: (idx, grid) => {
        seen.push(idx);
        grids.push(grid);
      },
    });
    expect(result.aborted).toBe(false);
    expect(seen).toEqual([0, 1, 2]);
    for (const grid of grids) {
      expect(grid).toBeInstanceOf(Float64Array);
      expect(grid.length).toBe(GRID.nx * GRID.ny * GRID.nz);
    }
    // Retained arrays must be distinct buffers, not a reused one.
    expect(grids[0]).not.toBe(grids[1]);
  });

  test("same seed reproduces the same realization", () => {
    const run = (seed: number): Float64Array => {
      let first: Float64Array | undefined;
      gaussianSimulation3D({
        xs,
        ys,
        zs,
        values,
        variogram: VARIOGRAM_3D,
        grid: GRID,
        seed,
        nRealizations: 1,
        onRealization: (_idx, grid) => {
          first = grid;
        },
      });
      return first!;
    };
    expect([...run(7)]).toEqual([...run(7)]);
  });

  test("returning true aborts the stream", () => {
    const seen: number[] = [];
    const result = gaussianSimulation3D({
      xs,
      ys,
      zs,
      values,
      variogram: VARIOGRAM_3D,
      grid: GRID,
      seed: 42,
      nRealizations: 5,
      onRealization: (idx) => {
        seen.push(idx);
        return idx === 1;
      },
    });
    expect(result.aborted).toBe(true);
    expect(seen).toEqual([0, 1]);
  });

  test("a callback exception propagates to the caller", () => {
    const boom = new Error("stop the presses");
    expect(() =>
      gaussianSimulation3D({
        xs,
        ys,
        zs,
        values,
        variogram: VARIOGRAM_3D,
        grid: GRID,
        seed: 42,
        nRealizations: 3,
        onRealization: () => {
          throw boom;
        },
      })
    ).toThrow(boom);
  });
});
