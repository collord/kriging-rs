import { useState, useRef, useEffect } from "react";
import {
  OrdinaryKriging,
  BinomialKriging,
  VariogramType,
  fitVariogram,
} from "kriging-rs-wasm";
import {
  generateSurfaceSamples,
  buildPredictionGrid,
  buildBinomialLogits,
  resolveBackendMode,
  summarizeTimings,
  summarizePhaseShares,
} from "../lib/sampleData";
import { computeEmpiricalVariogram } from "../lib/variogram";
import HelpTip, { FieldLabel } from "./HelpTip";
import {
  drawSurfacePlaceholder,
  drawResidualPlaceholder,
  drawVariogramPlaceholder,
  renderSurface,
  renderResidualPlot,
  renderVariogramPlot,
} from "../lib/canvas";

const CANVAS_W = 760;
const CANVAS_H_SURFACE = 440;
const CANVAS_H_RESIDUAL = 180;
const CANVAS_H_VARIOGRAM = 200;
const HARNESS_SAMPLE = generateSurfaceSamples(350);

function readOptions(state) {
  const nBins = Number(state.nBins);
  const maxDistance =
    state.maxDistanceKm.trim() === ""
      ? undefined
      : Number(state.maxDistanceKm);
  const variogramType = VariogramType[state.variogramModel];
  const variogramTypeName = state.variogramModel.toLowerCase();
  // Coerce to positive finite so WASM never receives 0/NaN (e.g. if user clears the input)
  let alpha = Number(state.binomialAlpha);
  let beta = Number(state.binomialBeta);
  if (!Number.isFinite(alpha) || alpha <= 0) alpha = 0.5;
  if (!Number.isFinite(beta) || beta <= 0) beta = 0.5;
  if (!Number.isFinite(nBins) || nBins <= 0) throw new Error("n_bins must be a positive integer");
  if (maxDistance !== undefined && (!Number.isFinite(maxDistance) || maxDistance <= 0)) {
    throw new Error("max distance must be positive when set");
  }
  return {
    nBins,
    maxDistance,
    variogramType,
    variogramTypeName,
    alpha,
    beta,
  };
}

async function runOrdinaryPerformanceHarness(options, backendSelection, webgpuAvailable) {
  const warmupRuns = 3;
  const measuredRuns = 10;
  const resolution = 36;
  const phaseSeries = {
    dataPrepMs: [],
    variogramFitMs: [],
    modelBuildMs: [],
    predictBatchMs: [],
    mappingMs: [],
    totalMs: [],
  };
  const backend = resolveBackendMode(backendSelection, webgpuAvailable);
  let checksum = 0;

  for (let runIndex = 0; runIndex < warmupRuns + measuredRuns; runIndex += 1) {
    const measured = runIndex >= warmupRuns;
    let t0 = performance.now();
    const sampleLats = Float64Array.from(HARNESS_SAMPLE.lats);
    const sampleLons = Float64Array.from(HARNESS_SAMPLE.lons);
    const sampleValues = Float64Array.from(HARNESS_SAMPLE.values);
    const grid = buildPredictionGrid(HARNESS_SAMPLE.lats, HARNESS_SAMPLE.lons, resolution);
    let t1 = performance.now();
    const dataPrepMs = t1 - t0;

    t0 = performance.now();
    const fitted = fitVariogram({
      sampleLats,
      sampleLons,
      values: sampleValues,
      variogramType: options.variogramType,
      maxDistance: options.maxDistance,
      nBins: options.nBins,
    });
    t1 = performance.now();
    const variogramFitMs = t1 - t0;

    t0 = performance.now();
    const model = OrdinaryKriging.fromFitted({
      lats: sampleLats,
      lons: sampleLons,
      values: sampleValues,
      fittedVariogram: fitted,
    });
    t1 = performance.now();
    const modelBuildMs = t1 - t0;

    t0 = performance.now();
    const predictions = backend.useGpu
      ? await model.predictBatchGpu(grid.predLats, grid.predLons)
      : model.predictBatch(grid.predLats, grid.predLons);
    t1 = performance.now();
    const predictBatchMs = t1 - t0;

    t0 = performance.now();
    const localChecksum = predictions.reduce((acc, p) => acc + p.value + p.variance, 0);
    checksum += localChecksum;
    t1 = performance.now();
    const mappingMs = t1 - t0;
    model.free();

    if (measured) {
      const totalMs =
        dataPrepMs + variogramFitMs + modelBuildMs + predictBatchMs + mappingMs;
      phaseSeries.dataPrepMs.push(dataPrepMs);
      phaseSeries.variogramFitMs.push(variogramFitMs);
      phaseSeries.modelBuildMs.push(modelBuildMs);
      phaseSeries.predictBatchMs.push(predictBatchMs);
      phaseSeries.mappingMs.push(mappingMs);
      phaseSeries.totalMs.push(totalMs);
    }
  }

  const totalStats = summarizeTimings(phaseSeries.totalMs);
  return {
    scenario: "ordinary_350_samples_36x36_grid",
    warmupRuns,
    measuredRuns,
    samplePoints: HARNESS_SAMPLE.lats.length,
    predictionPoints: resolution * resolution,
    options,
    backendSelection,
    backendResolved: backend.useGpu ? "webgpu" : "cpu",
    total: totalStats,
    phases: {
      dataPrep: summarizeTimings(phaseSeries.dataPrepMs),
      variogramFit: summarizeTimings(phaseSeries.variogramFitMs),
      modelBuild: summarizeTimings(phaseSeries.modelBuildMs),
      predictBatch: summarizeTimings(phaseSeries.predictBatchMs),
      mapping: summarizeTimings(phaseSeries.mappingMs),
    },
    phaseShare: summarizePhaseShares(
      {
        dataPrep: phaseSeries.dataPrepMs,
        variogramFit: phaseSeries.variogramFitMs,
        modelBuild: phaseSeries.modelBuildMs,
        predictBatch: phaseSeries.predictBatchMs,
        mapping: phaseSeries.mappingMs,
      },
      phaseSeries.totalMs,
    ),
    checksum,
  };
}

export default function SurfaceDemo({ uploadedData, onError, webgpuStatus }) {
  const [surfaceSample, setSurfaceSample] = useState(() => generateSurfaceSamples());
  const [resolution, setResolution] = useState(36);
  const [krigingType, setKrigingType] = useState("ordinary");
  const [layerMode, setLayerMode] = useState("prediction");
  const [residualMode, setResidualMode] = useState("scatter");
  const [backend, setBackend] = useState("auto");
  const [nBins, setNBins] = useState(12);
  const [maxDistanceKm, setMaxDistanceKm] = useState("");
  const [variogramModel, setVariogramModel] = useState("Exponential");
  const [binomialAlpha, setBinomialAlpha] = useState(0.5);
  const [binomialBeta, setBinomialBeta] = useState(0.5);
  const [running, setRunning] = useState(false);
  const [harnessRunning, setHarnessRunning] = useState(false);
  const [harnessReport, setHarnessReport] = useState(null);
  const [surfaceRange, setSurfaceRange] = useState({ min: "-", max: "-" });
  const [residualStats, setResidualStats] = useState({ mean: "-", rmse: "-" });
  const [lastRunResult, setLastRunResult] = useState(null);

  const surfaceRef = useRef(null);
  const residualRef = useRef(null);
  const variogramRef = useRef(null);

  const webgpuAvailable = webgpuStatus === "available";

  // Draw placeholders when no result
  useEffect(() => {
    if (lastRunResult) return;
    const sCtx = surfaceRef.current?.getContext("2d");
    const rCtx = residualRef.current?.getContext("2d");
    const vCtx = variogramRef.current?.getContext("2d");
    if (sCtx) drawSurfacePlaceholder(sCtx, CANVAS_W, CANVAS_H_SURFACE);
    if (rCtx) drawResidualPlaceholder(rCtx, CANVAS_W, CANVAS_H_RESIDUAL);
    if (vCtx) drawVariogramPlaceholder(vCtx, CANVAS_W, CANVAS_H_VARIOGRAM);
  }, [lastRunResult]);

  async function handleRunSurface() {
    const sample = uploadedData ?? generateSurfaceSamples();
    if (!uploadedData) setSurfaceSample(sample);
    if (!sample.lats?.length) {
      onError?.("No sample data");
      return;
    }
    setRunning(true);
    setLastRunResult(null);
    try {
      const options = readOptions({
        nBins,
        maxDistanceKm,
        variogramModel,
        binomialAlpha,
        binomialBeta,
      });
      const backendResolved = resolveBackendMode(backend, webgpuAvailable);
      const grid = buildPredictionGrid(sample.lats, sample.lons, resolution);
      let predictions;
      let samplePredictions;
      let observedValues;
      let selector;

      if (krigingType === "binomial") {
        const sampleLats = Float64Array.from(sample.lats);
        const sampleLons = Float64Array.from(sample.lons);
        const sampleSuccesses = Uint32Array.from(sample.successes);
        const sampleTrials = Uint32Array.from(sample.trials);
        const sampleLogits = buildBinomialLogits(
          sample.successes,
          sample.trials,
          options.alpha,
          options.beta,
        );
        const fitted = fitVariogram({
          sampleLats,
          sampleLons,
          values: sampleLogits,
          variogramType: options.variogramType,
          maxDistance: options.maxDistance,
          nBins: options.nBins,
        });
        const model = BinomialKriging.fromFittedVariogramWithPrior({
          lats: sampleLats,
          lons: sampleLons,
          successes: sampleSuccesses,
          trials: sampleTrials,
          fittedVariogram: fitted,
          prior: { alpha: options.alpha, beta: options.beta },
        });
        predictions = backendResolved.useGpu
          ? await model.predictBatchGpu(grid.predLats, grid.predLons)
          : model.predictBatch(grid.predLats, grid.predLons);
        samplePredictions = backendResolved.useGpu
          ? await model.predictBatchGpu(sampleLats, sampleLons)
          : model.predictBatch(sampleLats, sampleLons);
        model.free();
        selector = (e) => e.prevalence;
        observedValues = sample.successes.map((s, i) => s / sample.trials[i]);
      } else {
        const sampleLats = Float64Array.from(sample.lats);
        const sampleLons = Float64Array.from(sample.lons);
        const sampleValues = Float64Array.from(sample.values);
        const fitted = fitVariogram({
          sampleLats,
          sampleLons,
          values: sampleValues,
          variogramType: options.variogramType,
          maxDistance: options.maxDistance,
          nBins: options.nBins,
        });
        const model = OrdinaryKriging.fromFitted({
          lats: sampleLats,
          lons: sampleLons,
          values: sampleValues,
          fittedVariogram: fitted,
        });
        predictions = backendResolved.useGpu
          ? await model.predictBatchGpu(grid.predLats, grid.predLons)
          : model.predictBatch(grid.predLats, grid.predLons);
        samplePredictions = backendResolved.useGpu
          ? await model.predictBatchGpu(sampleLats, sampleLons)
          : model.predictBatch(sampleLats, sampleLons);
        model.free();
        selector = (e) => e.value;
        observedValues = sample.values;
      }

      const residuals = samplePredictions.map((entry, i) => observedValues[i] - selector(entry));

      const sCtx = surfaceRef.current?.getContext("2d");
      const rCtx = residualRef.current?.getContext("2d");
      const vCtx = variogramRef.current?.getContext("2d");

      if (sCtx) {
        const range = renderSurface(
          sCtx,
          predictions,
          resolution,
          grid.bounds,
          sample,
          observedValues,
          selector,
          layerMode,
          CANVAS_W,
          CANVAS_H_SURFACE,
        );
        setSurfaceRange(
          range ? { min: range.surfaceMin.toFixed(3), max: range.surfaceMax.toFixed(3) } : { min: "-", max: "-" },
        );
      }
      if (rCtx) {
        const stats = renderResidualPlot(
          rCtx,
          residuals,
          residualMode,
          CANVAS_W,
          CANVAS_H_RESIDUAL,
        );
        setResidualStats({
          mean: stats.mean.toFixed(4),
          rmse: stats.rmse.toFixed(4),
        });
      }
      const variogramPoints = computeEmpiricalVariogram(
        sample.lats,
        sample.lons,
        observedValues,
        options.nBins,
      );
      if (vCtx) {
        renderVariogramPlot(
          vCtx,
          variogramPoints,
          options.variogramTypeName,
          CANVAS_W,
          CANVAS_H_VARIOGRAM,
        );
      }

      setLastRunResult({
        predictions,
        grid,
        resolution,
        bounds: grid.bounds,
        sample,
        observedValues,
        selector,
        krigingType,
      });
    } catch (err) {
      onError?.(err?.message ?? String(err));
      setSurfaceRange({ min: "-", max: "-" });
      setResidualStats({ mean: "-", rmse: "-" });
      const sCtx = surfaceRef.current?.getContext("2d");
      const rCtx = residualRef.current?.getContext("2d");
      const vCtx = variogramRef.current?.getContext("2d");
      if (sCtx) drawSurfacePlaceholder(sCtx, CANVAS_W, CANVAS_H_SURFACE);
      if (rCtx) drawResidualPlaceholder(rCtx, CANVAS_W, CANVAS_H_RESIDUAL);
      if (vCtx) drawVariogramPlaceholder(vCtx, CANVAS_W, CANVAS_H_VARIOGRAM);
    } finally {
      setRunning(false);
    }
  }

  function handleRunHarness() {
    setHarnessRunning(true);
    setHarnessReport(null);
    const options = readOptions({
      nBins,
      maxDistanceKm,
      variogramModel,
      binomialAlpha,
      binomialBeta,
    });
    runOrdinaryPerformanceHarness(options, backend, webgpuAvailable)
      .then(setHarnessReport)
      .catch((err) => {
        onError?.(err?.message ?? String(err));
        setHarnessReport({ error: err?.message ?? String(err) });
      })
      .finally(() => setHarnessRunning(false));
  }

  function handleExportPng() {
    const canvas = surfaceRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "kriging-surface.png";
      a.click();
      URL.revokeObjectURL(a.href);
    }, "image/png");
  }

  function handleExportCsv() {
    if (!lastRunResult) return;
    const { predictions, grid, resolution, selector, krigingType } = lastRunResult;
    const header =
      krigingType === "binomial"
        ? "lat,lon,prevalence,variance"
        : "lat,lon,value,variance";
    const rows = [];
    for (let i = 0; i < predictions.length; i += 1) {
      const lat = grid.predLats[i];
      const lon = grid.predLons[i];
      const p = predictions[i];
      const val = krigingType === "binomial" ? p.prevalence : p.value;
      rows.push(`${lat},${lon},${val},${p.variance}`);
    }
    const csv = [header, ...rows].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    a.download = "kriging-grid.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div className="panel">
      <h2>
        2D Surface demo
        <HelpTip topic="surfaceSection" label="2D Surface demo" />
      </h2>
      <p>
        Predict an ordinary or binomial kriging surface from synthetic or uploaded data
        and render it as a heatmap. Optionally export the plot as PNG or the grid as CSV.
      </p>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.75rem",
          alignItems: "flex-end",
          marginBottom: "0.75rem",
        }}
      >
        <div className="control-group">
          <FieldLabel htmlFor="surfaceKrigingType" topic="krigingMode">
            Kriging mode
          </FieldLabel>
          <select
            id="surfaceKrigingType"
            value={krigingType}
            onChange={(e) => setKrigingType(e.target.value)}
          >
            <option value="ordinary">Ordinary (value)</option>
            <option value="binomial">Binomial (prevalence)</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="gridResolution" topic="gridResolution">
            Grid resolution
          </FieldLabel>
          <select
            id="gridResolution"
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
          >
            <option value={24}>24×24 (fast)</option>
            <option value={36}>36×36 (default)</option>
            <option value={48}>48×48 (detailed)</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="nBins" topic="empiricalBins">
            Empirical bins
          </FieldLabel>
          <select id="nBins" value={nBins} onChange={(e) => setNBins(Number(e.target.value))}>
            <option value={8}>8</option>
            <option value={12}>12</option>
            <option value={18}>18</option>
            <option value={24}>24</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="maxDistanceKm" topic="maxDistance">
            Max distance (optional)
          </FieldLabel>
          <input
            id="maxDistanceKm"
            type="number"
            min={0}
            step={0.1}
            placeholder="auto"
            value={maxDistanceKm}
            onChange={(e) => setMaxDistanceKm(e.target.value)}
          />
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="variogramModel" topic="variogramModel">
            Variogram model
          </FieldLabel>
          <select
            id="variogramModel"
            value={variogramModel}
            onChange={(e) => setVariogramModel(e.target.value)}
          >
            <option value="Spherical">Spherical</option>
            <option value="Exponential">Exponential</option>
            <option value="Gaussian">Gaussian</option>
            <option value="Cubic">Cubic</option>
            <option value="Stable">Stable</option>
            <option value="Matern">Matérn</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="surfaceLayer" topic="surfaceLayer">
            Surface layer
          </FieldLabel>
          <select
            id="surfaceLayer"
            value={layerMode}
            onChange={(e) => setLayerMode(e.target.value)}
          >
            <option value="prediction">Prediction</option>
            <option value="variance">Kriging variance</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="binomialAlpha" topic="binomialPrior">
            Binomial α
          </FieldLabel>
          <input
            id="binomialAlpha"
            type="number"
            min={0.0001}
            step={0.1}
            value={binomialAlpha}
            onChange={(e) => setBinomialAlpha(Number(e.target.value))}
          />
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="binomialBeta" topic="binomialPrior">
            Binomial β
          </FieldLabel>
          <input
            id="binomialBeta"
            type="number"
            min={0.0001}
            step={0.1}
            value={binomialBeta}
            onChange={(e) => setBinomialBeta(Number(e.target.value))}
          />
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="residualMode" topic="residualPlot">
            Residual plot
          </FieldLabel>
          <select
            id="residualMode"
            value={residualMode}
            onChange={(e) => setResidualMode(e.target.value)}
          >
            <option value="scatter">Scatter</option>
            <option value="histogram">Histogram</option>
            <option value="qq">QQ-style</option>
          </select>
        </div>
        <div className="control-group">
          <FieldLabel htmlFor="surfaceBackend" topic="backend">
            Backend
          </FieldLabel>
          <select
            id="surfaceBackend"
            value={backend}
            onChange={(e) => setBackend(e.target.value)}
          >
            <option value="auto">Auto (prefer WebGPU)</option>
            <option value="cpu">CPU only</option>
            <option value="webgpu">WebGPU required</option>
          </select>
        </div>
      </div>
      <div style={{ marginBottom: "0.5rem" }}>
        <button
          type="button"
          onClick={handleRunSurface}
          disabled={running}
        >
          {running ? "Running…" : "Run 2D surface"}
        </button>
        <button
          type="button"
          onClick={handleRunHarness}
          disabled={harnessRunning}
        >
          {harnessRunning ? "Running…" : "Run performance harness"}
        </button>
        <HelpTip topic="performanceHarness" label="Performance harness" />
        {lastRunResult && (
          <>
            <button type="button" onClick={handleExportPng}>
              Download PNG
            </button>
            <button type="button" onClick={handleExportCsv}>
              Download CSV
            </button>
          </>
        )}
      </div>
      <p style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
        {layerMode === "variance" ? "Variance" : "Value"} range: {surfaceRange.min} to{" "}
        {surfaceRange.max}
        <HelpTip topic="surfaceRange" label="Value or variance range" />
      </p>
      <canvas
        ref={surfaceRef}
        width={CANVAS_W}
        height={CANVAS_H_SURFACE}
        style={{ display: "block", maxWidth: "100%", border: "1px solid var(--border)", borderRadius: 6 }}
        aria-label="2D kriging heatmap"
      />
      <p style={{ fontSize: "0.9rem", marginTop: "0.5rem" }}>
        Residual mean: {residualStats.mean} | Residual RMSE: {residualStats.rmse}
        <HelpTip topic="residualStats" label="Residual statistics" />
      </p>
      <canvas
        ref={residualRef}
        width={CANVAS_W}
        height={CANVAS_H_RESIDUAL}
        style={{ display: "block", maxWidth: "100%", border: "1px solid var(--border)", borderRadius: 6, marginTop: "0.5rem" }}
        aria-label="Residual plot"
      />
      <canvas
        ref={variogramRef}
        width={CANVAS_W}
        height={CANVAS_H_VARIOGRAM}
        style={{ display: "block", maxWidth: "100%", border: "1px solid var(--border)", borderRadius: 6, marginTop: "0.5rem" }}
        aria-label="Empirical variogram"
      />
      <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "0.5rem" }}>
        Empirical variogram: lower semivariance at short distances indicates stronger
        local certainty.
        <HelpTip topic="empiricalVariogramPlot" label="Empirical variogram" />
      </p>
      {harnessReport && (
        <div style={{ marginTop: "1rem" }}>
          <strong>Performance harness</strong>
          {harnessReport.error ? (
            <pre>{harnessReport.error}</pre>
          ) : (
            <pre>{JSON.stringify(harnessReport, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
}
