import { useState, useRef } from "react";
import {
  OrdinaryKriging,
  VariogramType,
  fitVariogram,
} from "kriging-rs-wasm";
import { generateSurfaceSamples, buildPredictionGrid, resolveBackendMode } from "../lib/sampleData";
import { renderSurface } from "../lib/canvas";
import HelpTip from "./HelpTip";
import { VariogramMathHelp } from "../help/variogramMath.jsx";

const CANVAS_W = 360;
const CANVAS_H = 220;
const RESOLUTION = 32;

const VARIOGRAM_OPTIONS = [
  { value: "Spherical", label: "Spherical" },
  { value: "Exponential", label: "Exponential" },
  { value: "Gaussian", label: "Gaussian" },
  { value: "Cubic", label: "Cubic" },
  { value: "Stable", label: "Stable" },
  { value: "Matern", label: "Matérn" },
  { value: "ConfluentHypergeometric", label: "Confluent hypergeometric" },
];

export default function CompareView({ onError, webgpuAvailable = false }) {
  const [running, setRunning] = useState(false);
  const [modelLeft, setModelLeft] = useState("Exponential");
  const [modelRight, setModelRight] = useState("Spherical");
  const leftRef = useRef(null);
  const rightRef = useRef(null);

  async function runComparison() {
    const sample = generateSurfaceSamples(200);
    if (!sample.lats?.length || sample.lats.length < 4) {
      onError?.("Need at least 4 sample points");
      return;
    }
    setRunning(true);
    const backend = resolveBackendMode("auto", webgpuAvailable);
    const grid = buildPredictionGrid(sample.lats, sample.lons, RESOLUTION);
    const nBins = 12;
    const sampleLats = Float64Array.from(sample.lats);
    const sampleLons = Float64Array.from(sample.lons);
    const sampleValues = Float64Array.from(sample.values);

    try {
      const runOne = async (variogramModelName) => {
        const fitted = fitVariogram({
          sampleLats,
          sampleLons,
          values: sampleValues,
          variogramType: VariogramType[variogramModelName],
          nBins,
        });
        const model = OrdinaryKriging.fromFitted({
          lats: sampleLats,
          lons: sampleLons,
          values: sampleValues,
          fittedVariogram: fitted,
        });
        const predictions = backend.useGpu
          ? await model.predictBatchGpu(grid.predLats, grid.predLons)
          : model.predictBatch(grid.predLats, grid.predLons);
        model.free();
        return {
          predictions,
          selector: (e) => e.value,
          observed: sample.values,
        };
      };

      const [left, right] = await Promise.all([
        runOne(modelLeft),
        runOne(modelRight),
      ]);

      const leftCtx = leftRef.current?.getContext("2d");
      const rightCtx = rightRef.current?.getContext("2d");
      if (leftCtx) {
        renderSurface(
          leftCtx,
          left.predictions,
          RESOLUTION,
          grid.bounds,
          sample,
          left.observed,
          left.selector,
          "prediction",
          CANVAS_W,
          CANVAS_H,
        );
      }
      if (rightCtx) {
        renderSurface(
          rightCtx,
          right.predictions,
          RESOLUTION,
          grid.bounds,
          sample,
          right.observed,
          right.selector,
          "prediction",
          CANVAS_W,
          CANVAS_H,
        );
      }
    } catch (err) {
      onError?.(err?.message ?? String(err));
      const leftCtx = leftRef.current?.getContext("2d");
      const rightCtx = rightRef.current?.getContext("2d");
      if (leftCtx) {
        leftCtx.fillStyle = "#f7f7f7";
        leftCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);
        leftCtx.fillStyle = "#555";
        leftCtx.fillText("Error", 10, 20);
      }
      if (rightCtx) {
        rightCtx.fillStyle = "#f7f7f7";
        rightCtx.fillRect(0, 0, CANVAS_W, CANVAS_H);
        rightCtx.fillStyle = "#555";
        rightCtx.fillText("Error", 10, 20);
      }
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="panel">
      <h2>
        Compare variogram models
        <HelpTip topic="compareSection" label="Compare variogram models" />
      </h2>
      <p>
        Run ordinary kriging on the same synthetic dataset with two different variogram
        models and view the surfaces side-by-side.
      </p>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "1rem",
          alignItems: "flex-start",
          marginBottom: "0.75rem",
        }}
      >
        <div className="control-group">
          <span className="label-row">
            <label htmlFor="compareLeft">Left model</label>
            <HelpTip title={`Left model — ${modelLeft}`} label="Left model math">
              <VariogramMathHelp model={modelLeft} />
            </HelpTip>
          </span>
          <select
            id="compareLeft"
            value={modelLeft}
            onChange={(e) => setModelLeft(e.target.value)}
          >
            {VARIOGRAM_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div className="control-group">
          <span className="label-row">
            <label htmlFor="compareRight">Right model</label>
            <HelpTip title={`Right model — ${modelRight}`} label="Right model math">
              <VariogramMathHelp model={modelRight} />
            </HelpTip>
          </span>
          <select
            id="compareRight"
            value={modelRight}
            onChange={(e) => setModelRight(e.target.value)}
          >
            {VARIOGRAM_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div style={{ alignSelf: "flex-end" }}>
          <button type="button" onClick={runComparison} disabled={running}>
            {running ? "Running…" : "Run comparison"}
          </button>
        </div>
      </div>
      <div
        style={{
          display: "flex",
          gap: "1rem",
          flexWrap: "wrap",
          justifyContent: "flex-start",
        }}
      >
        <div>
          <p style={{ fontSize: "0.85rem", marginBottom: "0.25rem" }}>{modelLeft}</p>
          <canvas
            ref={leftRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={{
              border: "1px solid var(--border)",
              borderRadius: 6,
              background: "#f7f7f7",
            }}
            aria-label={`Surface ${modelLeft}`}
          />
        </div>
        <div>
          <p style={{ fontSize: "0.85rem", marginBottom: "0.25rem" }}>{modelRight}</p>
          <canvas
            ref={rightRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={{
              border: "1px solid var(--border)",
              borderRadius: 6,
              background: "#f7f7f7",
            }}
            aria-label={`Surface ${modelRight}`}
          />
        </div>
      </div>
    </div>
  );
}
