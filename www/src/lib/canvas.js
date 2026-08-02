import { variogramSemivariance, normalQuantile } from "./variogram";

export function valueToColor(value, min, max) {
  if (max <= min) return "hsl(0, 0%, 50%)";
  const t = Math.min(1, Math.max(0, (value - min) / (max - min)));
  const hue = (1 - t) * 240;
  return `hsl(${hue}, 85%, 50%)`;
}

export function drawSurfacePlaceholder(ctx, width, height) {
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#555";
  ctx.font = "16px sans-serif";
  ctx.fillText("Run 2D surface to render heatmap", 20, 30);
}

export function drawResidualPlaceholder(ctx, width, height) {
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#555";
  ctx.font = "14px sans-serif";
  ctx.fillText("Residual plot appears after running the 2D surface", 16, 28);
}

export function drawVariogramPlaceholder(ctx, width, height) {
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f7f7";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#555";
  ctx.font = "14px sans-serif";
  ctx.fillText("Empirical variogram appears after running the 2D surface", 16, 28);
}

export function renderResidualPlot(ctx, residuals, mode = "scatter", width, height) {
  if (!ctx || !residuals?.length) return { mean: 0, rmse: 0 };
  const padding = 22;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const minResidual = Math.min(...residuals);
  const maxResidual = Math.max(...residuals);
  const maxAbs = Math.max(Math.abs(minResidual), Math.abs(maxResidual), 1e-9);
  const mean = residuals.reduce((s, v) => s + v, 0) / residuals.length;
  const rmse = Math.sqrt(residuals.reduce((s, v) => s + v * v, 0) / residuals.length);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fdfdfd";
  ctx.fillRect(0, 0, width, height);
  const zeroY = padding + (plotHeight * maxAbs) / (2 * maxAbs);
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, zeroY);
  ctx.lineTo(width - padding, zeroY);
  ctx.stroke();
  ctx.strokeStyle = "#bbb";
  ctx.strokeRect(padding, padding, plotWidth, plotHeight);

  if (mode === "qq") {
    const sd = Math.sqrt(
      residuals.reduce((s, v) => s + (v - mean) ** 2, 0) / residuals.length,
    );
    const scale = Math.max(sd, 1e-9);
    const sorted = [...residuals].sort((a, b) => a - b).map((v) => (v - mean) / scale);
    const expected = sorted.map((_, i) => normalQuantile((i + 0.5) / sorted.length));
    const extent = Math.max(
      2.0,
      ...sorted.map((v) => Math.abs(v)),
      ...expected.map((v) => Math.abs(v)),
    );
    const toX = (v) => padding + ((v + extent) / (2 * extent)) * plotWidth;
    const toY = (v) => padding + ((extent - v) / (2 * extent)) * plotHeight;
    ctx.strokeStyle = "#7a7a7a";
    ctx.beginPath();
    ctx.moveTo(toX(-extent), toY(-extent));
    ctx.lineTo(toX(extent), toY(extent));
    ctx.stroke();
    for (let i = 0; i < sorted.length; i += 1) {
      ctx.fillStyle = "#5c2d91";
      ctx.fillRect(toX(expected[i]) - 1.5, toY(sorted[i]) - 1.5, 3, 3);
    }
  } else if (mode === "histogram") {
    const bins = 20;
    const span = Math.max(1e-9, maxResidual - minResidual);
    const counts = Array(bins).fill(0);
    for (const r of residuals) {
      const bin = Math.min(bins - 1, Math.max(0, Math.floor(((r - minResidual) / span) * bins)));
      counts[bin] += 1;
    }
    const maxCount = Math.max(...counts, 1);
    const barWidth = plotWidth / bins;
    for (let i = 0; i < bins; i += 1) {
      const barHeight = (counts[i] / maxCount) * (plotHeight - 8);
      ctx.fillStyle = "#4f81bd";
      ctx.fillRect(
        padding + i * barWidth + 1,
        padding + plotHeight - barHeight,
        Math.max(1, barWidth - 2),
        barHeight,
      );
    }
  } else {
    for (let i = 0; i < residuals.length; i += 1) {
      const x = padding + (i / Math.max(1, residuals.length - 1)) * plotWidth;
      const y = padding + ((maxAbs - residuals[i]) / (2 * maxAbs)) * plotHeight;
      ctx.fillStyle = residuals[i] >= 0 ? "#ba1b1b" : "#1f5aa6";
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }
  }
  ctx.fillStyle = "#444";
  ctx.font = "12px sans-serif";
  if (mode === "qq") ctx.fillText("QQ residual diagnostic (standardized)", 8, padding + 10);
  else if (mode === "histogram") ctx.fillText("Residual distribution", 8, padding + 10);
  else {
    ctx.fillText("+ residual", 8, padding + 10);
    ctx.fillText("- residual", 8, height - padding - 2);
  }
  return { mean, rmse };
}

export function renderVariogramPlot(
  ctx,
  variogramPoints,
  modelType,
  width,
  height,
  modelCurve = null,
) {
  if (!ctx) return { pointCount: 0 };
  const padding = 30;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fdfdfd";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#bbb";
  ctx.strokeRect(padding, padding, plotWidth, plotHeight);

  if (!variogramPoints?.length) {
    ctx.fillStyle = "#666";
    ctx.font = "13px sans-serif";
    ctx.fillText("Insufficient data for variogram", padding + 10, padding + 20);
    return { pointCount: 0 };
  }

  const maxDistance = Math.max(...variogramPoints.map((p) => p.distance), 1e-9);
  const maxSemivariance = Math.max(...variogramPoints.map((p) => p.semivariance), 1e-9);
  const minSemivariance = Math.min(...variogramPoints.map((p) => p.semivariance), 0);
  const overlayParams = {
    nugget: Math.max(0, minSemivariance),
    sill: maxSemivariance * 1.05,
    range: maxDistance * 0.45,
  };
  const modelMax = modelCurve?.length
    ? Math.max(...modelCurve.map((p) => p.semivariance), 0)
    : overlayParams.sill;
  const yMax = Math.max(maxSemivariance, modelMax, 1e-9);

  ctx.strokeStyle = "#2c7fb8";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < variogramPoints.length; i += 1) {
    const p = variogramPoints[i];
    const x = padding + (p.distance / maxDistance) * plotWidth;
    const y = padding + (1 - p.semivariance / yMax) * plotHeight;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  for (const p of variogramPoints) {
    const x = padding + (p.distance / maxDistance) * plotWidth;
    const y = padding + (1 - p.semivariance / yMax) * plotHeight;
    ctx.fillStyle = "#1f4e79";
    ctx.fillRect(x - 2, y - 2, 4, 4);
  }
  ctx.strokeStyle = "#d95f02";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  if (modelCurve?.length) {
    // Curve from the actual fitted model, evaluated by the Rust engine (supports every
    // family, including two-shape ones like confluent hypergeometric).
    for (let i = 0; i < modelCurve.length; i += 1) {
      const { distance, semivariance } = modelCurve[i];
      const x = padding + (distance / maxDistance) * plotWidth;
      const y = padding + (1 - semivariance / yMax) * plotHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  } else {
    // Fallback: illustrative curve from heuristic parameters using the JS model formulas.
    for (let i = 0; i <= 120; i += 1) {
      const distance = (i / 120) * maxDistance;
      const semivariance = variogramSemivariance(
        distance,
        modelType,
        overlayParams.nugget,
        overlayParams.sill,
        overlayParams.range,
      );
      const x = padding + (distance / maxDistance) * plotWidth;
      const y = padding + (1 - semivariance / yMax) * plotHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.fillStyle = "#444";
  ctx.font = "12px sans-serif";
  ctx.fillText("Semivariance", 8, padding - 8);
  ctx.fillText("Distance", width - 85, height - 8);
  ctx.fillStyle = "#1f4e79";
  ctx.fillRect(width - 220, 12, 10, 10);
  ctx.fillStyle = "#444";
  ctx.fillText("Empirical", width - 205, 21);
  ctx.fillStyle = "#d95f02";
  ctx.fillRect(width - 130, 12, 10, 10);
  ctx.fillStyle = "#444";
  ctx.fillText("Model fit", width - 115, 21);
  return { pointCount: variogramPoints.length };
}

export function renderSurface(
  ctx,
  predictions,
  resolution,
  bounds,
  samplePoints,
  observedValues,
  predictionSelector,
  layerMode,
  canvasWidth,
  canvasHeight,
) {
  if (!ctx) return null;
  const predictionValues = predictions.map(predictionSelector);
  const layerValues =
    layerMode === "variance"
      ? predictions.map((e) => e.variance)
      : predictionValues;
  const layerMin = Math.min(...layerValues);
  const layerMax = Math.max(...layerValues);
  const obsMin = Math.min(...observedValues);
  const obsMax = Math.max(...observedValues);
  const pointColorMin = Math.min(layerMin, obsMin);
  const pointColorMax = Math.max(layerMax, obsMax);
  const tileWidth = canvasWidth / resolution;
  const tileHeight = canvasHeight / resolution;

  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  for (let row = 0; row < resolution; row += 1) {
    for (let col = 0; col < resolution; col += 1) {
      const index = row * resolution + col;
      ctx.fillStyle = valueToColor(layerValues[index], layerMin, layerMax);
      ctx.fillRect(
        Math.floor(col * tileWidth),
        Math.floor(row * tileHeight),
        Math.ceil(tileWidth),
        Math.ceil(tileHeight),
      );
    }
  }
  const denseOverlay = samplePoints.lats.length > 120;
  for (let i = 0; i < samplePoints.lats.length; i += 1) {
    const lat = samplePoints.lats[i];
    const lon = samplePoints.lons[i];
    const x = ((lon - bounds.lonMin) / (bounds.lonMax - bounds.lonMin)) * canvasWidth;
    const y = ((bounds.latMax - lat) / (bounds.latMax - bounds.latMin)) * canvasHeight;
    ctx.beginPath();
    ctx.arc(x, y, denseOverlay ? 1.3 : 4, 0, Math.PI * 2);
    ctx.fillStyle = valueToColor(observedValues[i], pointColorMin, pointColorMax);
    ctx.fill();
    if (!denseOverlay) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, canvasWidth - 1, canvasHeight - 1);
  return { surfaceMin: layerMin, surfaceMax: layerMax };
}
