// UI thread: owns the DOM and talks to the kriging worker. It never imports the WASM
// package itself (only its types), so nothing here can block on a kriging call.
import type { VariogramTypeName } from "kriging-rs-wasm";
import { makeSynthetic, parseCsv, type GridResult, type Samples, type VariogramResult } from "./core";
import type { Request, Response } from "./worker";

// ---------- worker client ----------
type Body<T> = T extends unknown ? Omit<T, "id"> : never;
const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let nextId = 1;
const pending = new Map<number, { resolve: (r: Response) => void; reject: (e: Error) => void }>();
let resolveReady!: () => void;
const workerReady = new Promise<void>((r) => (resolveReady = r));
worker.onmessage = (ev: MessageEvent<Response>) => {
  const msg = ev.data;
  if (msg.ok && msg.type === "ready") return resolveReady();
  const p = pending.get(msg.id);
  if (!p) return;
  pending.delete(msg.id);
  if (msg.ok) p.resolve(msg);
  else p.reject(new Error(msg.code ? `${msg.error} [${msg.code}]` : msg.error));
};
worker.onerror = (e) => setStatus(`Worker error: ${e.message}`, "err");
function call(req: Body<Request>, transfer: Transferable[] = []): Promise<Response> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ ...req, id }, transfer);
  });
}

// ---------- DOM ----------
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $("status");
const familyEl = $<HTMLSelectElement>("family");
const nBinsEl = $<HTMLInputElement>("nBins");
const resEl = $<HTMLSelectElement>("res");
const nbrsEl = $<HTMLInputElement>("nbrs");
const showPtsEl = $<HTMLInputElement>("showPts");
const mapCanvas = $<HTMLCanvasElement>("map");
const legendCanvas = $<HTMLCanvasElement>("legend");
const vgSvg = document.getElementById("vgPlot") as unknown as SVGSVGElement;

function setStatus(text: string, cls: "" | "busy" | "err" = "") {
  statusEl.textContent = text;
  statusEl.className = cls;
}

// ---------- state ----------
let data: Samples | null = null;
let vg: VariogramResult | null = null;
let grid: GridResult | null = null;
let layer: "estimate" | "variance" = "estimate";
let generation = 0; // bump to invalidate in-flight results

// ---------- pipeline ----------
async function loadSamples(s: Samples, label: string) {
  data = s;
  const g = ++generation;
  $("dataInfo").textContent = `${label}: ${s.xs.length} samples`;
  // Send copies: the main thread keeps its arrays for the point overlay.
  await call({ type: "load", xs: s.xs.slice(), ys: s.ys.slice(), values: s.values.slice() });
  if (g === generation) await runVariogram();
}

async function runVariogram() {
  if (!data) return;
  const g = ++generation;
  setStatus("Fitting variogram…", "busy");
  try {
    const r = await call({
      type: "variogram",
      family: familyEl.value as VariogramTypeName,
      nBins: Math.max(4, Math.min(40, Number(nBinsEl.value) || 15)),
    });
    if (g !== generation || !r.ok || r.type !== "variogram") return;
    vg = r.result;
    drawVariogram(vg);
    const m = vg.model;
    $("vgInfo").textContent =
      `nugget ${fmt(m.nugget)}  sill ${fmt(m.sill)}\nrange ${fmt(m.range)} m` +
      (m.shape !== undefined ? `  shape ${fmt(m.shape)}` : "") +
      `\nfit ${r.ms.toFixed(0)} ms` +
      (vg.nUsed < data.xs.length ? ` (${vg.nUsed}-pt subsample)` : "");
    await runKrige(g);
  } catch (e) {
    if (g === generation) setStatus(`Variogram failed: ${(e as Error).message}`, "err");
  }
}

async function runKrige(g = ++generation) {
  if (!data || !vg) return;
  const n = Number(resEl.value);
  setStatus(`Kriging ${n}×${n}…`, "busy");
  const t0 = performance.now();
  try {
    const maxNeighbors = Number(nbrsEl.value) || 32;
    const r = await call({ type: "krige", model: vg.model, nx: n, ny: n, maxNeighbors });
    if (g !== generation || !r.ok || r.type !== "krige") return;
    grid = r.result;
    drawMap();
    $("gridInfo").textContent =
      `kriging ${grid.ms.toFixed(0)} ms in worker, ` +
      `round-trip ${(performance.now() - t0).toFixed(0)} ms`;
    setStatus("Ready");
  } catch (e) {
    if (g === generation) setStatus(`Kriging failed: ${(e as Error).message}`, "err");
  }
}

// ---------- helpers ----------
function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-2) return v.toExponential(2);
  return v.toPrecision(a >= 100 ? 5 : 3);
}
function niceTicks(max: number, count = 5): number[] {
  const step0 = max / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(t);
  return out;
}

// Viridis (matplotlib) sampled at 9 stops.
const VIRIDIS = [
  [68, 1, 84], [71, 44, 122], [59, 81, 139], [44, 113, 142], [33, 144, 141],
  [39, 173, 129], [92, 200, 99], [170, 220, 50], [253, 231, 37],
];
function colour(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---------- variogram plot (SVG) ----------
function drawVariogram(v: VariogramResult) {
  const W = 460, H = 300, L = 56, R = 14, T = 14, B = 42;
  const xMax = v.maxDistance;
  const yMax = Math.max(...v.empirical.semivariances, ...v.curve.semivariances) * 1.08 || 1;
  const sx = (d: number) => L + (d / xMax) * (W - L - R);
  const sy = (g: number) => H - B - (g / yMax) * (H - T - B);
  const maxCount = Math.max(...v.empirical.counts);
  const parts: string[] = [];
  for (const t of niceTicks(xMax)) {
    parts.push(`<line x1="${sx(t)}" x2="${sx(t)}" y1="${T}" y2="${H - B}" stroke="currentColor" stroke-opacity="0.1"/>`,
      `<text x="${sx(t)}" y="${H - B + 16}" font-size="11" text-anchor="middle" fill="currentColor">${fmt(t)}</text>`);
  }
  for (const t of niceTicks(yMax)) {
    parts.push(`<line x1="${L}" x2="${W - R}" y1="${sy(t)}" y2="${sy(t)}" stroke="currentColor" stroke-opacity="0.1"/>`,
      `<text x="${L - 6}" y="${sy(t) + 4}" font-size="11" text-anchor="end" fill="currentColor">${fmt(t)}</text>`);
  }
  parts.push(`<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="currentColor"/>`,
    `<line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" stroke="currentColor"/>`,
    `<text x="${(L + W - R) / 2}" y="${H - 6}" font-size="12" text-anchor="middle" fill="currentColor">lag distance h (m)</text>`,
    `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="currentColor">γ(h)</text>`);
  // Sill reference line.
  parts.push(`<line x1="${L}" x2="${W - R}" y1="${sy(v.model.sill)}" y2="${sy(v.model.sill)}" stroke="#e0752d" stroke-dasharray="4 4" stroke-opacity="0.7"/>`);
  const path = v.curve.distances.map((d, i) => `${i ? "L" : "M"}${sx(d).toFixed(1)},${sy(v.curve.semivariances[i]).toFixed(1)}`).join("");
  parts.push(`<path d="${path}" fill="none" stroke="#e0752d" stroke-width="2.2"/>`);
  v.empirical.distances.forEach((d, i) => {
    const r = 2.5 + 4 * Math.sqrt(v.empirical.counts[i] / maxCount);
    parts.push(`<circle cx="${sx(d)}" cy="${sy(v.empirical.semivariances[i])}" r="${r.toFixed(1)}" fill="#2a6fdb" fill-opacity="0.85"><title>h=${fmt(d)} m, γ=${fmt(v.empirical.semivariances[i])}, pairs=${v.empirical.counts[i]}</title></circle>`);
  });
  parts.push(`<g font-size="11" fill="currentColor"><circle cx="${W - R - 120}" cy="${T + 10}" r="4" fill="#2a6fdb"/><text x="${W - R - 112}" y="${T + 14}">empirical (size ∝ pairs)</text>`,
    `<line x1="${W - R - 126}" x2="${W - R - 114}" y1="${T + 28}" y2="${T + 28}" stroke="#e0752d" stroke-width="2.2"/><text x="${W - R - 112}" y="${T + 32}">${v.model.variogramType} model</text></g>`);
  vgSvg.innerHTML = parts.join("");
}

// ---------- map ----------
const off = document.createElement("canvas");
let mapRect = { x: 0, y: 0, w: 0, h: 0 };

function drawMap() {
  if (!grid || !data) return;
  const arr = layer === "estimate" ? grid.estimate : grid.variance;
  // Colour scale clipped to the 1st-99th percentile so a few extreme cells cannot flatten it.
  const sorted = Float32Array.from(arr).filter(Number.isFinite).sort();
  const lo = sorted[Math.floor(0.01 * (sorted.length - 1))] ?? 0;
  const hi = sorted[Math.ceil(0.99 * (sorted.length - 1))] ?? 1;
  const span = hi - lo || 1;
  off.width = grid.nx; off.height = grid.ny;
  const octx = off.getContext("2d")!;
  const img = octx.createImageData(grid.nx, grid.ny);
  // core.toNorthUp already made row 0 = north, matching ImageData row 0 = top.
  for (let k = 0; k < arr.length; k++) {
    const [r, g, b] = colour((arr[k] - lo) / span);
    img.data[4 * k] = r; img.data[4 * k + 1] = g; img.data[4 * k + 2] = b;
    img.data[4 * k + 3] = Number.isFinite(arr[k]) ? 255 : 0;
  }
  octx.putImageData(img, 0, 0);

  const ctx = mapCanvas.getContext("2d")!;
  const W = mapCanvas.width, H = mapCanvas.height, pad = 34;
  ctx.clearRect(0, 0, W, H);
  const e = grid.extent;
  const ew = e.xmax - e.xmin, eh = e.ymax - e.ymin;
  const scale = Math.min((W - 2 * pad) / ew, (H - 2 * pad) / eh); // equal-aspect: 1 m = 1 m
  mapRect = { w: ew * scale, h: eh * scale, x: (W - ew * scale) / 2, y: (H - eh * scale) / 2 };
  ctx.imageSmoothingEnabled = grid.nx >= 150;
  ctx.drawImage(off, mapRect.x, mapRect.y, mapRect.w, mapRect.h);

  if (showPtsEl.checked) {
    const n = data.xs.length;
    const rad = n > 2000 ? 1.2 : n > 500 ? 1.8 : 2.6;
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    ctx.strokeStyle = "rgba(0,0,0,0.8)";
    ctx.lineWidth = 0.7;
    for (let i = 0; i < n; i++) {
      const px = mapRect.x + (data.xs[i] - e.xmin) * scale;
      const py = mapRect.y + (e.ymax - data.ys[i]) * scale; // north up
      ctx.beginPath(); ctx.arc(px, py, rad, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
  }
  // Frame, corner coordinates, north arrow, scale bar.
  const fg = getComputedStyle(document.body).color;
  ctx.strokeStyle = fg; ctx.fillStyle = fg; ctx.lineWidth = 1;
  ctx.strokeRect(mapRect.x, mapRect.y, mapRect.w, mapRect.h);
  ctx.font = "11px system-ui";
  ctx.textAlign = "left"; ctx.fillText(`E ${e.xmin.toFixed(0)}`, mapRect.x, mapRect.y + mapRect.h + 14);
  ctx.textAlign = "right"; ctx.fillText(`E ${e.xmax.toFixed(0)}`, mapRect.x + mapRect.w, mapRect.y + mapRect.h + 14);
  ctx.fillText(`N ${e.ymax.toFixed(0)}`, mapRect.x + mapRect.w, mapRect.y - 5);
  ctx.textAlign = "left"; ctx.fillText(`N ${e.ymin.toFixed(0)}`, mapRect.x, mapRect.y + mapRect.h + 28);
  const ax = mapRect.x + 14, ay = mapRect.y + 10;
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000";
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax - 6, ay + 16); ctx.lineTo(ax + 6, ay + 16); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.font = "bold 11px system-ui"; ctx.textAlign = "center"; ctx.fillStyle = "#fff"; ctx.fillText("N", ax, ay + 30);
  const barM = (() => { const t = ew / 4, mag = 10 ** Math.floor(Math.log10(t)); return [5, 2.5, 2, 1].map((m) => m * mag).find((b) => b <= t) ?? t; })();
  const bx = mapRect.x + mapRect.w - 12 - barM * scale, by = mapRect.y + mapRect.h - 12;
  ctx.fillStyle = "#fff"; ctx.fillRect(bx, by - 3, barM * scale, 4); ctx.strokeRect(bx, by - 3, barM * scale, 4);
  ctx.font = "11px system-ui"; ctx.fillText(`${fmt(barM)} m`, bx + (barM * scale) / 2, by - 7);

  drawLegend(lo, hi);
  $("mapTitle").textContent = layer === "estimate" ? "Kriged estimate" : "Kriging variance";
}

function drawLegend(lo: number, hi: number) {
  const ctx = legendCanvas.getContext("2d")!;
  const W = legendCanvas.width, H = legendCanvas.height, top = 34, bot = H - 34, bw = 16;
  ctx.clearRect(0, 0, W, H);
  for (let y = top; y <= bot; y++) {
    const [r, g, b] = colour(1 - (y - top) / (bot - top));
    ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
    ctx.fillRect(4, y, bw, 1);
  }
  const fg = getComputedStyle(document.body).color;
  ctx.strokeStyle = fg; ctx.strokeRect(4, top, bw, bot - top);
  ctx.fillStyle = fg; ctx.font = "11px system-ui"; ctx.textAlign = "left";
  for (let i = 0; i <= 5; i++) {
    const v = lo + ((hi - lo) * i) / 5, y = bot - ((bot - top) * i) / 5;
    ctx.fillRect(4 + bw, y, 4, 1);
    ctx.fillText(fmt(v), 4 + bw + 7, y + 4);
  }
  ctx.fillText(layer === "estimate" ? "value" : "σ²", 4, top - 10);
  ctx.fillText("p1–p99", 4, bot + 22);
}

mapCanvas.addEventListener("mousemove", (ev) => {
  if (!grid) return;
  const b = mapCanvas.getBoundingClientRect();
  const px = ((ev.clientX - b.left) / b.width) * mapCanvas.width;
  const py = ((ev.clientY - b.top) / b.height) * mapCanvas.height;
  const u = (px - mapRect.x) / mapRect.w, v = (py - mapRect.y) / mapRect.h;
  if (u < 0 || u >= 1 || v < 0 || v >= 1) return void ($("hover").innerHTML = "&nbsp;");
  const c = Math.floor(u * grid.nx), r = Math.floor(v * grid.ny), k = r * grid.nx + c;
  const e = grid.extent;
  const x = e.xmin + u * (e.xmax - e.xmin), y = e.ymax - v * (e.ymax - e.ymin);
  $("hover").textContent = `E ${x.toFixed(0)}  N ${y.toFixed(0)}   estimate ${fmt(grid.estimate[k])}   variance ${fmt(grid.variance[k])}`;
});

// ---------- wiring ----------
$<HTMLButtonElement>("synth").onclick = () => {
  const n = Number($<HTMLSelectElement>("synthN").value);
  void loadSamples(makeSynthetic(n, 42), "Synthetic plume");
};
$<HTMLInputElement>("csv").onchange = async (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0];
  if (!file) return;
  try {
    await loadSamples(parseCsv(await file.text()), file.name);
  } catch (e) {
    setStatus(`CSV error: ${(e as Error).message}`, "err");
  }
};
familyEl.onchange = () => void runVariogram();
nBinsEl.onchange = () => void runVariogram();
resEl.onchange = () => void runKrige();
nbrsEl.onchange = () => void runKrige();
showPtsEl.onchange = () => drawMap();
document.querySelectorAll<HTMLInputElement>("input[name=layer]").forEach((el) =>
  el.addEventListener("change", () => { layer = el.value as typeof layer; drawMap(); }));

await workerReady;
setStatus("WASM ready");
void loadSamples(makeSynthetic(300, 42), "Synthetic plume");
