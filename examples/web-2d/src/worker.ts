// All WASM work runs here so the page stays responsive while kriging.
// Pattern: init() once, keep the dataset in the worker, answer typed requests.
import init, { KrigingError } from "kriging-rs-wasm";
import type { FittedVariogram, VariogramTypeName } from "kriging-rs-wasm";
import { extentOf, krigeGrid, padExtent, variogram, type Samples } from "./core";

export type Request =
  | { id: number; type: "load"; xs: Float64Array; ys: Float64Array; values: Float64Array }
  | { id: number; type: "variogram"; family: VariogramTypeName; nBins: number }
  | { id: number; type: "krige"; model: FittedVariogram; nx: number; ny: number; maxNeighbors: number };

export type Response =
  | { id: number; ok: true; type: "ready" }
  | { id: number; ok: true; type: "load"; n: number }
  | { id: number; ok: true; type: "variogram"; result: ReturnType<typeof variogram>; ms: number }
  | { id: number; ok: true; type: "krige"; result: ReturnType<typeof krigeGrid> }
  | { id: number; ok: false; error: string; code?: string };

// Minimal typing for the dedicated-worker global (avoids mixing DOM and WebWorker libs).
const ctx = self as unknown as {
  postMessage(msg: Response, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent<Request>) => void) | null;
};

let samples: Samples | null = null;
// In a browser, init() fetches the .wasm next to the package's JS (Vite emits it as an asset).
const ready = init().then(() => ctx.postMessage({ id: 0, ok: true, type: "ready" }));

ctx.onmessage = async (ev) => {
  const req = ev.data;
  try {
    await ready;
    switch (req.type) {
      case "load":
        samples = { xs: req.xs, ys: req.ys, values: req.values };
        ctx.postMessage({ id: req.id, ok: true, type: "load", n: req.xs.length });
        break;
      case "variogram": {
        if (!samples) throw new Error("no data loaded");
        const t0 = performance.now();
        const result = variogram(samples, req.family, req.nBins);
        ctx.postMessage({ id: req.id, ok: true, type: "variogram", result, ms: performance.now() - t0 });
        break;
      }
      case "krige": {
        if (!samples) throw new Error("no data loaded");
        const extent = padExtent(extentOf(samples));
        const result = krigeGrid(samples, req.model, req.nx, req.ny, extent, req.maxNeighbors);
        // Transfer the grids instead of copying them.
        ctx.postMessage({ id: req.id, ok: true, type: "krige", result }, [
          result.estimate.buffer,
          result.variance.buffer,
        ]);
        break;
      }
    }
  } catch (e) {
    // KrigingError carries a stable `code` (e.g. "singular_covariance") for UI messages.
    const code = e instanceof KrigingError ? e.code : undefined;
    ctx.postMessage({ id: req.id, ok: false, error: e instanceof Error ? e.message : String(e), code });
  }
};
