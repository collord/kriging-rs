# 2-D web starter: site kriging on projected coordinates

A small, typed Vite + TypeScript app built on `kriging-rs-wasm`. It is meant to be copied:
load `x, y, value` samples in projected metres, fit a variogram, krige a grid in a Web
Worker, and draw the estimate or kriging variance north-up with the samples on top.

## Run it

Build the package once, then start the app:

```bash
cd npm/kriging-rs-wasm && npm install && npm run build   # needs Rust + wasm-pack
cd ../../examples/web-2d && npm install
npm run dev      # http://localhost:5173
npm run build    # strict tsc, then a production bundle in dist/
npm run check    # runs src/core.ts headlessly in Node and checks the results
```

## Files

| File | Role |
|------|------|
| [src/core.ts](src/core.ts) | The whole kriging pipeline, with no DOM code. Start reading here. |
| [src/worker.ts](src/worker.ts) | Calls `init()` once, holds the dataset, answers typed requests. All WASM work runs here. |
| [src/main.ts](src/main.ts) | UI thread: controls, SVG variogram plot, canvas map, legend, hover readout. Imports only types from the package. |
| [scripts/check.ts](scripts/check.ts) | Runs `core.ts` in Node (300 and 3000 samples): finite output, north-east hotspot, error against the true field. |
| [vite.config.ts](vite.config.ts) | `server.fs.allow` for the `file:` dependency; ES-module worker. |

## Library calls

| Step | Call | Notes |
|------|------|-------|
| Load | `await init()` | In a browser or worker it fetches the `.wasm` beside the package JS (Vite emits it as an asset). In Node, pass the bytes; see `scripts/check.ts`. |
| Experimental variogram | `computeEmpiricalVariogramProjected({ xs, ys, values, maxDistance, nBins })` | Euclidean, in coordinate units. O(n²), so `core.ts` subsamples above 1500 points. |
| Fit | `fitVariogramProjected({ ..., variogramType })` | `range` comes back in metres. |
| Model curve | `evaluateNestedVariogram([fitted], h)` | The library's own formulas. Start `h` just above 0: `γ(0) = 0` exactly and the nugget is the jump at 0⁺. |
| Krige | `ProjectedKriging.fromFitted({ ..., neighborhood: { maxNeighbors: 32 } })` then `predictGrid(...)` | The neighborhood keeps thousands of samples fast (~0.5 s for 200×200). Without one, the model factorizes the full n×n system. |
| Clean up | `model.free()` | WASM memory is not garbage-collected. |

## Conventions that matter

- **Units.** The variogram `range`, `maxDistance` and `maxRadius` are all in the
  coordinates' units. Raw UTM is fine: the package centres coordinates before its `f32`
  core sees them.
- **Grid orientation.** `predictGrid` returns row 0 = **south** (`yMin`), like a map's
  y axis. Canvas and `ImageData` row 0 is the **top**. `core.ts` flips once, in
  `toNorthUp`, so everything after it is north-up. If you skip that step the map comes out
  upside down.
- **Variance includes the nugget.** Near a sample the kriging variance tends to the nugget,
  not to zero, unless you krige exactly at the sample.
- **Workers.** Kriging a grid takes hundreds of milliseconds, so keep it off the UI thread.
  The worker posts results back with transferable `Float32Array`s, and the UI drops stale
  replies using a generation counter.
- **TypeScript.** The package's `.d.ts` files declare `[Symbol.dispose]`, so `tsconfig`
  needs `"ESNext.Disposable"` in `lib` (see [tsconfig.json](tsconfig.json)).

## Adapting it

- **Lat/lon data:** use `fitVariogram` and `OrdinaryKriging` (Haversine, kilometres) or
  project to metres first. Lat/lon `predictGrid` uses the same row-0-south convention.
- **Anisotropy:** compare `computeDirectionalEmpiricalVariogram` along a few azimuths, then
  pass `majorAngleDeg` / `rangeRatio` to `ProjectedKriging.fromFitted`.
- **Validation:** `leaveOneOutProjected` / `kFoldProjected` report RMSE and MSDR; MSDR ≈ 1
  means the variogram's variance is well calibrated.
- **Count data:** `BinomialProjectedKriging` for prevalence surfaces from successes/trials.
- **Uncertainty maps:** `conditionalSimulateProjected` draws realizations for
  exceedance probabilities.
