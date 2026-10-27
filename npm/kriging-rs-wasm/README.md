# kriging-rs-wasm

TypeScript-first WebAssembly package for [kriging-rs](https://github.com/m-murphy/kriging-rs): ordinary / simple / universal / binomial kriging (both 2-D spatial and 2+1-D spatio-temporal) with optional WebGPU acceleration.

## Quick start

1. Install: `npm install kriging-rs-wasm`
2. Initialize once: `await init()`
3. Build a model and predict:

```ts
import init, { OrdinaryKriging } from "kriging-rs-wasm";
await init();

const model = new OrdinaryKriging({
  lats: [37.7, 37.71, 37.72],
  lons: [-122.45, -122.44, -122.43],
  values: [10, 12, 11],
  variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1.5, range: 5.0 },
});
const pred = model.predict(37.705, -122.435);
```

## Requirements

- Browser or Node.js; ES2022. For custom WASM loading you can pass pre-fetched bytes to `init(wasmArrayBuffer)`.

## API overview

| Export | Purpose |
|--------|---------|
| `init` | Initialize the WASM module (call and await once before any other API). |
| `OrdinaryKriging` | Spatial interpolation of **continuous** values (e.g. temperature, elevation). Supports search neighborhoods via `setNeighborhood` / `neighborhood`. |
| `SimpleKriging` | Ordinary kriging's fixed-mean analogue; requires a known `mean`. |
| `UniversalKriging` | Kriging with a polynomial drift basis (`"constant"` / `"linear"` / `"quadratic"` in lat/lon). |
| `ProjectedKriging` | Planar kriging on `(x, y)` coordinates with 2D anisotropy (`majorAngleDeg`, `rangeRatio`). |
| `BinomialKriging` | **Prevalence/proportion** surfaces from count data (successes out of trials). Also `fromPrecomputedLogits` to bypass empirical-Bayes shrinkage. |
| `fitVariogram` | Fit a variogram model to sample data; optional `estimator: "classical" \| "cressie-hawkins"`. Use the result to build an `OrdinaryKriging` model. |
| `computeEmpiricalVariogram` | Compute the (isotropic) empirical variogram cloud directly, without fitting a parametric model. |
| `computeDirectionalEmpiricalVariogram` | Directional empirical variogram on planar `(x, y)` data for diagnosing anisotropy. |
| `leaveOneOut` / `kFold` | Cross-validation on ordinary kriging. |
| `leaveOneOutSimple` / `kFoldSimple` | Cross-validation on simple kriging (known, constant mean). |
| `leaveOneOutUniversal` / `kFoldUniversal` | Cross-validation on universal kriging with a constant / linear / quadratic drift; drift refit per fold. |
| `leaveOneOutProjected` / `kFoldProjected` | Cross-validation on projected (planar) kriging with optional 2-D anisotropy. |
| `leaveOneOutBinomial` / `kFoldBinomial` | Cross-validation on binomial kriging; reports residuals on **both** the logit and prevalence scales. |
| `conditionalSimulate` | Sequential Gaussian simulation (ordinary kriging); deterministic for a given `seed`. |
| `conditionalSimulateSimple` | SGS with simple kriging (known `mean`). |
| `conditionalSimulateUniversal` | SGS with universal kriging (polynomial drift refit per step). |
| `conditionalSimulateProjected` | SGS on planar `(x, y)` coordinates with optional 2-D anisotropy. |
| `conditionalSimulateBinomial` | SGS for count data on the logit scale; returns samples on **both** the logit and prevalence scales. |
| `evaluateNestedVariogram` | Evaluate a nested (additive) variogram at given distances for offline inspection/plotting. |
| `interpolateOrdinaryToGrid` | One-shot: fit + build + predict on grid + free; returns value and variance grids. |
| `interpolateBinomialToGrid` | One-shot: fit + build + predict on grid + free; returns prevalence and variance grids. |
| `VariogramType` | Enum for variogram model type (optional; you can pass string names like `"exponential"` instead). |
| `KrigingError` | Error class thrown on invalid inputs or model build failure; `cause` holds the underlying error; optional `code` for UI (e.g. `singular_covariance`, `mismatched_arrays`). |
| `KrigingErrorCode` | Type of stable error codes (`not_loaded`, `model_freed`, `mismatched_arrays`, etc.). |
| `webgpuAvailable` | Check if WebGPU-backed batch prediction is available (requires GPU build). |
| `SpaceTimeOrdinaryKriging` | Spatio-temporal ordinary kriging over `(lat, lon, time)`. |
| `SpaceTimeSimpleKriging` | ST simple kriging with a known, constant mean. |
| `SpaceTimeUniversalKriging` | ST universal kriging with polynomial trends in space and/or time. |
| `SpaceTimeBinomialKriging` | ST binomial kriging from `(successes, trials)` over `(lat, lon, time)`. |
| `SpaceTimeProjectedOrdinaryKriging` | ST ordinary kriging on projected `(x, y, time)` with 2-D anisotropy. |
| `computeEmpiricalSpaceTimeVariogram` | 2-D empirical space-time variogram (spatial × temporal bins). |
| `fitSpaceTimeVariogram` | Fit a separable or product-sum ST variogram against the empirical surface. |
| `leaveOneOutSpaceTime` / `kFoldSpaceTime` | Cross-validation on ST ordinary kriging. |
| `leaveOneOutSpaceTimeSimple` / `kFoldSpaceTimeSimple` | Cross-validation on ST simple kriging (known `mean`). |
| `leaveOneOutSpaceTimeUniversal` / `kFoldSpaceTimeUniversal` | Cross-validation on ST universal kriging; trend refit per fold. |
| `leaveOneOutSpaceTimeBinomial` / `kFoldSpaceTimeBinomial` | Cross-validation on ST binomial kriging; dual-scale residuals. |
| `conditionalSimulateSpaceTime` | Sequential Gaussian simulation for ST ordinary kriging; deterministic for a given `seed`. |
| `conditionalSimulateSpaceTimeSimple` | ST SGS with simple kriging (known `mean`). |
| `conditionalSimulateSpaceTimeUniversal` | ST SGS with universal kriging (polynomial trend refit per step). |
| `conditionalSimulateSpaceTimeBinomial` | ST SGS for count data on the logit scale; returns both logit and prevalence samples. |

**When to use which:** Use **ordinary kriging** when you have continuous measurements at locations (e.g. sensor values, elevations). Use **binomial kriging** when you have counts (successes and trials) and want to estimate a proportion or prevalence surface.

## Build

From this directory:

```bash
npm install
npm run build
```

The build performs:

- `wasm-pack` generation into `pkg/`
- TypeScript facade compilation into `dist/`

For WebGPU-backed batch prediction, use:

```bash
npm run build:wasm:gpu
npm run build:ts
```

## Verify

```bash
npm run verify
```

This checks:

- Type contracts (`tsc --noEmit`)
- WASM + TypeScript build
- Runtime smoke test
- Test suite (`npm test`)

## Testing in your app

The library loads WASM synchronously after you call `init()` (optionally with pre-fetched bytes). In test runners:

- **Node:** Call and await `init()` before any test that uses the API. You can pass WASM bytes from disk, e.g. `await init(await readFile(pathToWasm))`.
- **Vitest / Vite:** Some setups (e.g. Vite 8 with SSR or certain module resolutions) can throw when importing this package (e.g. `__vite_ssr_exportName__ is not defined`). If that happens, try: (1) running tests in Node (Vitest's `pool: 'node'` or `environment: 'node'`), or (2) loading WASM in a setup file and re-exporting a ready flag, or (3) using `vi.mock` to provide a test double for the module. The package's own tests use Vitest with `init(wasmBytes)` in a `beforeAll` and load the built WASM from `pkg/`.

## Usage

Call and await `init()` once before using any other API. **VariogramType** (e.g. `VariogramType.Exponential`) is only safe to use after `init()` has been awaited; accessing it before init will throw when the value is used. You can pass string names like `"exponential"` instead.

Supported variogram types: `"spherical"`, `"exponential"`, `"gaussian"`, `"cubic"`, `"stable"`, `"matern"`, `"power"`, `"holeeffect"`. You can pass the model type as a string (e.g. `"exponential"`) or as `VariogramType.Exponential`. For `stable` and `matern`, pass an optional `shape` when constructing a model; `fitVariogram` returns a `shape` field when the fitted model is stable or Matérn.

### Ordinary kriging

```ts
import init, {
  OrdinaryKriging,
  fitVariogram,
} from "kriging-rs-wasm";

await init();

const model = new OrdinaryKriging({
  lats: [37.7, 37.71, 37.72],
  lons: [-122.45, -122.44, -122.43],
  values: [10, 12, 11],
  variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1.5, range: 5.0 },
});

const prediction = model.predict(37.705, -122.435);

// Fit variogram from data (options object; variogramType can be string or VariogramType enum)
const lats = [37.7, 37.71, 37.72];
const lons = [-122.45, -122.44, -122.43];
const values = [10, 12, 11];
const fitted = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: "exponential",
  nBins: 12,  // optional; default 12
});
const fittedModel = new OrdinaryKriging({
  lats,
  lons,
  values,
  variogram: {
    variogramType: fitted.variogramType,
    nugget: fitted.nugget,
    sill: fitted.sill,
    range: fitted.range,
    shape: fitted.shape,  // optional; used for stable/matern
  },
});
const batch = fittedModel.predictBatch(lats, lons);
```

**Convenience factories (fit → model):** To avoid manually spreading `fitted` fields, use the static factories:

- **Ordinary:** `OrdinaryKriging.fromFitted({ lats, lons, values, fittedVariogram: fitted })`. Optional `nuggetOverride` overrides the fitted nugget (e.g. for a UI-tuned sigma²).
- **Binomial:** `BinomialKriging.fromFittedVariogram({ lats, lons, successes, trials, fittedVariogram })`. Optional `nuggetOverride` overrides the fitted nugget.
- **Binomial with prior:** `BinomialKriging.fromFittedVariogramWithPrior({ lats, lons, successes, trials, fittedVariogram, prior: { alpha, beta } })`. Optional `nuggetOverride` overrides the fitted nugget.

Example:

```ts
const fitted = fitVariogram({
  sampleLats: lats,
  sampleLons: lons,
  values,
  variogramType: "exponential",
  nBins: 12,
});
const model = OrdinaryKriging.fromFitted({ lats, lons, values, fittedVariogram: fitted });
const pred = model.predict(37.705, -122.435);
```

### Binomial kriging (prevalence surfaces)

For count data (successes out of trials) at locations. **For binary 0/1 data** (e.g. presence/absence at each point), use `successes = values` and `trials = 1` at each location.

```ts
import init, { BinomialKriging } from "kriging-rs-wasm";

await init();

const lats = [37.7, 37.71, 37.72];
const lons = [-122.45, -122.44, -122.43];
const successes = [2, 5, 3];  // counts
const trials = [10, 10, 10];

const model = new BinomialKriging({
  lats,
  lons,
  successes,
  trials,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 100 },
});
const pred = model.predict(37.705, -122.435);
// pred.prevalence in [0, 1], pred.variance, pred.logitValue
```

With a Beta prior (e.g. when counts are small):

```ts
const model = BinomialKriging.newWithPrior({
  lats,
  lons,
  successes,
  trials,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 100 },
  prior: { alpha: 1, beta: 1 },
});
```

**Model contract (Rust / wasm consumers):** The default `BinomialKriging` path is
empirical-Bayes–smoothed logit + **ordinary kriging with per-site logit observation
variance** (calibrated default) + `logistic`, not a full binomial-likelihood field
model. `getBuildNotes()` returns `BinomialBuildNotes` (version, prior, logit
inflation, etc.); rows with `trials === 0` are dropped. See
`benches/BROWSER_BENCHMARKS.md` in the repository for large-grid prediction
benchmarks.

### Batch prediction and typed arrays

For large prediction grids, use `predictBatchArrays` to get `Float64Array` outputs and avoid per-point object allocation:

```ts
const { values, variances } = model.predictBatchArrays(gridLats, gridLons);
// values.length === gridLats.length; same for variances
```

For ordinary kriging the result is `{ values, variances }`; for binomial it is `{ prevalences, logitValues, variances }`.

### Grid prediction (bounds + resolution)

For a rectangular grid defined by bounds and cell counts, use `predictGrid` so the library handles building cell-center coordinates and reshaping results:

```ts
const { values, variances } = model.predictGrid({
  west: -122.5,
  south: 37.6,
  east: -122.3,
  north: 37.8,
  xCells: 50,
  yCells: 40,
});
```

**Grid layout:** Results are 2D arrays (not flat). Row index = latitude index, column index = longitude index. First row (`j = 0`) = south, last row = north; first column (`i = 0`) = west, last column = east. So `values[j][i]` is the prediction at the cell with latitude row `j` and longitude column `i`. Internally the library uses row-major order (south to north, then west to east within each row). Ordinary kriging returns `{ values, variances }`; binomial returns `{ prevalences, logitValues, variances }`, all with shape `[yCells][xCells]`.

### Search neighborhoods (ordinary kriging)

By default every prediction uses all sample stations, which scales as O(n³) per model build and O(n) per prediction. For large datasets or locally-supported variograms you can restrict each prediction to the closest stations (by count and/or radius):

```ts
const model = new OrdinaryKriging({ lats, lons, values, variogram });

model.setNeighborhood({ maxNeighbors: 32, maxRadius: 50 }); // km
// ...predict(), predictBatch(), predictBatchArrays() all now use the neighborhood
const current = model.neighborhood(); // { maxNeighbors: 32, maxRadius: 50 }

model.setNeighborhood({}); // clear; back to full-data fast path
```

If both are given, the intersection is used (top-K within radius). `maxRadius` must be finite and positive.

### Simple kriging (known mean)

Use `SimpleKriging` when the spatial mean is known or estimated externally. The residual system uses the supplied `mean` as a fixed offset rather than estimating it from the data.

```ts
import { SimpleKriging } from "kriging-rs-wasm";

const model = new SimpleKriging({
  lats, lons, values,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 50 },
  mean: 12.3,
});
const { values: preds, variances } = model.predictBatchArrays(gridLats, gridLons);
model.free();
```

### Universal kriging (polynomial drift)

Use `UniversalKriging` when the field has a systematic trend in lat/lon that ordinary kriging would smear into its residual variogram.

```ts
import { UniversalKriging } from "kriging-rs-wasm";

const model = new UniversalKriging({
  lats, lons, values,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 50 },
  trend: "linear", // "constant" | "linear" | "quadratic"
});
```

- `"constant"` — equivalent to ordinary kriging.
- `"linear"` — drift basis `[1, lat, lon]`.
- `"quadratic"` — `[1, lat, lon, lat², lat·lon, lon²]`.

### Projected (planar) kriging with 2D anisotropy

For already-projected coordinates (e.g. meters in a local frame), use `ProjectedKriging`. Distances are Euclidean, and 2D anisotropy is expressed as a rotation angle plus a minor-to-major range ratio:

```ts
import { ProjectedKriging } from "kriging-rs-wasm";

const model = new ProjectedKriging({
  xs, ys, values,
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 500 },
  majorAngleDeg: 30,    // CCW from +x
  rangeRatio: 0.5,      // minor / major range, in (0, 1]
});
```

### Binomial kriging from pre-computed logits

When you have finite logit estimates from an external fit (e.g. a mean-field model) and want to skip the default empirical-Bayes shrinkage:

```ts
const model = BinomialKriging.fromPrecomputedLogits({
  lats, lons,
  logits, // Float64Array / number[] of finite logit values per station
  variogram: { variogramType: "exponential", nugget: 0.01, sill: 1.0, range: 100 },
});
```

Non-finite logits throw `KrigingError` with code `invalid_input`.

### Empirical variograms (without fitting)

For diagnostics, or when you want to plug the empirical cloud into a custom fitter:

```ts
import { computeEmpiricalVariogram, computeDirectionalEmpiricalVariogram } from "kriging-rs-wasm";

// Isotropic (geographic lat/lon; Haversine distance in km)
const iso = computeEmpiricalVariogram({
  sampleLats: lats, sampleLons: lons, values,
  nBins: 20,                  // default 12
  estimator: "cressie-hawkins", // or "classical" (default)
});
// iso.distances, iso.semivariances (Float64Array), iso.counts (Uint32Array)

// Directional (planar x/y; Euclidean distance)
const azimuth = computeDirectionalEmpiricalVariogram({
  xs, ys, values,
  maxDistance: 2000,
  nBins: 20,
  azimuthDeg: 0,     // along +x
  toleranceDeg: 22.5,
});
```

The same `estimator` option is accepted by `fitVariogram({ ..., estimator: "cressie-hawkins" })` for robust fits.

### Cross-validation

Diagnose predictive skill and variogram calibration via leave-one-out or K-fold CV. Per-station residuals include observed/predicted values and kriging variance; the summary reports `n`, `meanError`, `rmse`, and `msdr` (mean squared deviation ratio; ≈ 1 when the variogram is well calibrated). Folds are deterministic round-robin (station `i` → fold `i % k`); shuffle inputs for randomized folds.

Every kriging variant ships a CV entry point:

```ts
import {
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
} from "kriging-rs-wasm";

// Ordinary (no trend, unknown mean).
const ordinary = leaveOneOut({ lats, lons, values, variogram });
const ordinaryK = kFold({ lats, lons, values, variogram, k: 10 });

// Simple (known mean; held fixed across folds).
const simple = leaveOneOutSimple({ lats, lons, values, variogram, mean: 12.5 });

// Universal (polynomial drift; refit per fold, no leakage).
const universal = leaveOneOutUniversal({
  lats, lons, values, variogram,
  trend: "linear", // "constant" | "linear" | "quadratic"
});

// Projected (planar x/y; optional 2-D anisotropy).
const projected = leaveOneOutProjected({
  xs, ys, values, variogram,
  majorAngleDeg: 45,
  rangeRatio: 0.5, // pass 1 for isotropic
});
```

All of the above return `{ residuals, summary, arrays }` where `arrays` contains typed `Float64Array`/`Uint32Array` views suitable for plotting.

#### Binomial cross-validation (both scales)

Binomial CV is special: a held-out station has a logit-scale observation (natural for MSDR and variogram calibration) **and** a prevalence-scale observation (natural for probability-scale error metrics). `leaveOneOutBinomial` / `kFoldBinomial` report both, along with a two-sided summary.

```ts
const binomial = leaveOneOutBinomial({
  lats, lons, successes, trials, variogram,
  // priorAlpha, priorBeta — optional; must appear together, defaults to Beta(1, 1).
});

// Each residual carries both scales:
const r = binomial.residuals[0];
r.observedLogit;          // NaN when trials[i] === 0
r.predictedLogit;
r.logitVariance;          // kriging variance on logit scale
r.observedPrevalence;     // NaN when trials[i] === 0
r.predictedPrevalence;    // in [0, 1]
r.prevalenceVariance;     // delta-method approximation
r.logitError;             // observedLogit − predictedLogit (NaN for trials === 0)
r.prevalenceError;        // observedPrevalence − predictedPrevalence (NaN for trials === 0)

// Summary reports both scales; aggregates skip trials === 0 automatically.
binomial.summary.n;            // total residuals (includes trials === 0)
binomial.summary.nEvaluated;   // residuals with trials > 0
binomial.summary.logit;        // { n, meanError, rmse, msdr }
binomial.summary.prevalence;   // { n, meanError, rmse, msdr }
```

Stations with `trials[i] === 0` are unobservable: they contribute no training fold and their observed fields are `NaN`, but the model's prediction is still populated so you can audit which stations were unobservable.

### Conditional simulation (Sequential Gaussian Simulation)

Generate realizations of the spatial process conditioned on observed stations. Deterministic for a given `seed`; pass a different seed for each realization. One helper per kriging variant, mirroring the cross-validation surface:

| Helper | When to use |
| --- | --- |
| `conditionalSimulate` | Ordinary kriging — unknown mean, no trend. |
| `conditionalSimulateSimple` | You have an externally known **mean** (climatology, pooled historical average). |
| `conditionalSimulateUniversal` | The process has a known polynomial drift (`"constant"`, `"linear"`, `"quadratic"`). |
| `conditionalSimulateProjected` | Data is on planar `(x, y)` coordinates; optional 2-D geometric anisotropy. |
| `conditionalSimulateBinomial` | Count data (`successes`, `trials`). Simulation happens on the logit scale; result is reported on **both** scales. |

```ts
import {
  conditionalSimulate,
  conditionalSimulateSimple,
  conditionalSimulateUniversal,
  conditionalSimulateProjected,
  conditionalSimulateBinomial,
} from "kriging-rs-wasm";

// Ordinary (no trend)
const sample = conditionalSimulate({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningValues: values,
  targetLats: gridLats,
  targetLons: gridLons,
  variogram,
  seed: 42n,                 // bigint or number; default 0n
  // targetOrder: Uint32Array.of(...) // optional visit order; default is input order
});
// sample: Float64Array, length === targetLats.length

// Simple (known mean)
const simple = conditionalSimulateSimple({
  conditioningLats: lats, conditioningLons: lons, conditioningValues: values,
  targetLats: gridLats, targetLons: gridLons,
  variogram, mean: 11.5, seed: 7,
});

// Universal (polynomial drift refit at each step)
const uni = conditionalSimulateUniversal({
  conditioningLats: lats, conditioningLons: lons, conditioningValues: values,
  targetLats: gridLats, targetLons: gridLons,
  variogram, trend: "linear", seed: 7,
});

// Projected (planar, optional anisotropy)
const proj = conditionalSimulateProjected({
  conditioningXs: xs, conditioningYs: ys, conditioningValues: values,
  targetXs: gridXs, targetYs: gridYs,
  variogram, majorAngleDeg: 30, rangeRatio: 0.5, seed: 7,
});

// Binomial (dual-scale output)
const bin = conditionalSimulateBinomial({
  conditioningLats: lats, conditioningLons: lons,
  successes, trials,
  targetLats: gridLats, targetLons: gridLons,
  variogram, seed: 7,
  // priorAlpha: 0.5, priorBeta: 0.5   // optional Jeffreys; default Beta(1, 1)
});
bin.logitSamples;      // Float64Array — simulated logit values (unbounded)
bin.prevalenceSamples; // Float64Array — logistic(logitSamples), in (0, 1)
```

Binomial stations with `trials[i] === 0` are dropped from the initial conditioning pool (they carry no information); the simulator still requires at least two remaining valid stations. Every variant honors the observed data exactly (up to solver tolerance) and visits targets in input order unless `targetOrder` is supplied.

### Nested variogram evaluation

Evaluate an additive composite variogram (e.g. short-range exponential plus long-range spherical) at a set of distances. The model is summed on the semivariance scale; covariance is derived via `C₀ − γ(h)`.

```ts
import { evaluateNestedVariogram } from "kriging-rs-wasm";

const { distances: h, semivariances, covariances } = evaluateNestedVariogram(
  [
    { variogramType: "exponential", nugget: 0.0, sill: 0.3, range: 5 },
    { variogramType: "spherical",  nugget: 0.0, sill: 0.7, range: 50 },
  ],
  [0, 1, 2, 5, 10, 25, 50, 75, 100],
);
```

`γ(0)` is exactly `0`: the nugget is the jump just past the origin, so the semivariance at the
first positive distance already includes it. When plotting a model curve, start the distance
grid just above zero to draw the nugget as the intercept.

Nested variograms are not yet accepted by the kriging constructors; this entry point is for offline evaluation and plotting.

### One-shot interpolate to grid

For the common flow "fit variogram → build model → predict on grid → free", you can use a single call:

```ts
// Ordinary: sample data + grid spec + variogram type
const { values, variances } = interpolateOrdinaryToGrid({
  lats, lons, values: sampleValues,
  west: -122.5, south: 37.6, east: -122.3, north: 37.8,
  xCells: 50, yCells: 40,
  variogramType: "exponential",
  nBins: 12,
  nuggetOverride: 0.01,  // optional
});

// Binomial: count data + grid spec + variogram type; optional prior
const { prevalences, variances } = interpolateBinomialToGrid({
  lats, lons, successes, trials,
  west: -122.5, south: 37.6, east: -122.3, north: 37.8,
  xCells: 50, yCells: 40,
  variogramType: "exponential",
  prior: { alpha: 1, beta: 1 },  // optional
});
```

The model is created internally and freed before returning, so you do not need to call `free()`.

### WebGPU (optional)

If the package is built with the GPU feature (`npm run build:wasm:gpu`), you can use async batch prediction and check availability:

```ts
import init, { OrdinaryKriging, webgpuAvailable } from "kriging-rs-wasm";
await init();

if (await webgpuAvailable()) {
  const predictions = await model.predictBatchGpu(gridLats, gridLons);
}
```

If `predictBatchGpu` is called without a GPU build, it throws.

### Space-time kriging

All four kriging families are mirrored in the `SpaceTime*` classes, which add a `time`
axis to the usual `(lat, lon)` inputs. Use `"separable"` when the space and time processes
can be modeled independently, and `"productSum"` when you need an additive interaction
term on top of the marginals (`k1·C_s·C_t + k2·C_s + k3·C_t`). The two families form a
TypeScript discriminated union, so `k1`, `k2`, and `k3` are only accepted (and required)
when `family: "productSum"`.

Use `fromFitted(...)` to pipe the result of `fitSpaceTimeVariogram` straight into a model
without re-stating the variogram fields:

```ts
import init, {
  SpaceTimeOrdinaryKriging,
  fitSpaceTimeVariogram,
} from "kriging-rs-wasm";
await init();

const fit = fitSpaceTimeVariogram({
  lats,
  lons,
  times,
  values,
  nSpatialBins: 10,
  nTemporalBins: 6,
  family: "separable",
  spatialModel: "exponential",
  temporalModel: "exponential",
});

using model = SpaceTimeOrdinaryKriging.fromFitted({
  lats,
  lons,
  times,
  values,
  fittedVariogram: fit.fit,
});

const { value, variance } = model.predict(37.77, -122.42, 1.5);
```

The `using` declaration (ES2023) calls `model[Symbol.dispose]()` — equivalent to
`model.free()` — when the block exits, so you don't have to remember to release
WASM-held memory. This works on every 2-D and space-time kriging class.

If you have the variogram parameters by hand, the regular constructor also accepts
`fit.fit` directly thanks to the discriminated union:

```ts
const model = new SpaceTimeOrdinaryKriging({
  lats,
  lons,
  times,
  values,
  variogram: fit.fit,
});
```

To render a 2-D heatmap at a specific time slice, use `predictGridAtTime`:

```ts
const grid = model.predictGridAtTime({
  west: -122.5,
  south: 37.7,
  east: -122.35,
  north: 37.82,
  xCells: 64,
  yCells: 48,
  time: 1.5,
});
// grid.values[j][i] and grid.variances[j][i], shape [yCells][xCells]
```

Dates as the time axis: convert between `Date[]` and the scalar time axis via
`timesFromDates` / `datesFromTimes` with an optional `TimeUnit` (`"ms" | "s" |
"minutes" | "hours" | "days"`):

```ts
import { timesFromDates, datesFromTimes } from "kriging-rs-wasm";

const times = timesFromDates(dateObjects, "days");
// ...build and use the model...
const dates = datesFromTimes(times, "days");
```

Universal space-time kriging adds a polynomial trend via `trend`: one of `"constant"`,
`"linearInTime"`, `"quadraticInTime"`, `"linearInSpace"`, `"linearInSpaceAndTime"`,
`"quadraticInSpaceAndTime"`. Binomial space-time kriging takes `successes`/`trials`
arrays in addition to `lats`/`lons`/`times` and reports prevalence on `[0, 1]`.

Space-time cross-validation mirrors the 2-D surface — `leaveOneOutSpaceTime` / `kFoldSpaceTime`
for ordinary, plus `_simple`, `_universal`, and `_binomial` variants (the last returns dual-scale
logit+prevalence residuals):

```ts
import { leaveOneOutSpaceTime, kFoldSpaceTimeBinomial } from "kriging-rs-wasm";

const cv = leaveOneOutSpaceTime({
  lats,
  lons,
  times,
  values,
  variogram: { family: "separable", spatial, temporal },
});
console.log(cv.summary.rmse, cv.summary.msdr);

const bin = kFoldSpaceTimeBinomial({
  lats,
  lons,
  times,
  successes,
  trials,
  variogram: { family: "separable", spatial, temporal },
  k: 5,
});
console.log(bin.summary.logit.rmse, bin.summary.prevalence.rmse);
```

Space-time conditional simulation (Sequential Gaussian Simulation) is also available for every
ST variant; results are deterministic for a given `seed`. `conditionalSimulateSpaceTimeBinomial`
samples on the logit scale and returns both logit and prevalence samples:

```ts
import {
  conditionalSimulateSpaceTime,
  conditionalSimulateSpaceTimeBinomial,
} from "kriging-rs-wasm";

const samples = conditionalSimulateSpaceTime({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats,
  targetLons,
  targetTimes,
  variogram: { family: "separable", spatial, temporal },
  seed: 42n,
});

const binSamples = conditionalSimulateSpaceTimeBinomial({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  successes,
  trials,
  targetLats,
  targetLons,
  targetTimes,
  variogram: { family: "separable", spatial, temporal },
  prior: { alpha: 1, beta: 1 }, // optional; alpha and beta are set together
  seed: 42n,
});
// binSamples.logitSamples, binSamples.prevalenceSamples
```

To draw multiple independent realizations in one call, use `conditionalSimulateMany`
(2-D) or `conditionalSimulateManySpaceTime`. Each draw uses `baseSeed + k`, so the
result is deterministic and trivially reproducible:

```ts
import { conditionalSimulateManySpaceTime } from "kriging-rs-wasm";

const realizations = conditionalSimulateManySpaceTime({
  conditioningLats: lats,
  conditioningLons: lons,
  conditioningTimes: times,
  conditioningValues: values,
  targetLats,
  targetLons,
  targetTimes,
  variogram: { family: "separable", spatial, temporal },
  nRealizations: 100,
  baseSeed: 42,
});
// Flat Float64Array of length nRealizations * targetLats.length, row-major
// (row k holds the k-th realization in input order).
```

## Disease prevalence mapping cookbook

End-to-end recipe for the package's primary use case: turning sparse
`(latitude, longitude, successes, trials)` (or `(x, y, …)` for projected data)
into a defensible prevalence surface plus calibrated uncertainty. The pieces
below combine into the typical workflow **fit → validate → simulate → aggregate**.

### 1. One-shot map (great for a first look)

`interpolateBinomialToGrid` fits a variogram on the EB-smoothed logits, builds
a `BinomialKriging` model, predicts a grid, and (optionally) runs CV — all in
a single call. The fitted variogram is returned so you can reuse it later
without re-fitting.

```ts
import init, { interpolateBinomialToGrid } from "kriging-rs-wasm";
await init();

const result = interpolateBinomialToGrid({
  lats, lons, successes, trials,
  west: -1, south: 36, east: 1, north: 38,
  xCells: 100, yCells: 100,
  variogramType: "exponential",
  prior: { alpha: 1, beta: 1 },           // optional Beta prior
  estimator: "cressie-hawkins",            // robust empirical variogram
  withCv: { k: 5 },                        // optional 5-fold CV summary
});
result.prevalences;       // [yCells][xCells]
result.fittedVariogram;   // re-usable for simulation
result.buildNotes;        // prior, dropped zero-trial rows, calibration version, …
result.cv?.prevalence.rmse; // calibration on the prevalence scale
```

### 2. Fit the variogram once, then build a re-usable model

For repeated work (multiple grids, simulations, polygon roll-ups) it is
cheaper to fit once and reuse:

```ts
import {
  BinomialKriging,
  fitVariogram,
  computeEmpiricalVariogram,
} from "kriging-rs-wasm";

// Empirical variogram on EB-smoothed logits is the recommended starting point;
// fitVariogram does the smoothing internally for binomial inputs.
const fitted = fitVariogram({
  sampleLats: lats, sampleLons: lons,
  successes, trials,
  variogramType: "exponential",
  estimator: "cressie-hawkins",
  prior: { alpha: 1, beta: 1 },
});
const model = BinomialKriging.fromFittedVariogramWithPrior({
  lats, lons, successes, trials,
  fittedVariogram: fitted,
  prior: { alpha: 1, beta: 1 },
});
```

### 3. Cross-validate to calibrate the variogram

Binomial CV reports residuals on **both** scales: the logit scale (matches the
underlying ordinary-kriging variance, calibratable via MSDR ≈ 1) and the
prevalence scale (intuitive, delta-method variance):

```ts
import { kFoldBinomial } from "kriging-rs-wasm";

const cv = kFoldBinomial({
  lats, lons, successes, trials,
  k: 5,
  variogram: fitted,
  prior: { alpha: 1, beta: 1 },
});
cv.summary.logit.rmse;       // posterior calibration on the logit scale
cv.summary.prevalence.rmse;  // and on prevalence
cv.summary.logit.msdr;       // ≈ 1 means the variogram nugget/sill is well-tuned
```

### 4. Sample an ensemble of plausible maps (SGS)

Sequential Gaussian Simulation gives you many independent plausible
prevalence rasters honoring the data and the variogram. Use the
`*Many*` variant — it amortizes the conditioning factorization across
realizations:

```ts
import { simulateBinomialGridEnsemble } from "kriging-rs-wasm";

const ensemble = simulateBinomialGridEnsemble({
  lats, lons, successes, trials,
  west: -1, south: 36, east: 1, north: 38,
  xCells: 100, yCells: 100,
  variogram: fitted,
  prior: { alpha: 1, beta: 1 },
  nRealizations: 200,
  baseSeed: 42n,           // realization k uses seed = baseSeed + k
});
// ensemble.{logit,prevalence}Samples are flat row-major Float64Array
// of length nRealizations * (xCells * yCells); ensemble.nTargets covers each.
```

For a one-shot ensemble + per-cell reduction (mean, variance, quantiles,
exceedance probabilities) use `simulateBinomialGridSummary` and pick the
scale you want quantiles on:

```ts
import { simulateBinomialGridSummary } from "kriging-rs-wasm";

const summary = simulateBinomialGridSummary({
  lats, lons, successes, trials,
  west: -1, south: 36, east: 1, north: 38,
  xCells: 100, yCells: 100,
  variogram: fitted,
  nRealizations: 200,
  baseSeed: 42n,
  quantiles: [0.025, 0.5, 0.975],     // 95% credible interval
  exceedanceThresholds: [0.10, 0.25],  // P(prevalence > 10%, > 25%)
  summarizeOn: "prevalence",
});
summary.meanPrevalence;       // [yCells][xCells]
summary.quantiles[1].values;  // median prevalence map
summary.exceedances[0].values; // P(prevalence > 0.10) per cell
```

### 5. Roll up to administrative polygons (population-weighted)

A regulator usually wants the posterior distribution of prevalence inside a
district, not a 10 000-cell raster. `aggregatePrevalenceByPolygon` reduces
the ensemble to per-polygon mean / variance / quantiles in a single WASM call:

```ts
import {
  aggregatePrevalenceByPolygon,
  polygonCellsFromMask,
} from "kriging-rs-wasm";

// Population raster (or a binary indicator mask) shaped [yCells][xCells].
const district = polygonCellsFromMask({
  xCells: ensemble.xCells, yCells: ensemble.yCells,
  mask: populationRaster,   // truthy values become weights
  id: "district-A",
});
const [report] = aggregatePrevalenceByPolygon({
  ensemble,
  polygons: [district],
  quantiles: [0.025, 0.5, 0.975],
});
report.mean;        // population-weighted posterior mean prevalence
report.variance;    // posterior variance (n - 1 denominator)
report.quantiles;   // [{ probability: 0.025, value }, …]
```

### 6. Projected coordinates and 2-D anisotropy

For data in a local equal-area projection where directional range matters,
use the projected variants — same workflow, planar distances, 2-D anisotropy:

```ts
import {
  BinomialProjectedKriging,
  kFoldBinomialProjected,
  conditionalSimulateManyBinomialProjected,
} from "kriging-rs-wasm";

const model = new BinomialProjectedKriging({
  xs, ys, successes, trials,
  variogram: { variogramType: "exponential", nugget: 0.05, sill: 1, range: 1500 },
  majorAngleDeg: 30,    // azimuth of long axis in degrees CCW from +x
  rangeRatio: 0.5,       // 1.0 = isotropic
});
const cv = kFoldBinomialProjected({ /* same shape; reports both scales */ });
const ensemble = conditionalSimulateManyBinomialProjected({
  conditioningXs: xs, conditioningYs: ys, successes, trials,
  targetXs, targetYs,
  variogram: { variogramType: "exponential", nugget: 0.05, sill: 1, range: 1500 },
  majorAngleDeg: 30, rangeRatio: 0.5,
  nRealizations: 200,
});
```

### 7. Date-aware space-time slices

Space-time binomial kriging works on a numeric `times` axis, but most callers
think in calendar dates. The `*AtDate` helpers convert via `timesFromDates`
under the hood — pass the `timeUnit` and `epoch` you used when building the
model:

```ts
import {
  SpaceTimeBinomialKriging,
  fitSpaceTimeVariogram,
  timesFromDates,
  simulateBinomialSpaceTimeGridSummaryAtDate,
} from "kriging-rs-wasm";

const epoch = new Date(Date.UTC(2024, 0, 1));
const times = timesFromDates(observationDates, "days", epoch);
const fit = fitSpaceTimeVariogram({
  lats, lons, times, values: logits, // or use the binomial CV-driven workflow
  family: "separable",
  spatialModel: "exponential",
  temporalModel: "exponential",
  nSpatialBins: 12, nTemporalBins: 8,
});
const model = SpaceTimeBinomialKriging.fromFitted({
  lats, lons, times, successes, trials,
  fittedVariogram: fit.fit,
});

// Predict an entire grid at the most recent reporting date.
const grid = model.predictGridAtDate({
  west: -1, south: 36, east: 1, north: 38,
  xCells: 100, yCells: 100,
  date: new Date("2024-09-01"),
  timeUnit: "days", epoch,
});

// Or simulate an ensemble at that date and pull credible intervals + exceedances:
const summary = simulateBinomialSpaceTimeGridSummaryAtDate({
  west: -1, south: 36, east: 1, north: 38,
  xCells: 100, yCells: 100,
  lats, lons, dates: observationDates,
  successes, trials,
  variogram: fit.fit,
  date: new Date("2024-09-01"),
  timeUnit: "days", epoch,
  nRealizations: 200,
  baseSeed: 7n,
  quantiles: [0.025, 0.5, 0.975],
  exceedanceThresholds: [0.10],
});
```

### Tips

- **Reuse the fitted variogram** across CV, simulation, and grids — it's the
  most expensive step.
- **Prefer `simulateBinomialGridEnsemble` over per-cell loops** when you need
  any aggregation (district means, exceedance maps, …); the row-major buffers
  feed straight into `ensembleMean` / `ensembleVariance` /
  `ensembleQuantiles` / `ensembleExceedanceProbability`.
- **`BinomialCvSummary.logit.msdr ≈ 1`** is the cleanest signal that the
  variogram is well calibrated; values far from 1 suggest tuning the nugget.
- **Default binomial prior is Beta(1, 1)** (uniform on prevalence, matching the
  Rust crate). Pass explicit `prior` / `priorAlpha`+`priorBeta` when you want
  Jeffreys `Beta(½, ½)` or another shrinkage choice.

## Error handling

Constructors (`OrdinaryKriging`, `SimpleKriging`, `UniversalKriging`, `ProjectedKriging`, `BinomialKriging`, `BinomialKriging.newWithPrior`, `BinomialKriging.fromPrecomputedLogits`, the `SpaceTime*Kriging` classes), `fitVariogram`, `fitSpaceTimeVariogram`, and the top-level helpers (`computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`, `computeEmpiricalSpaceTimeVariogram`, `leaveOneOut`, `kFold`, `leaveOneOutSimple`, `kFoldSimple`, `leaveOneOutUniversal`, `kFoldUniversal`, `leaveOneOutProjected`, `kFoldProjected`, `leaveOneOutBinomial`, `kFoldBinomial`, `leaveOneOutSpaceTime`, `kFoldSpaceTime`, `leaveOneOutSpaceTimeSimple`, `kFoldSpaceTimeSimple`, `leaveOneOutSpaceTimeUniversal`, `kFoldSpaceTimeUniversal`, `leaveOneOutSpaceTimeBinomial`, `kFoldSpaceTimeBinomial`, `conditionalSimulate`, `conditionalSimulateSimple`, `conditionalSimulateUniversal`, `conditionalSimulateProjected`, `conditionalSimulateBinomial`, `conditionalSimulateSpaceTime`, `conditionalSimulateSpaceTimeSimple`, `conditionalSimulateSpaceTimeUniversal`, `conditionalSimulateSpaceTimeBinomial`, `evaluateNestedVariogram`) throw on invalid inputs or model build failure (e.g. singular covariance). Errors are rethrown as `KrigingError` with the underlying cause attached as `cause`. For UI-friendly messages, use the optional **`code`** property (when present), which is one of: `not_loaded`, `model_freed`, `mismatched_arrays`, `invalid_variogram`, `invalid_bins`, `singular_covariance`, `too_few_points`, `unknown_variogram`, `unknown_family`, `unknown_trend`, `unknown_estimator`, `invalid_input`, `backend_unavailable`, `internal_error`. Not every error has a code. `backend_unavailable` specifically indicates that the WASM package was built without the export in question (e.g. without the `gpu` feature), so the API can't fulfill the call in the current build.

Typical causes:

- Mismatched array lengths (lats, lons, values or successes/trials)
- Invalid coordinates or variogram parameters
- Using the API before calling and awaiting `init()`

```ts
import { KrigingError } from "kriging-rs-wasm";
try {
  const model = new OrdinaryKriging({
    lats,
    lons,
    values,
    variogram: { variogramType: "gaussian", nugget: 0.01, sill: 1, range: 100 },
  });
} catch (e) {
  if (e instanceof KrigingError) {
    if (e.code === "singular_covariance") {
      showMessage("Model failed: singular covariance matrix.");
    } else if (e.code === "mismatched_arrays") {
      showMessage("Check that lats, lons, and values have the same length.");
    } else {
      console.error(e.message, e.cause);
    }
  }
  throw e;
}
```

## Publishing

From `npm/kriging-rs-wasm`, run `npm run verify`, then `npm publish` (dry-run: `npm publish --dry-run`). The package publishes the `dist/`, `pkg/`, and `README.md` listed in `files`. Follow semver for releases.

## Notes

- Call and await `init(...)` once before invoking model constructors or variogram-fitting APIs. You can pass pre-fetched WASM bytes: `await init(wasmArrayBuffer)`.
- Coordinates are in degrees (latitude, longitude); distances use Haversine (great-circle).
- For GPU-enabled exports, build with `npm run build:wasm:gpu`.
- **Resource management:** When a model is no longer needed, call `model.free()` to release WASM-held memory. This is optional but recommended in long-lived applications. **`free()` is safe to call multiple times** (subsequent calls are no-ops), so you can call it in a `finally` block without worrying about double-free.
