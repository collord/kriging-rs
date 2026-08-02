# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — 3-D fork (`3d-fork-v1` branch)

- **3-D core types**: `Coord3D`, `Orientation`, `PlanarDataset3D`,
  `Anisotropy3D` (rotation + diagonal stretch), `Grid3D`. GSLib I/O via
  `from_gslib` / `to_gslib`; all GSLib conventions (clockwise-from-north
  azimuth, dip-positive-down, truncated-π DEG2RAD) isolated in
  `interop::gslib_anisotropy` with `setrot.f` parity to 1e-6.
- **3-D variograms**: `compute_empirical_variogram_3d` (omnidirectional,
  parallelized) and `compute_directional_variogram_3d` with
  `DirectionFilter3D` (GSLib `gamv`-compatible cone + bandwidth filter,
  lag-centred binning, bitwise pair-set parity with `gamv`).
- **3-D variogram fitting**: `fit_spherical_3d_joint` (pairs-weighted
  Nelder–Mead over shared `(nugget, sill)` and per-axis ranges),
  `fit_spherical_3d_two_stage` (vertical-anchored, sill clamped at
  sample variance), `fit_spherical_3d_with_fixed_nugget` (refits sill
  and ranges with nugget held). Returns `Spherical3DJointFit`.
- **3-D kriging**: `OrdinaryKrigingModel3D`, `SimpleKrigingModel3D`,
  `UniversalKrigingModel3D` (linear trend basis `[1, x, y, z]`),
  sharing a robust solver (`solve_*_3d`) with automatic nugget
  inflation, condition-number reporting, and an optional
  `Neighborhood3D` filter.
- **3-D SGS**: `gaussian_simulation_3d_stream`,
  `gaussian_simulation_3d_stream_parallel`,
  `gaussian_simulation_3d_stream_with`. Streaming realization API;
  engine retains nothing between calls. `SgsOutputSpace::ScoreSpace`
  skips NST back-transform. `NormalScoreTransform` provides empirical-
  CDF NST for pre/post-processing.
- **3-D cross-validation**: `cv_3d::leave_one_out_ordinary_3d` /
  `_simple_3d` / `_universal_3d_linear`, reusing upstream
  `CvResidual` / `CvSummary`.
- **Anisotropy-aware kd-trees**: `KdTree3D` and `MutableKdTree3D` over
  `kiddo`, with coordinates pre-transformed into the anisotropy
  ellipsoid's principal frame. `MutableKdTree3D::add` applies a
  deterministic per-id ulp-scale jitter to avoid kiddo bucket-overflow
  on dense colinear inputs.
- **Rayon parallelism** on hot 3-D paths (empirical variogram,
  directional variogram, SGS across realizations).
- **WASM bindings**: `WasmOrdinaryKriging3D`, `WasmSimpleKriging3D`,
  `WasmUniversalKriging3D`, `gaussianSimulation3D`,
  `computeEmpiricalVariogram3D`, `computeDirectionalVariogram3D`,
  `fitSpherical3D*`. Mirrors upstream's 2-D WASM API shape.
- Native perf benchmark: `examples/sgs_oilsands_bench.rs`.

### Added — variogram forms & cokriging

- **Confluent Hypergeometric variogram kernel.** New `VariogramType::ConfluentHypergeometric`
  / `VariogramModel::ConfluentHypergeometric { nugget, sill, range, nu, alpha }`, a two-shape
  family with Matérn-like smoothness `nu > 0` and a separate tail-decay `alpha > 0` giving
  polynomial (long-range) tails; it converges to Matérn as `alpha → ∞`. The correlation uses
  the confluent hypergeometric function of the second kind `U(a, b, z)`, evaluated by
  double-exponential quadrature in log space (no new dependency; stable for large `alpha`).
  Reference: Ma & Bhadra (2023), *Beyond Matérn: On a Class of Interpretable Confluent
  Hypergeometric Covariance Functions*, JASA 118(543):2045–2058,
  <https://doi.org/10.1080/01621459.2022.2027775>.
- **Two-shape constructor** `VariogramModel::new_with_shapes(nugget, sill, range, type, shape1,
  shape2)` and accessor `VariogramModel::shape2()`. `new_with_shape` now delegates to it, and
  `fit_variogram` sweeps a small `(nu, alpha)` grid for the CH family.
- **WASM / TypeScript:** the `"confluenthypergeometric"` variogram type and an optional second
  shape parameter (`shape2`, the CH tail-decay `alpha`) are threaded through the sequential
  Gaussian simulation surface (geographic, projected, binomial, multi-realization, and
  space-time variants) and surfaced on fitted-variogram results.

### Fixed

- **Cubic variogram coefficients.** The cubic model used `7x² − 8.5x³ + 3.5x⁵ − 0.5x⁷`, which
  overshoots the sill near the range (reaching ≈1.5·partial_sill just inside it) and is not
  monotone. Corrected to the standard `7x² − 8.75x³ + 3.5x⁵ − 0.75x⁷` (Chilès & Delfiner 2012),
  which meets the sill continuously at the range. Fixed in the engine and the `www` demo.

## [0.4.0] - 2026-04-26

### Added

- **Calibrated binomial kriging (default):** per-site logit *observation* variance
  (Empirical-Bayes `Beta(α,β)` + delta method) on the ordinary-kriging diagonal, with
  automatic inflation retries and always-returned [`BinomialBuildNotes`]
  (`calibrationVersion`, `logitInflation`, `nBuildAttempts`, prior, dropped zero-trial
  indices). The geographic fit type is `BinomialFit` (`BinomialCalibratedResult<BinomialKrigingModel>`);
  use [`BinomialKrigingModel::new_with_config`] for full control. (The older separate
  “hetero-only” constructor is removed; `new` / `new_with_prior` are heteroskedastic
  by default with `HeteroskedasticBinomialConfig::default()`.)
- **WASM:** `WasmBinomialKriging`, `WasmBinomialProjectedKriging`, and
  `WasmSpaceTimeBinomialKriging` store build notes and expose `getBuildNotes()`. Count
  inputs with `trials==0` are dropped before fit (at least two retained sites required).
- **TypeScript:** `interpolateBinomialToGrid` includes `buildNotes` on
  `InterpolateBinomialToGridResult`. `getBuildNotes()` is available on
  `BinomialKriging`, `BinomialProjectedKriging`, and `SpaceTimeBinomialKriging`.
- **Empirical variogram (binomial):** `compute_empirical_variogram_binomial_calibrated`
  (pair-noise aware classical estimator) for the calibrated default.
- `OrdinaryKrigingModel::new_with_extra_diagonal` for arbitrary (homogeneous or) **site-specific**
  non-spatial noise on the covariance diagonal.
- Browser-representative workload and **prediction-only** benchmark: `benches/BROWSER_BENCHMARKS.md`
  and Criterion `bench_binomial_browser_representative` (`400` stations, `200×200` target grid;
  optional `gpu-blocking` sub-bench for WebGPU class RHS).

### Changed (breaking / migration from 0.3.x)

- **Default binomial prior is `Beta(1, 1)`** in Rust and when the TS/JS `prior` is
  omitted; use explicit `0.5/0.5` for Jeffreys.
- **Rust return types:** `BinomialKrigingModel::new` / `new_with_prior` /
  `new_with_config` (and the analogous **projected** and **space–time** binomial
  builders, including some `from_precomputed_logits*`) return
  `BinomialCalibratedResult<…>` (alias `BinomialFit` in geographic 2D). Use
  `.model`, `.into_model()`, or `Deref` to `&InnerModel` for `predict` / `predict_batch`.
- **WASM/TS** consumers must **rebuild** the published `pkg` for `getBuildNotes` and
  the `trials==0` / minimum-site validation behavior.
- **Binomial SGS** (`conditional_simulate*`) uses heteroskedastic (logit observation
  variance) conditioning in the per-target sequential fits for closer alignment with
  calibrated binomial prediction.

## [0.3.0] - 2026-04-16

0.3.0 is a large feature release. Highlights:

- A new **spatio-temporal kriging** module that extends the 2-D surface with a time axis.
- **Cross-validation** (leave-one-out and K-fold) and **conditional simulation** (Sequential Gaussian Simulation) — both new to the project — shipped for every kriging variant in both 2-D and space-time.
- A focused **disease prevalence mapping cookbook**: tightened one-shot binomial interpolation, fast multi-realization SGS, generic ensemble aggregators, grid-shaped binomial simulation, polygon roll-ups, projected binomial kriging, and date-aware space-time helpers.
- The remaining `kriging-rs-wasm` feature parity gaps from 0.2.x are filled (`SimpleKriging`, `UniversalKriging`, `ProjectedKriging`, neighborhoods, more variogram families).
- A round of TypeScript ergonomics: discriminated-union variograms, `fromFitted`, `Symbol.dispose`, date helpers, fixed-time grids, and multi-realization SGS.

### Added

#### Disease prevalence mapping cookbook (TypeScript + Rust)

A focused workflow for the project's primary use case: building, validating,
sampling, and reporting prevalence surfaces from binomial count data.

- **Tightened one-shot binomial interpolation.** `interpolateBinomialToGrid`
  now fits the variogram on the same EB-smoothed logits used by
  `BinomialKriging`, accepts a `prior` and an `estimator` (`"classical" |
  "cressie-hawkins"`), exposes the resulting `fittedVariogram` on the result,
  and can run leave-one-out / k-fold CV in the same call via
  `withCv: true | "loo" | { k }`. The new `BinomialCvSummary` (both logit and
  prevalence scales) is reported alongside the grid.
- **Multi-realization SGS — fast path.** New Rust entry points
  `conditional_simulate_many{,_spacetime}{,_binomial}` amortize the
  conditioning factorization across realizations and ship behind WASM bridges
  (`conditionalSimulateMany{,Binomial}` and the ST variants now do a single
  JS↔WASM crossing per ensemble).
- **Generic ensemble aggregators.** Pure-TS `ensembleMean`, `ensembleVariance`,
  `ensembleQuantiles`, and `ensembleExceedanceProbability` operate over the
  flat row-major buffers produced by the `*Many*` simulators.
- **Grid-shaped binomial simulation.** `simulateBinomialGrid`,
  `simulateBinomialGridEnsemble`, and `simulateBinomialGridSummary` build
  conditioning + target geometry from a `GeoGridBounds` and return either
  nested `[yCells][xCells]` arrays (single realization), a flat ensemble
  buffer (many realizations), or per-cell mean / variance / quantile /
  exceedance maps (one-shot summary).
- **Polygon aggregation over ensembles.** New Rust `aggregate` module
  (`polygon_weighted_summary`, `polygon_weighted_summaries_batch`) reduces an
  ensemble buffer to per-polygon mean / variance / quantiles using
  `(indices, weights)` cell lists; bridged through WASM as
  `aggregatePolygonsOverEnsemble`. The TS `aggregatePrevalenceByPolygon` and
  `polygonCellsFromMask` helpers wrap this for population-weighted
  district-level reporting.
- **Projected binomial kriging.** New Rust `BinomialProjectedKrigingModel`
  with `ProjectedBinomialObservation` (Beta priors and pre-computed logits
  supported), CV (`leave_one_out_binomial_projected`,
  `k_fold_binomial_projected`), SGS
  (`conditional_simulate_binomial_projected`,
  `conditional_simulate_many_binomial_projected`), and a WASM bridge
  (`WasmBinomialProjectedKriging`). The TS layer ships
  `BinomialProjectedKriging` plus `leaveOneOutBinomialProjected`,
  `kFoldBinomialProjected`, `conditionalSimulateBinomialProjected`, and
  `conditionalSimulateManyBinomialProjected` for projected (planar, optionally
  anisotropic) prevalence mapping.
- **Date-aware space-time helpers.** `SpaceTimeBinomialKriging` gains
  `predictAtDate`, `predictBatchAtDates`, `predictBatchArraysAtDates`, and
  `predictGridAtDate` (date-axis configured per call via
  `DateAxisOptions = { timeUnit?, epoch? }`). New free functions
  `simulateBinomialSpaceTimeGrid{,Ensemble,Summary}` and their `*AtDate`
  counterparts mirror the spatial grid family for ST binomial conditioning.

#### Spatio-temporal kriging (new `spacetime` module)

- **Coordinates & metrics.** `SpaceTimeCoord<C>` and `SpaceTimeDataset<C>`, generic over any spatial-coordinate type that implements the new `SpatialMetric` trait. Two implementations ship out of the box: `GeoMetric` (Haversine on `GeoCoord`) and `ProjectedMetric` (Euclidean on `ProjectedCoord` with optional `Anisotropy2D`). All re-exported from `spacetime`.
- **Variogram families.** `SpaceTimeVariogram::new_separable(spatial, temporal)` (normalized product) and `SpaceTimeVariogram::new_product_sum(spatial, temporal, k1, k2, k3)` (`k₁·C_s·C_t + k₂·C_s + k₃·C_t`), with admissibility checks (nonnegative `k_i`, `k₁+k₂+k₃ > 0`, power-law marginals rejected).
- **Empirical & fitted variograms.** `compute_empirical_spacetime_variogram` computes a 2-D spatio-temporal variogram (Matheron and Cressie–Hawkins estimators); `spatial_marginal` / `temporal_marginal` project it to 1-D slices. `fit_spacetime_variogram` fits the two marginals independently and, for product-sum, solves a non-negative least-squares problem for `(k1, k2, k3)`. Supporting types: `EmpiricalSpaceTimeVariogram`, `SpaceTimeVariogramConfig`, `SpaceTimeFitConfig`, `SpaceTimeFitResult`.
- **Kriging models** (all generic over `SpatialMetric`): `SpaceTimeOrdinaryKrigingModel<M>`, `SpaceTimeSimpleKrigingModel<M>`, `SpaceTimeUniversalKrigingModel<M>` (trends: `Constant`, `LinearInTime`, `QuadraticInTime`, `LinearInSpace`, `LinearInSpaceAndTime`, `QuadraticInSpaceAndTime`), and `SpaceTimeBinomialKrigingModel<M>` (Beta priors via `SpaceTimeBinomialObservation`; pre-computed logits supported).
- **WASM / TypeScript bindings.** `WasmSpaceTimeOrdinaryKriging`, `WasmSpaceTimeSimpleKriging`, `WasmSpaceTimeUniversalKriging`, `WasmSpaceTimeBinomialKriging`, `WasmSpaceTimeOrdinaryProjectedKriging`, plus `wasmComputeEmpiricalSpaceTimeVariogram` / `wasmFitSpaceTimeVariogram`. TypeScript wrappers under `src/spacetime/`: `SpaceTime{Ordinary,Simple,Universal,Binomial,ProjectedOrdinary}Kriging`, `computeEmpiricalSpaceTimeVariogram`, `fitSpaceTimeVariogram`. New `KrigingErrorCode` values: `"unknown_family"`, `"unknown_trend"`, `"unknown_estimator"`.

#### Cross-validation (new in 0.3.0, every variant)

Cross-validation is brand new in this release: leave-one-out and K-fold helpers
ship for every kriging variant, both 2-D and space-time, from day one.

- **Rust (2-D):** `leave_one_out` / `k_fold` (ordinary) plus `_simple`, `_universal`, `_projected`, `_binomial`. Fold iteration is factored into shared `for_each_loo_fold` / `for_each_k_fold` helpers, so adding new variants is trivial.
- **Rust (space-time):** `leave_one_out_spacetime` / `k_fold_spacetime` (ordinary), plus `_simple`, `_universal`, `_binomial` — generic over `SpatialMetric` (ordinary / simple / binomial) or `SpatialBasis` (universal).
- **TypeScript:** `leaveOneOut` / `kFold` (ordinary) plus `Simple`, `Universal`, `Projected`, `Binomial`, `SpaceTime`, `SpaceTimeSimple`, `SpaceTimeUniversal`, `SpaceTimeBinomial` suffixes for both verbs.
- **Binomial CV reports both scales.** New `BinomialCvResidual` / `BinomialCvSummary` / `BinomialCvResult` types carry per-station observed/predicted values and variances on **both** the logit scale (directly comparable to continuous kriging; calibratable via MSDR) and the prevalence scale (delta-method variance). Stations with `trials == 0` retain their index with `NaN` observed fields and are automatically skipped in summary aggregation.

#### Conditional simulation (Sequential Gaussian Simulation, new in 0.3.0, every variant)

Also brand new: SGS helpers ship for every kriging variant, both 2-D and
space-time.

- **Rust (2-D):** `conditional_simulate` (ordinary) plus `_simple`, `_universal`, `_projected`, `_binomial`. Target-order validation is factored into a shared `resolve_target_order` helper.
- **Rust (space-time):** `conditional_simulate_spacetime` (ordinary), plus `_simple`, `_universal`, `_binomial`.
- **TypeScript:** `conditionalSimulate` (ordinary) plus `Simple`, `Universal`, `Projected`, `Binomial`, `SpaceTime`, `SpaceTimeSimple`, `SpaceTimeUniversal`, `SpaceTimeBinomial` suffixes. All are deterministic for a given `seed` and accept an optional `targetOrder` override.
- **Binomial simulation reports both scales.** New `BinomialSimulationResult` (Rust and TS) carries `logit_samples` / `logitSamples` (unbounded) and `prevalence_samples` / `prevalenceSamples` (in `(0, 1)`, by construction `logistic(logit_samples)`). Simulation runs on the logit scale; stations with `trials == 0` are dropped from the initial conditioning pool. Beta prior supported via `prior_alpha` / `prior_beta` (Rust) or `prior?: BinomialPriorParams` (TypeScript).

#### WASM / npm feature parity

`kriging-rs-wasm` now mirrors the full Rust surface — these bullets close the
2-D gaps left over from 0.2.x (ordinary + binomial only). The new CV and SGS
helpers are listed in their own sections above.

- New classes: `SimpleKriging` (known mean), `UniversalKriging` (trends `"constant" | "linear" | "quadratic"`), and `ProjectedKriging` (planar `(x, y)` with 2-D anisotropy via `majorAngleDeg`, `rangeRatio`).
- `OrdinaryKriging.setNeighborhood({ maxNeighbors?, maxRadius? })` and `neighborhood()` for search-neighborhood configuration.
- `BinomialKriging.fromPrecomputedLogits(...)` factory that bypasses empirical-Bayes shrinkage.
- Top-level helpers: `computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`, `evaluateNestedVariogram`.
- `fitVariogram` and `computeEmpiricalVariogram` accept `estimator: "classical" | "cressie-hawkins"`.
- `VariogramTypeName` gains `"power"` and `"holeeffect"`.

#### TypeScript DX polish (`kriging-rs-wasm`)

Higher-level ergonomics layered on top of the WASM bindings.

- **Discriminated union variograms.** `SpaceTimeVariogramParams` and `FittedSpaceTimeVariogram` are now discriminated on `family`: `"separable"` (no product-sum coefficients) or `"productSum"` (requires `k1`, `k2`, `k3`). TypeScript enforces the coefficient shape; passing snake_case is a type error.
- **`fromFitted(...)`** static on every space-time kriging class — pass a `fitSpaceTimeVariogram` result directly as `fittedVariogram`, no destructuring required.
- **`[Symbol.dispose]()`** on every 2-D and space-time kriging class, enabling `using model = new OrdinaryKriging(...)` (ES2023 explicit resource management) to release WASM memory automatically.
- **Unified binomial prior.** CV and SGS options take a single `prior?: BinomialPriorParams` object (both `alpha` and `beta` must be set together). Validation is centralized in the new `resolveBinomialPrior` helper.
- **Date helpers.** `timesFromDates(dates, unit?, epoch?)` and `datesFromTimes(times, unit?, epoch?)` with `TimeUnit = "ms" | "s" | "minutes" | "hours" | "days"` for converting between JS `Date` and the scalar time axis used by ST models.
- **`predictGridAtTime(options)`** on `SpaceTimeOrdinaryKriging` and `SpaceTimeBinomialKriging` — predicts a rectangular lat/lon grid at a fixed time slice and returns 2-D arrays shaped `[yCells][xCells]`.
- **Multi-realization SGS.** `conditionalSimulateMany({ ..., nRealizations, baseSeed? })` and `conditionalSimulateManySpaceTime(...)` draw N independent realizations with seeds `baseSeed + k` and return a flat row-major `Float64Array(nRealizations * nTargets)`.

#### Rust ergonomics

- `OrdinaryKrigingModel::set_neighborhood` — in-place counterpart of `with_neighborhood`, for FFI-friendly updates that don't consume `self`.

### Changed

- **No more silent fallbacks for missing WASM symbols.** Every TypeScript wrapper for an unconditional WASM export (`leaveOneOut*`, `kFold*`, `conditionalSimulate*`, `conditionalSimulateMany*`, `computeEmpiricalVariogram`, `computeDirectionalEmpiricalVariogram`, `evaluateNestedVariogram`, `aggregatePolygonsOverEnsemble`) now calls the WASM export directly without a `typeof === "function"` guard or pure-JS fallback. The `RawModule` shape in `internal/wasm-shapes.ts` declares these as required (only the `gpu`-feature-gated `webgpuAvailable` and instance-level `predictBatchGpu` / `predictBatchGpuOrCpu` remain optional, since they really may be absent from a GPU-less build). This eliminates a class of dead code and ensures any genuine packaging mistake fails loudly at first call rather than silently falling back to a slower path.
- **CV summary statistics now propagate `NaN`.** The internal `requireFiniteOrZero` helper has been replaced with `requireFiniteOrNaN` (in `internal/convert.ts`), and is now used for `meanError`, `rmse`, and `msdr` on `CvSummary` / `BinomialCvSummary`, plus `residuals` on `FittedSpaceTimeVariogram`. Previously these were silently coerced to `0` when the underlying Rust statistic was `NaN` (for example when zero residuals were evaluated), which masked "no data to summarize" as "perfectly calibrated". The honest `NaN` is now returned; non-numeric payloads still throw.
- **`unpackBinomialManyPayload` no longer carries fallback `nRealizations` / `nTargets`.** The Rust `binomial_many_to_js` serializer always populates those fields, so the JS-side `?? fallback` defaults were defensive against our own code. They have been removed; both fields are now resolved with `requireNumber`, surfacing any inconsistency in the WASM payload immediately.

### Documentation

- **Disease prevalence mapping cookbook** in `npm/kriging-rs-wasm/README.md`
  walking through fit → validate → simulate → aggregate end-to-end.
- **`examples/prevalence_mapping.rs`** — Rust runnable example showing the
  same workflow against the native crate.

## [0.2.3] - 2026-04-06

### Fixed

- **npm package (kriging-rs-wasm)** – Load the wasm-pack glue with a static `import` so bundlers (e.g. Vite) include `pkg/kriging_rs.js` and the `.wasm` file in the app output. Previously a dynamic `import("../pkg/kriging_rs.js")` was not analyzed at build time, so production requests to `/pkg/kriging_rs.js` failed unless consumers copied `pkg/` into `dist/`.
- **npm package (kriging-rs-wasm)** – `typecheck:contracts` and `verify` run `build:wasm` before TypeScript so `pkg/kriging_rs.js` exists on fresh checkouts (fixes CI TS2307 after the static import change).

## [0.2.2] - 2025-03-15

### Fixed

- **npm package (kriging-rs-wasm)** – The published tarball again correctly includes the `pkg/` directory (`kriging_rs.js`, `kriging_rs_bg.wasm`, etc.). Root `.gitignore` was narrowed to `/pkg/` and `/dist/` so npm pack no longer excludes the package’s build output (monorepo ignore behavior). The build script now removes wasm-pack’s generated `pkg/.gitignore` (which ignored all files) so npm pack includes the WASM artifacts.

## [0.2.1] - 2025-03-15

### Fixed

- **npm package (kriging-rs-wasm)** – The published tarball now includes the `pkg/` directory (wasm-pack output: `kriging_rs.js`, `kriging_rs_bg.wasm`, etc.). Previously `build:wasm` used `--out-dir npm/kriging-rs-wasm/pkg`, which from the package directory resolved to a nested path, so the top-level `pkg/` listed in `files` was never created. Build scripts now use `--out-dir pkg`. `prepublishOnly` also verifies that `pkg/` exists and contains the expected files before publishing.

## [0.2.0] - 2025-03-15

### Added

- **npm package (kriging-rs-wasm)**
  - `predictGrid(options)` on `OrdinaryKriging` and `BinomialKriging`: rectangular grid prediction from bounds and cell counts; returns 2D value/variance (and prevalence/logit) grids. Types: `PredictGridOptions`, `OrdinaryGridOutput`, `BinomialGridOutput`.
  - One-shot helpers: `interpolateOrdinaryToGrid(options)` and `interpolateBinomialToGrid(options)` (fit → build model → predict grid → free). Types: `InterpolateOrdinaryToGridOptions`, `InterpolateBinomialToGridOptions`.
  - `KrigingError.code` and `KrigingErrorCode` for UI-friendly error handling.

### Changed

- **npm package (kriging-rs-wasm)**
  - `init()` now returns `Promise<void>` (was `Promise<unknown>`) so callers can use it directly without a wrapper.
  - Optional `nuggetOverride` on `OrdinaryKrigingFromFittedOptions`, `BinomialKrigingFromFittedVariogramOptions`, and `BinomialKrigingFromFittedVariogramWithPriorOptions` to override the fitted variogram nugget when building the model.
  - `model.free()` is idempotent: safe to call multiple times; subsequent calls are no-ops. The TypeScript wrapper clears its reference after the first call so use-after-free throws a clear error. Documented in README and JSDoc.

## [0.1.0] - 2025-03-15

### Added

- **Rust crate (kriging-rs)**
  - Ordinary kriging: `OrdinaryKrigingModel`, `Prediction`, single-point and batch prediction.
  - Binomial kriging: `BinomialKrigingModel`, `BinomialObservation`, `BinomialPrediction`, `BinomialPrior`; optional prior; single-point and batch prediction.
  - Empirical variogram: `compute_empirical_variogram`, `EmpiricalVariogram`, `VariogramConfig`, `PositiveReal`.
  - Variogram fitting: `fit_variogram`, `FitResult`.
  - Parametric variogram models: `VariogramModel`, `VariogramType` — Spherical, Exponential, Gaussian, Cubic, Stable, Matérn (Stable and Matérn with optional shape parameter).
  - `GeoCoord` (lat/lon with validation), Haversine distance and distance matrix; `GeoDataset` for coordinates and values.
  - `Real` type alias (`f32`), `KrigingError`; utilities: `Probability`, `clamp_probability`, `logistic`, `logit`, `logit_clamped`.
  - Optional `wasm` feature for browser WASM bindings (see npm package).
  - Optional `gpu` and `gpu-blocking` features for WebGPU-based batch prediction and RHS covariance building via `wgpu`; `GpuBackend`, `GpuSupport`, `detect_gpu_support`, `build_rhs_covariances_gpu`, and related APIs.
- **npm package (kriging-rs-wasm)**
  - TypeScript-first WASM bindings; `init()` required before use.
  - `OrdinaryKriging` and `BinomialKriging` (constructors, `predict`, `predict_batch`, `predict_batch_arrays`).
  - `fitVariogram` with configurable bins and variogram type (string or `VariogramType` enum).
  - Variogram types: spherical, exponential, gaussian, cubic, stable, matern (optional `shape` for stable and matern).
  - `KrigingError` (JS class with `cause`); `webgpuAvailable` when built with GPU support.
  - Batch and typed-array prediction APIs.

[0.4.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.4.0
[0.3.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.3.0
[0.2.3]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.3
[0.2.2]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.2
[0.2.1]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.1
[0.2.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.2.0
[0.1.0]: https://github.com/m-murphy/kriging-rs/releases/tag/v0.1.0
