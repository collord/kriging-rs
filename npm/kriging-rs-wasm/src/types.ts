/**
 * Public types and interfaces for kriging-rs-wasm. Purely type-level; no runtime
 * behavior lives in this module.
 *
 * @module
 */

/**
 * Supported variogram model type names (string form, e.g. in {@link FittedVariogram}).
 */
export type VariogramTypeName =
  | "spherical"
  | "exponential"
  | "gaussian"
  | "cubic"
  | "stable"
  | "matern"
  | "power"
  | "holeeffect"
  | "confluenthypergeometric";

/**
 * Empirical variogram estimator choice.
 *
 * - `"classical"` — Matheron's estimator (mean-squared differences; default).
 * - `"cressie-hawkins"` — robust fourth-root estimator, resistant to outliers.
 */
export type EmpiricalEstimator = "classical" | "cressie-hawkins";

/**
 * Universal kriging drift / trend basis.
 *
 * - `"constant"` — equivalent to ordinary kriging.
 * - `"linear"` — `[1, lat, lon]`.
 * - `"quadratic"` — `[1, lat, lon, lat², lat·lon, lon²]`.
 */
export type UniversalTrend = "constant" | "linear" | "quadratic";

/**
 * Input type for numeric coordinate or value arrays; accepts plain arrays or typed arrays.
 */
export type NumericArrayInput = number[] | ArrayLike<number>;

/**
 * Input type for integer counts (e.g. successes, trials); accepts plain arrays or typed arrays.
 * Values represent counts and should be non-negative integers (semantic only; no runtime check).
 */
export type IntegerArrayInput = number[] | ArrayLike<number>;

/**
 * Result of a single ordinary kriging prediction.
 * @property value - Interpolated value at the location
 * @property variance - Kriging variance (prediction uncertainty)
 */
export interface OrdinaryPrediction {
  value: number;
  variance: number;
}

/**
 * Result of a single binomial kriging prediction (prevalence surface).
 * @property prevalence - Estimated prevalence in [0, 1]
 * @property logitValue - Logit-scale value
 * @property variance - Kriging variance of the **logit**, not of prevalence. See
 *   {@link BinomialPrediction.prevalenceVariance} for a probability-scale approximation.
 * @property prevalenceVariance - Delta-method approximation of Var(prevalence),
 *   i.e. `[p(1-p)]^2 * variance`. Use this for approximate CIs on the probability scale.
 */
export interface BinomialPrediction {
  prevalence: number;
  logitValue: number;
  variance: number;
  prevalenceVariance: number;
}

/**
 * Batch ordinary kriging output as typed arrays (avoids per-point object allocation).
 * Use for large prediction grids.
 */
export interface OrdinaryBatchArrayOutput {
  values: Float64Array;
  variances: Float64Array;
}

/**
 * Batch binomial kriging output as typed arrays (avoids per-point object allocation).
 * Use for large prediction grids. `variances` is the kriging variance of the logit;
 * `prevalenceVariances` is the delta-method probability-scale approximation.
 */
export interface BinomialBatchArrayOutput {
  prevalences: Float64Array;
  logitValues: Float64Array;
  variances: Float64Array;
  prevalenceVariances: Float64Array;
}

/**
 * Options for grid prediction: rectangular bounds in degrees and number of cells.
 * Cell centers are computed in row-major order (first row = south, last row = north;
 * within a row, west to east). Result grids have shape [yCells][xCells] (row index = latitude).
 */
export interface PredictGridOptions {
  /** Western longitude in degrees. */
  west: number;
  /** Southern latitude in degrees. */
  south: number;
  /** Eastern longitude in degrees. */
  east: number;
  /** Northern latitude in degrees. */
  north: number;
  /** Number of cells in the x (longitude) direction. */
  xCells: number;
  /** Number of cells in the y (latitude) direction. */
  yCells: number;
}

/**
 * Ordinary kriging grid output: 2D arrays with shape [yCells][xCells].
 * values[j][i] and variances[j][i] correspond to row j (latitude), column i (longitude).
 */
export interface OrdinaryGridOutput {
  values: number[][];
  variances: number[][];
}

/**
 * Binomial kriging grid output: 2D arrays with shape [yCells][xCells]. `variances` is the
 * kriging variance on the logit scale; `prevalenceVariances` is the delta-method
 * probability-scale approximation `[p(1-p)]^2 * variance`.
 */
export interface BinomialGridOutput {
  prevalences: number[][];
  logitValues: number[][];
  variances: number[][];
  prevalenceVariances: number[][];
}

/**
 * Options for space-time grid prediction at a fixed time slice. Same spatial
 * bounds and cell counts as {@link PredictGridOptions}, plus the scalar `time`
 * at which every cell is evaluated.
 */
export interface PredictGridAtTimeOptions extends PredictGridOptions {
  /** Time value (same units used to build the model) applied to every grid cell. */
  time: number;
}

/**
 * Per-call configuration for date ↔ numeric-time conversion in date-aware
 * space-time helpers (`predictAtDate`, `predictGridAtDate`,
 * `simulateBinomialSpaceTimeGridAtDate`, …).
 *
 * The numeric `times` axis used by space-time kriging is unitless internally;
 * these options describe how JavaScript `Date` objects should be projected onto
 * that axis so callers don't have to manually compute milliseconds-since-epoch
 * offsets at every call site. Defaults match
 * {@link timesFromDates}: `timeUnit = "days"`, `epoch = new Date(0)`.
 */
export interface DateAxisOptions {
  /** Time granularity; same value used when the model was built. Defaults to `"days"`. */
  timeUnit?: import("./time.js").TimeUnit;
  /** Reference epoch; same value used when the model was built. Defaults to the Unix epoch. */
  epoch?: Date;
}

/**
 * Options for {@link SpaceTimeBinomialKriging.predictGridAtDate}: grid bounds
 * plus a `Date` at which every cell is evaluated and optional date-axis
 * configuration.
 */
export interface PredictGridAtDateOptions
  extends PredictGridOptions,
    DateAxisOptions {
  /** Date at which to evaluate every grid cell. */
  date: Date;
}

/**
 * Options for one-shot ordinary kriging: fit variogram from sample data, build model, predict on grid, then free.
 */
export interface InterpolateOrdinaryToGridOptions {
  /** Sample latitudes in degrees. */
  lats: NumericArrayInput;
  /** Sample longitudes in degrees. */
  lons: NumericArrayInput;
  /** Sample values (same length as lats/lons). */
  values: NumericArrayInput;
  /** Grid bounds and cell counts. */
  west: number;
  south: number;
  east: number;
  north: number;
  xCells: number;
  yCells: number;
  /** Variogram model type (e.g. "exponential"). */
  variogramType: VariogramTypeName | number;
  /** Optional number of bins for empirical variogram (default 12). */
  nBins?: number;
  /** Optional max distance for binning. */
  maxDistance?: number;
  /** Optional nugget override when building model from fitted variogram. */
  nuggetOverride?: number;
}

/**
 * Options for one-shot binomial kriging: fit variogram on the same EB-smoothed
 * logits the kriger consumes, build a model, predict on a rectangular grid, then
 * free the model.
 *
 * The variogram is fit on `logit((s + α) / (n + α + β))` per station, matching
 * the values the binomial kriger interpolates internally — so the fitted
 * `nugget`, `sill`, and `range` are calibrated for the same field. The default
 * prior is `Beta(1, 1)`; override with `prior` to match a custom shrinkage
 * choice.
 *
 * Pass `withCv: true` (or `withCv: { k: <folds> }`) to additionally run
 * leave-one-out (or k-fold) binomial cross-validation against the fitted
 * variogram and include the resulting {@link BinomialCvSummary} on the
 * returned object — useful for calibrating the variogram before publishing a
 * map.
 */
export interface InterpolateBinomialToGridOptions {
  /** Sample latitudes in degrees. */
  lats: NumericArrayInput;
  /** Sample longitudes in degrees. */
  lons: NumericArrayInput;
  /** Success counts (same length as lats/lons). */
  successes: IntegerArrayInput;
  /** Trial counts (same length as lats/lons). */
  trials: IntegerArrayInput;
  /** Grid bounds and cell counts. */
  west: number;
  south: number;
  east: number;
  north: number;
  xCells: number;
  yCells: number;
  /** Variogram model type (e.g. "exponential"). */
  variogramType: VariogramTypeName | number;
  /** Optional number of bins for empirical variogram (default 12). */
  nBins?: number;
  /** Optional max distance for binning (km). */
  maxDistance?: number;
  /** Optional nugget override when building model from fitted variogram. */
  nuggetOverride?: number;
  /** Optional Beta(alpha, beta) prior for binomial model (default Beta(1, 1)). */
  prior?: BinomialPriorParams;
  /**
   * Empirical estimator passed to {@link fitVariogram}. `"cressie-hawkins"` is
   * recommended for noisy count data; defaults to `"classical"` (Matheron).
   */
  estimator?: EmpiricalEstimator;
  /**
   * If set, also runs binomial cross-validation against the fitted variogram
   * and exposes the {@link BinomialCvSummary} on the returned object.
   *
   * - `true` (or `"loo"`): leave-one-out CV via {@link leaveOneOutBinomial}.
   * - `{ k: number }`: k-fold CV via {@link kFoldBinomial}.
   */
  withCv?: boolean | "loo" | { k: number };
}

/**
 * Result of {@link interpolateBinomialToGrid}. Extends {@link BinomialGridOutput}
 * with the fitted variogram used internally, {@link BinomialBuildNotes} from the
 * kriging build, and (when `withCv` is set) the binomial CV summary.
 */
export interface InterpolateBinomialToGridResult extends BinomialGridOutput {
  /** Variogram parameters fit from the EB-smoothed logits. */
  fittedVariogram: FittedVariogram;
  /** Diagnostics from the binomial model build (prior, dropped rows, inflation, …). */
  buildNotes: BinomialBuildNotes;
  /** Binomial CV summary; present iff the caller supplied `withCv`. */
  cv?: BinomialCvSummary;
}

/**
 * Common geographic-grid bounds and cell counts. Cell `(j, i)` covers the box
 * with center at `lat = south + (j + 0.5) * dy` and `lon = west + (i + 0.5) * dx`,
 * where `dx = (east - west) / xCells` and `dy = (north - south) / yCells`.
 *
 * `j` is the **row** index (latitude axis, 0 at the south edge) and `i` the
 * **column** index (longitude axis, 0 at the west edge), matching
 * {@link BinomialGridOutput} layout.
 */
export interface GeoGridBounds {
  west: number;
  south: number;
  east: number;
  north: number;
  /** Number of cells along the longitude axis (columns). */
  xCells: number;
  /** Number of cells along the latitude axis (rows). */
  yCells: number;
}

/**
 * Options for {@link simulateBinomialGrid}: draw a single binomial SGS realization
 * over a regular lat/lon grid and shape it as nested 2-D arrays `[yCells][xCells]`.
 *
 * `prior` defaults to `Beta(1, 1)`. The `seed` controls reproducibility.
 */
export interface SimulateBinomialGridOptions extends GeoGridBounds {
  /** Conditioning station latitudes (degrees). */
  lats: NumericArrayInput;
  /** Conditioning station longitudes (degrees). */
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  /** Variogram parameters governing the simulation kernel. */
  variogram: VariogramParams;
  prior?: BinomialPriorParams;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
}

/**
 * Single-realization grid output for {@link simulateBinomialGrid}.
 *
 * Each 2-D array has shape `[yCells][xCells]`. By construction
 * `prevalences[j][i] === logistic(logitSamples[j][i])` element-wise.
 */
export interface BinomialGridSimulation {
  logitSamples: number[][];
  prevalences: number[][];
}

/**
 * Options for {@link simulateBinomialGridEnsemble}: draw `nRealizations` independent
 * SGS realizations over the same grid. Returns the **flat row-major** ensemble
 * buffers (same layout as {@link BinomialSimulationManyResult}) so they can be fed
 * directly into the `ensemble*` aggregators without re-flattening.
 */
export interface SimulateBinomialGridEnsembleOptions extends GeoGridBounds {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: VariogramParams;
  prior?: BinomialPriorParams;
  /** Number of independent realizations (must be >= 1). */
  nRealizations: number;
  /** Base seed; the k-th realization uses `baseSeed + k`. */
  baseSeed?: number | bigint;
}

/**
 * Result of {@link simulateBinomialGridEnsemble}.
 *
 * `nTargets === xCells * yCells` (cells in row-major `(j, i)` order with `i`
 * fastest). Each typed array has length `nRealizations * nTargets`. Use
 * {@link reshapeGridRow} to pull row `k` back into a `[yCells][xCells]` matrix,
 * or feed the buffers straight into `ensembleMean`/`ensembleQuantiles`/etc.
 */
export interface BinomialGridEnsemble extends GeoGridBounds {
  nRealizations: number;
  /** Equals `xCells * yCells`. */
  nTargets: number;
  /** Row-major `[nRealizations * nTargets]` of simulated logits. */
  logitSamples: Float64Array;
  /** Row-major `[nRealizations * nTargets]` of simulated prevalences in `(0, 1)`. */
  prevalenceSamples: Float64Array;
}

/**
 * Options for {@link simulateBinomialGridSummary}: draw an ensemble and reduce it to
 * per-cell summary maps in one call. Quantiles are reported on the prevalence
 * scale by default (the natural reporting scale for disease maps); set
 * `summarizeOn: "logit"` to report on the logit scale instead.
 */
export interface SimulateBinomialGridSummaryOptions
  extends SimulateBinomialGridEnsembleOptions {
  /**
   * Probabilities in `[0, 1]` to report (e.g. `[0.025, 0.5, 0.975]` for a 95% CI
   * around the median). Pass an empty list to skip quantile computation.
   */
  quantiles?: ReadonlyArray<number>;
  /**
   * Optional thresholds (on the chosen `summarizeOn` scale, defaulting to
   * prevalence). For each `t` returns `P(value > t)` per cell.
   */
  exceedanceThresholds?: ReadonlyArray<number>;
  /**
   * Scale for `quantiles` and `exceedanceThresholds` summaries. Defaults to
   * `"prevalence"`. The `meanLogit`/`varianceLogit` and `meanPrevalence`
   * fields are always computed regardless.
   */
  summarizeOn?: "prevalence" | "logit";
}

/**
 * Options for {@link simulateBinomialSpaceTimeGrid}: draw a single binomial
 * space-time SGS realization over a regular lat/lon grid at a fixed `time`,
 * shaped as `[yCells][xCells]` arrays.
 *
 * Like {@link SimulateBinomialGridOptions} but conditioned on space-time
 * observations `(lats, lons, times, successes, trials)` evaluated against a
 * fitted space-time variogram.
 */
export interface SimulateBinomialSpaceTimeGridOptions extends GeoGridBounds {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  /** Conditioning station times (same units used by `time`). */
  times: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: SpaceTimeVariogramParams;
  prior?: BinomialPriorParams;
  /** Time at which every grid cell is evaluated (same units as `times`). */
  time: number;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
}

/**
 * Options for {@link simulateBinomialSpaceTimeGridEnsemble}: draw `nRealizations`
 * independent ST realizations at a fixed `time` over the same grid. Returns
 * flat row-major ensemble buffers (same layout as
 * {@link BinomialSimulationManyResult}).
 */
export interface SimulateBinomialSpaceTimeGridEnsembleOptions
  extends Omit<SimulateBinomialSpaceTimeGridOptions, "seed"> {
  nRealizations: number;
  /** Base seed; the k-th realization uses `baseSeed + k`. */
  baseSeed?: number | bigint;
}

/**
 * Options for {@link simulateBinomialSpaceTimeGridSummary}: draw an ensemble at
 * a fixed `time` and reduce it to per-cell summary maps in one call. Mirrors
 * {@link SimulateBinomialGridSummaryOptions}.
 */
export interface SimulateBinomialSpaceTimeGridSummaryOptions
  extends SimulateBinomialSpaceTimeGridEnsembleOptions {
  quantiles?: ReadonlyArray<number>;
  exceedanceThresholds?: ReadonlyArray<number>;
  summarizeOn?: "prevalence" | "logit";
}

/**
 * Date-aware variant of {@link SimulateBinomialSpaceTimeGridOptions}. The
 * scalar evaluation time is given as a JS `Date`; {@link DateAxisOptions}
 * configures how it (and the conditioning `dates`) are projected to the
 * model's numeric time axis.
 */
export interface SimulateBinomialSpaceTimeGridAtDateOptions
  extends GeoGridBounds,
    DateAxisOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  /** Conditioning station dates. */
  dates: readonly Date[];
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: SpaceTimeVariogramParams;
  prior?: BinomialPriorParams;
  /** Date at which every grid cell is evaluated. */
  date: Date;
  seed?: number | bigint;
}

/** Date-aware variant of {@link SimulateBinomialSpaceTimeGridEnsembleOptions}. */
export interface SimulateBinomialSpaceTimeGridEnsembleAtDateOptions
  extends Omit<SimulateBinomialSpaceTimeGridAtDateOptions, "seed"> {
  nRealizations: number;
  baseSeed?: number | bigint;
}

/** Date-aware variant of {@link SimulateBinomialSpaceTimeGridSummaryOptions}. */
export interface SimulateBinomialSpaceTimeGridSummaryAtDateOptions
  extends SimulateBinomialSpaceTimeGridEnsembleAtDateOptions {
  quantiles?: ReadonlyArray<number>;
  exceedanceThresholds?: ReadonlyArray<number>;
  summarizeOn?: "prevalence" | "logit";
}

/**
 * Result of {@link simulateBinomialGridSummary}: per-cell point estimates plus
 * optional quantile and exceedance maps on the chosen scale. All 2-D arrays have
 * shape `[yCells][xCells]`.
 */
export interface BinomialGridSummary extends GeoGridBounds {
  nRealizations: number;
  /** Per-cell ensemble mean on the logit scale. */
  meanLogit: number[][];
  /** Per-cell sample variance on the logit scale (n - 1 denominator). */
  varianceLogit: number[][];
  /** Per-cell ensemble mean on the prevalence scale. */
  meanPrevalence: number[][];
  /**
   * Per-cell quantile maps in the same order as {@link SimulateBinomialGridSummaryOptions.quantiles}.
   * Empty array if the caller did not request any quantiles.
   */
  quantiles: { probability: number; values: number[][] }[];
  /**
   * Per-cell exceedance probability maps in the same order as
   * {@link SimulateBinomialGridSummaryOptions.exceedanceThresholds}. Empty array
   * if no thresholds were requested.
   */
  exceedances: { threshold: number; values: number[][] }[];
  /** Scale used for `quantiles` and `exceedances`. */
  summarizeOn: "prevalence" | "logit";
}

// ---------------------------------------------------------------------------
// Polygon aggregation
// ---------------------------------------------------------------------------

/**
 * One polygon for {@link aggregatePrevalenceByPolygon}, expressed as a list of
 * cell indices into the underlying ensemble buffer plus per-cell weights.
 *
 * Cell indices are interpreted in the natural ordering of the supplied
 * ensemble (row-major `(j, i)` for grid ensembles produced by
 * {@link simulateBinomialGridEnsemble}). Weights must be finite and
 * non-negative; they are renormalized so only relative magnitudes matter.
 * Typical choices: all 1's for simple area means, cell population counts for
 * population-weighted prevalence, or fractional cell-coverage weights from a
 * polygon-rasterization step.
 */
export interface PolygonCells {
  /** Optional caller-supplied identifier echoed back in the result. */
  id?: string;
  /** Cell indices into the ensemble buffer (length `cells`). */
  indices: IntegerArrayInput;
  /**
   * Weights per cell (same length as `indices`). All weights must be finite
   * and `>= 0` with at least one strictly positive value.
   */
  weights: NumericArrayInput;
}

/**
 * Options for {@link aggregatePrevalenceByPolygon}: reduce one or more polygons
 * over a binomial ensemble buffer (typically from
 * {@link simulateBinomialGridEnsemble} or
 * {@link conditionalSimulateManyBinomial}).
 */
export interface AggregatePrevalenceByPolygonOptions {
  /**
   * Ensemble to aggregate. Either a {@link BinomialGridEnsemble} (the common
   * case for grid simulations) or a flat {@link BinomialSimulationManyResult}
   * for arbitrary target lists.
   */
  ensemble: BinomialGridEnsemble | BinomialSimulationManyResult;
  polygons: ReadonlyArray<PolygonCells>;
  /**
   * Probabilities in `[0, 1]` to report. May be empty.
   */
  quantiles?: ReadonlyArray<number>;
  /**
   * Which scale to summarize on. `"prevalence"` (default) reports area means
   * of prevalences; `"logit"` reports area means of the logit-scale field
   * (often more well-behaved when polygons span large prevalence ranges).
   */
  summarizeOn?: "prevalence" | "logit";
}

/** Per-polygon summary returned by {@link aggregatePrevalenceByPolygon}. */
export interface PolygonAggregateResult {
  /** Echoed from the input `PolygonCells.id` if supplied. */
  id?: string;
  /** Number of realizations summarized. */
  nRealizations: number;
  /** Sum of polygon weights (useful for chained roll-ups). */
  totalWeight: number;
  /** Posterior mean of the polygon-weighted area-mean. */
  mean: number;
  /**
   * Sample variance (`n - 1` denominator) across realizations, or `null` if
   * fewer than 2 realizations were supplied.
   */
  variance: number | null;
  /**
   * Quantile values in the same order as the input `quantiles`. Empty when
   * no quantiles were requested.
   */
  quantiles: { probability: number; value: number }[];
  /** Scale used for the summary (`"prevalence"` or `"logit"`). */
  summarizeOn: "prevalence" | "logit";
}

/**
 * Options for {@link polygonCellsFromMask}: convert a 2-D mask aligned with a
 * grid into a {@link PolygonCells} record (flat indices + weights).
 */
export interface PolygonCellsFromMaskOptions {
  /**
   * `[yCells][xCells]` mask. Falsy entries (`0`, `false`, `null`, negative,
   * `NaN`) are excluded; truthy numeric entries are used as the cell weight.
   * Pass `1`/`0` for a pure indicator mask.
   */
  mask: ArrayLike<ArrayLike<number | boolean | null | undefined>>;
  xCells: number;
  yCells: number;
  /** Optional id to attach to the resulting polygon. */
  id?: string;
}

/**
 * Fitted variogram parameters from {@link fitVariogram}.
 * Use these to construct an {@link OrdinaryKriging} model.
 */
export interface FittedVariogram {
  variogramType: VariogramTypeName;
  nugget: number;
  sill: number;
  range: number;
  /** Shape parameter (alpha for stable, nu for matern/confluent-hypergeometric); present only for shaped models. */
  shape?: number;
  /** Second shape parameter (tail-decay `alpha` for confluent-hypergeometric); present only for that model. */
  shape2?: number;
  residuals: number;
}

/**
 * Options for {@link fitVariogram}. Pass a single object with sample data and
 * variogram model type; optional settings control binning for the empirical variogram.
 */
export interface FitVariogramOptions {
  /** Sample latitudes in degrees. */
  sampleLats: NumericArrayInput;
  /** Sample longitudes in degrees. */
  sampleLons: NumericArrayInput;
  /** Sample values (same length as sampleLats/sampleLons). */
  values: NumericArrayInput;
  /** Variogram model type: string (e.g. `"exponential"`) or {@link VariogramType} enum value. */
  variogramType: VariogramTypeName | number;
  /** Optional maximum distance for binning; omit for automatic choice. */
  maxDistance?: number;
  /** Number of distance bins for the empirical variogram (default 12). */
  nBins?: number;
  /** Empirical estimator: `"classical"` (default) or `"cressie-hawkins"` (robust). */
  estimator?: EmpiricalEstimator;
}

/**
 * Variogram parameters for model construction (nugget, sill, range, optional shape).
 */
export interface VariogramParams {
  variogramType: VariogramTypeName;
  nugget: number;
  sill: number;
  range: number;
  /** Shape parameter for stable/matern/confluent-hypergeometric (nu); omit for other types. */
  shape?: number;
  /** Second shape parameter for confluent-hypergeometric (tail-decay `alpha`); omit for other types. */
  shape2?: number;
}

/**
 * Options for constructing an ordinary kriging model. Pass a single object to
 * {@link OrdinaryKriging} constructor.
 */
export interface OrdinaryKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
}

/**
 * Options for constructing a binomial kriging model. Pass a single object to
 * {@link BinomialKriging} constructor.
 */
export interface BinomialKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: VariogramParams;
}

/**
 * Beta(alpha, beta) prior parameters for binomial kriging.
 */
export interface BinomialPriorParams {
  alpha: number;
  beta: number;
}

/**
 * Build-time diagnostics for a calibrated binomial kriging model (geographic,
 * projected, or space–time). Returned by {@link BinomialKriging.getBuildNotes}
 * and included on {@link InterpolateBinomialToGridResult}.
 *
 * Field names match the WASM JSON (`camelCase`).
 */
export interface BinomialBuildNotes {
  /** Calibration pipeline version; bumps when statistical meaning changes. */
  calibrationVersion: number;
  /** Multiplier applied to base logit observation variances (1 = first attempt). */
  logitInflation: number;
  /** Number of factorization attempts (1-based). */
  nBuildAttempts: number;
  /** Beta prior used for EB-smoothed logits and observation variances. */
  prior: BinomialPriorParams;
  /** Original input row indices with `trials === 0` (dropped before fit). */
  zeroTrialDroppedIndices: number[];
  /** True when the model was built from caller logits only (no per-trial variances). */
  fromPrecomputedLogitsOnly: boolean;
}

/**
 * Options for constructing a binomial kriging model with a prior. Pass a single
 * object to {@link BinomialKriging.newWithPrior}.
 */
export interface BinomialKrigingWithPriorOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: VariogramParams;
  prior: BinomialPriorParams;
}

/**
 * Search-neighborhood restriction for a kriging model. Omit both fields (or pass `{}`)
 * to clear the neighborhood. When both are given, the intersection applies (k-nearest
 * within radius).
 */
export interface NeighborhoodOptions {
  /** Keep only the `k` closest stations at each prediction location. */
  maxNeighbors?: number;
  /** Keep only stations within this great-circle distance (kilometers). */
  maxRadius?: number;
}

/**
 * Options for constructing a simple kriging model (with known mean).
 */
export interface SimpleKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Known mean used for the residual kriging system. */
  mean: number;
}

/**
 * Options for constructing a universal kriging model with a polynomial drift basis.
 */
export interface UniversalKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Drift basis: `"constant"`, `"linear"` (in lat/lon), or `"quadratic"`. */
  trend: UniversalTrend;
}

/**
 * Projected (planar) kriging options with 2D anisotropy.
 *
 * Coordinates are arbitrary planar `(x, y)` values (for example projected meters);
 * distances are Euclidean. The anisotropy model rotates the correlation ellipse by
 * `majorAngleDeg` (counter-clockwise from +x) and scales the minor axis by
 * `rangeRatio ∈ (0, 1]`.
 */
export interface ProjectedKrigingOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Azimuth of the major axis, in degrees counter-clockwise from +x. */
  majorAngleDeg: number;
  /** Ratio of minor to major range, in `(0, 1]`. */
  rangeRatio: number;
}

/**
 * Options for binomial kriging from pre-computed logits (bypasses the default
 * empirical-Bayes shrinkage from success/trial counts).
 */
export interface BinomialFromPrecomputedLogitsOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  logits: NumericArrayInput;
  variogram: VariogramParams;
}

/**
 * Options for {@link BinomialProjectedKriging} on planar `(x, y)` coordinates with
 * 2-D anisotropy. Distances are Euclidean (optionally anisotropy-deformed); the
 * `range` is in the same linear units as `xs`/`ys`.
 */
export interface BinomialProjectedKrigingOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: VariogramParams;
  /** Azimuth of the major axis, in degrees counter-clockwise from +x. */
  majorAngleDeg: number;
  /** Ratio of minor to major range, in `(0, 1]`. `1` = isotropic. */
  rangeRatio: number;
}

/** Options for {@link BinomialProjectedKriging.newWithPrior}. */
export interface BinomialProjectedKrigingWithPriorOptions
  extends BinomialProjectedKrigingOptions {
  prior: BinomialPriorParams;
}

/**
 * Options for {@link BinomialProjectedKriging.fromPrecomputedLogits}; see
 * {@link BinomialFromPrecomputedLogitsOptions}.
 */
export interface BinomialProjectedFromPrecomputedLogitsOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  logits: NumericArrayInput;
  variogram: VariogramParams;
  majorAngleDeg: number;
  rangeRatio: number;
}

/**
 * Result of {@link computeEmpiricalVariogram} / {@link computeDirectionalEmpiricalVariogram}.
 * One entry per distance bin in increasing-distance order.
 */
export interface EmpiricalVariogramResult {
  distances: Float64Array;
  semivariances: Float64Array;
  /** Pair counts per bin. */
  counts: Uint32Array;
}

/**
 * Options for {@link computeEmpiricalVariogram}.
 */
export interface ComputeEmpiricalVariogramOptions {
  sampleLats: NumericArrayInput;
  sampleLons: NumericArrayInput;
  values: NumericArrayInput;
  /** Optional maximum distance for binning; omit for automatic choice. */
  maxDistance?: number;
  /** Number of distance bins for the empirical variogram (default 12). */
  nBins?: number;
  /** Estimator choice; defaults to `"classical"`. */
  estimator?: EmpiricalEstimator;
}

/**
 * Options for {@link computeDirectionalEmpiricalVariogram}.
 * Uses planar `(x, y)` coordinates with Euclidean distances.
 */
export interface ComputeDirectionalEmpiricalVariogramOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  values: NumericArrayInput;
  /** Maximum distance for binning (same units as `xs`/`ys`). */
  maxDistance: number;
  /** Number of distance bins. */
  nBins: number;
  /** Azimuth in degrees measured counter-clockwise from +x. */
  azimuthDeg: number;
  /** Half-angle tolerance in degrees (0, 90]. */
  toleranceDeg: number;
}

/** Per-station residual from cross-validation. */
export interface CvResidual {
  /** Index of the held-out station in the original input arrays. */
  index: number;
  /** Observed value at the held-out station. */
  observed: number;
  /** Kriging prediction from the training fold. */
  predicted: number;
  /** Kriging variance. */
  variance: number;
  /** Signed residual `observed − predicted`. */
  error: number;
}

/** Summary statistics over cross-validation residuals. */
export interface CvSummary {
  n: number;
  /** Mean signed error (bias). */
  meanError: number;
  /** Root mean squared error. */
  rmse: number;
  /** Mean squared deviation ratio; ≈ 1 when variogram is well calibrated. */
  msdr: number;
}

/** Result of {@link leaveOneOut} / {@link kFold}. */
export interface CvResult {
  /** Per-station residuals in input order. */
  residuals: CvResidual[];
  /** Aggregate summary statistics. */
  summary: CvSummary;
  /** Typed-array view of per-station predictions (convenient for plotting). */
  arrays: {
    indices: Uint32Array;
    observed: Float64Array;
    predicted: Float64Array;
    variances: Float64Array;
  };
}

/**
 * Options for {@link leaveOneOut}. Uses ordinary kriging with the supplied variogram.
 */
export interface LeaveOneOutOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
}

/**
 * Options for {@link kFold}. Folds are deterministic round-robin (station `i` → fold `i % k`).
 * Caller should shuffle inputs for randomized validation.
 */
export interface KFoldOptions extends LeaveOneOutOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutSimple}. Simple kriging treats the supplied `mean` as
 * known for every fold (no in-fold refit), matching practice for an externally estimated
 * mean.
 */
export interface LeaveOneOutSimpleOptions extends LeaveOneOutOptions {
  /** Known constant mean used by simple kriging inside each fold. */
  mean: number;
}

/** Options for {@link kFoldSimple}. */
export interface KFoldSimpleOptions extends LeaveOneOutSimpleOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutUniversal}. Trend coefficients are re-estimated inside
 * each fold from the training stations, so the trend contributes no in-sample leakage.
 */
export interface LeaveOneOutUniversalOptions extends LeaveOneOutOptions {
  /** Polynomial drift basis. `"constant"` is equivalent to ordinary kriging. */
  trend: UniversalTrend;
}

/** Options for {@link kFoldUniversal}. */
export interface KFoldUniversalOptions extends LeaveOneOutUniversalOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutProjected}. Uses planar `(x, y)` coordinates and the
 * kriging variogram's `range` must be expressed in the same linear units. When
 * `rangeRatio === 1` the model is isotropic and `majorAngleDeg` is ignored.
 */
export interface LeaveOneOutProjectedOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Angle of the major (longer-range) axis in degrees (0 = +x, counter-clockwise). */
  majorAngleDeg: number;
  /** Ratio of minor range to major range, in (0, 1]. `1` = isotropic. */
  rangeRatio: number;
}

/** Options for {@link kFoldProjected}. */
export interface KFoldProjectedOptions extends LeaveOneOutProjectedOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutBinomial}. A station whose `trials[i] === 0` is treated as
 * unobservable: it participates in no training fold, and its residual carries `NaN` for
 * observed fields (prediction is still populated). The `summary.logit` and
 * `summary.prevalence` aggregates skip those stations automatically.
 */
export interface LeaveOneOutBinomialOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: ArrayLike<number> | Uint32Array;
  trials: ArrayLike<number> | Uint32Array;
  variogram: VariogramParams;
  /** Optional Beta(alpha, beta) prior; defaults to Beta(1, 1) when omitted. */
  prior?: BinomialPriorParams;
}

/** Options for {@link kFoldBinomial}. */
export interface KFoldBinomialOptions extends LeaveOneOutBinomialOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutBinomialProjected}. Same conventions as
 * {@link LeaveOneOutBinomialOptions} but on planar `(x, y)` coordinates with
 * 2-D geometric anisotropy. Pass `rangeRatio === 1` for isotropic (the
 * `majorAngleDeg` is then ignored).
 */
export interface LeaveOneOutBinomialProjectedOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  successes: ArrayLike<number> | Uint32Array;
  trials: ArrayLike<number> | Uint32Array;
  variogram: VariogramParams;
  /** Angle of the major (longer-range) axis in degrees (0 = +x, counter-clockwise). */
  majorAngleDeg: number;
  /** Ratio of minor range to major range, in (0, 1]. `1` = isotropic. */
  rangeRatio: number;
  /** Optional Beta(alpha, beta) prior; defaults to Beta(1, 1) when omitted. */
  prior?: BinomialPriorParams;
}

/** Options for {@link kFoldBinomialProjected}. */
export interface KFoldBinomialProjectedOptions
  extends LeaveOneOutBinomialProjectedOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Per-station residual from binomial cross-validation. Reports the held-out observation
 * and the model's prediction on **both** the logit scale (directly comparable to
 * continuous kriging and MSDR-calibratable) and the prevalence scale (intuitive; delta-
 * method variance).
 *
 * When `trials === 0`, observed fields are `NaN`; see {@link LeaveOneOutBinomialOptions}.
 */
export interface BinomialCvResidual {
  /** Index of the held-out station in the original input arrays. */
  index: number;
  /** Held-out success count. */
  successes: number;
  /** Held-out trial count. `0` means the observation is undefined (`NaN` observed fields). */
  trials: number;
  /** Observed logit (log-odds of prevalence). `NaN` when `trials === 0`. */
  observedLogit: number;
  /** Model prediction on the logit scale. */
  predictedLogit: number;
  /** Kriging variance on the logit scale. */
  logitVariance: number;
  /** Observed prevalence `successes / trials`. `NaN` when `trials === 0`. */
  observedPrevalence: number;
  /** Model prediction on the prevalence scale (logistic of `predictedLogit`). */
  predictedPrevalence: number;
  /** Delta-method approximation of the variance of `predictedPrevalence`. */
  prevalenceVariance: number;
  /** Signed logit-scale error `observedLogit − predictedLogit` (`NaN` when `trials === 0`). */
  logitError: number;
  /** Signed prevalence-scale error `observedPrevalence − predictedPrevalence` (`NaN` when `trials === 0`). */
  prevalenceError: number;
}

/**
 * Aggregate binomial-CV summary reported on **both** scales. `nEvaluated` excludes
 * stations with `trials === 0` (which contribute `NaN` observations).
 */
export interface BinomialCvSummary {
  /** Total residuals, including any with `trials === 0`. */
  n: number;
  /** Number of residuals with `trials > 0`, i.e. those contributing to `logit`/`prevalence`. */
  nEvaluated: number;
  /** Summary statistics on the logit scale. */
  logit: CvSummary;
  /** Summary statistics on the prevalence scale. */
  prevalence: CvSummary;
}

/** Result of {@link leaveOneOutBinomial} / {@link kFoldBinomial}. */
export interface BinomialCvResult {
  /** Per-station residuals in input order. */
  residuals: BinomialCvResidual[];
  /** Aggregate summary on both scales. */
  summary: BinomialCvSummary;
  /** Typed-array view of per-station fields (convenient for plotting). */
  arrays: {
    indices: Uint32Array;
    successes: Uint32Array;
    trials: Uint32Array;
    observedLogit: Float64Array;
    predictedLogit: Float64Array;
    logitVariance: Float64Array;
    observedPrevalence: Float64Array;
    predictedPrevalence: Float64Array;
    prevalenceVariance: Float64Array;
  };
}

/**
 * Options for {@link conditionalSimulate}.
 *
 * Returns one sample per target in input order. When called repeatedly with the same
 * arguments (including `seed`), the output is deterministic.
 */
export interface ConditionalSimulateOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  variogram: VariogramParams;
  /** RNG seed for reproducibility (defaults to `0n`). Accepts number or bigint. */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Options for {@link conditionalSimulateMany}.
 *
 * Replaces the single `seed` field with `nRealizations` and a `baseSeed`. The k-th
 * realization is drawn with `baseSeed + BigInt(k)` so every draw is independent yet
 * deterministic.
 */
export interface ConditionalSimulateManyOptions
  extends Omit<ConditionalSimulateOptions, "seed"> {
  /** Number of independent realizations to draw (must be >= 1). */
  nRealizations: number;
  /** Seed for the first realization. Successive realizations use `baseSeed + k`. */
  baseSeed?: number | bigint;
}

/**
 * Options for {@link conditionalSimulateSimple}.
 *
 * Simulation uses simple kriging with the supplied known `mean` at every step.
 */
export interface ConditionalSimulateSimpleOptions extends ConditionalSimulateOptions {
  /** Known constant mean used by simple kriging inside the simulation loop. */
  mean: number;
}

/**
 * Options for {@link conditionalSimulateUniversal}.
 *
 * Trend coefficients are re-estimated at each simulation step. Requires at least
 * `p + 1` conditioning stations, where `p = 1` (constant), `3` (linear), or `6` (quadratic).
 */
export interface ConditionalSimulateUniversalOptions extends ConditionalSimulateOptions {
  /** Polynomial drift basis. `"constant"` is equivalent to ordinary kriging. */
  trend: UniversalTrend;
}

/**
 * Options for {@link conditionalSimulateProjected}.
 *
 * Uses planar `(x, y)` coordinates and optional 2-D geometric anisotropy. Pass
 * `rangeRatio = 1` for isotropic simulation (angle is then ignored).
 */
export interface ConditionalSimulateProjectedOptions {
  conditioningXs: NumericArrayInput;
  conditioningYs: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetXs: NumericArrayInput;
  targetYs: NumericArrayInput;
  variogram: VariogramParams;
  /** Angle of the major (longer-range) axis in degrees (0 = +x, counter-clockwise). */
  majorAngleDeg: number;
  /** Ratio of minor range to major range, in (0, 1]. `1` = isotropic. */
  rangeRatio: number;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Options for {@link conditionalSimulateBinomial}.
 *
 * Simulation happens on the **logit** scale (where the Gaussian assumption is natural) and
 * results are reported on both the logit and prevalence scales via
 * {@link BinomialSimulationResult}. Stations with `trials === 0` are dropped from the
 * initial conditioning pool.
 */
export interface ConditionalSimulateBinomialOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  variogram: VariogramParams;
  /** Optional Beta(alpha, beta) prior; defaults to Beta(1, 1) when omitted. */
  prior?: BinomialPriorParams;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Result of {@link conditionalSimulateBinomial}. Contains samples on both the logit scale
 * (unbounded) and the prevalence scale (in `(0, 1)`), in the original target input order.
 *
 * By construction, `prevalenceSamples[i] === logistic(logitSamples[i])`.
 */
export interface BinomialSimulationResult {
  /** Simulated logit values (unbounded). */
  logitSamples: Float64Array;
  /** Simulated prevalence values in `(0, 1)`. */
  prevalenceSamples: Float64Array;
}

/**
 * Options for {@link conditionalSimulateManyBinomial}. Mirrors
 * {@link ConditionalSimulateBinomialOptions} but replaces `seed` with `nRealizations`
 * and `baseSeed`. Each realization `k` is drawn with `seed = baseSeed + BigInt(k)` so
 * the k-th row of the result is bit-identical to a single
 * {@link conditionalSimulateBinomial} call with that seed.
 */
export interface ConditionalSimulateManyBinomialOptions
  extends Omit<ConditionalSimulateBinomialOptions, "seed"> {
  /** Number of independent realizations to draw (must be >= 1). */
  nRealizations: number;
  /** Seed for the first realization. Successive realizations use `baseSeed + k`. */
  baseSeed?: number | bigint;
}

/**
 * Options for {@link conditionalSimulateBinomialProjected}. Same as
 * {@link ConditionalSimulateBinomialOptions} but on planar `(x, y)` coordinates
 * with optional 2-D geometric anisotropy. Pass `rangeRatio === 1` for isotropic.
 */
export interface ConditionalSimulateBinomialProjectedOptions {
  conditioningXs: NumericArrayInput;
  conditioningYs: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  targetXs: NumericArrayInput;
  targetYs: NumericArrayInput;
  variogram: VariogramParams;
  /** Angle of the major (longer-range) axis in degrees (0 = +x, counter-clockwise). */
  majorAngleDeg: number;
  /** Ratio of minor range to major range, in (0, 1]. `1` = isotropic. */
  rangeRatio: number;
  /** Optional Beta(alpha, beta) prior; defaults to Beta(1, 1) when omitted. */
  prior?: BinomialPriorParams;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Options for {@link conditionalSimulateManyBinomialProjected}. Mirrors
 * {@link ConditionalSimulateBinomialProjectedOptions} but with `nRealizations`
 * and `baseSeed` instead of `seed`.
 */
export interface ConditionalSimulateManyBinomialProjectedOptions
  extends Omit<ConditionalSimulateBinomialProjectedOptions, "seed"> {
  /** Number of independent realizations to draw (must be >= 1). */
  nRealizations: number;
  /** Seed for the first realization. Successive realizations use `baseSeed + k`. */
  baseSeed?: number | bigint;
}

/**
 * Options for {@link conditionalSimulateManySpaceTimeBinomial}. Mirrors
 * {@link ConditionalSimulateSpaceTimeBinomialOptions} but with `nRealizations` and
 * `baseSeed` instead of `seed`.
 */
export interface ConditionalSimulateManySpaceTimeBinomialOptions
  extends Omit<ConditionalSimulateSpaceTimeBinomialOptions, "seed"> {
  /** Number of independent realizations to draw (must be >= 1). */
  nRealizations: number;
  /** Seed for the first realization. */
  baseSeed?: number | bigint;
}

/**
 * Result of {@link conditionalSimulateManyBinomial} and
 * {@link conditionalSimulateManySpaceTimeBinomial}.
 *
 * Each typed array is row-major of length `nRealizations * nTargets`. Row `k`
 * (`logitSamples.subarray(k * nTargets, (k + 1) * nTargets)`) corresponds to the k-th
 * independent realization in input target order. By construction
 * `prevalenceSamples[i] === logistic(logitSamples[i])` element-wise.
 */
export interface BinomialSimulationManyResult {
  /** Number of independent realizations stacked into the buffers. */
  nRealizations: number;
  /** Number of target locations per realization. */
  nTargets: number;
  /** Row-major simulated logits, length `nRealizations * nTargets`. */
  logitSamples: Float64Array;
  /** Row-major simulated prevalences in `(0, 1)`, length `nRealizations * nTargets`. */
  prevalenceSamples: Float64Array;
}

/** A single component of a nested (additive) variogram model. */
export interface NestedVariogramComponent {
  variogramType: VariogramTypeName;
  nugget: number;
  sill: number;
  range: number;
  shape?: number;
  /** Second shape parameter for confluent-hypergeometric (tail-decay `alpha`); omit otherwise. */
  shape2?: number;
}

/** Result of {@link evaluateNestedVariogram}. */
export interface NestedVariogramEvaluation {
  distances: Float64Array;
  semivariances: Float64Array;
  covariances: Float64Array;
}

/**
 * Options for {@link OrdinaryKriging.fromFitted}: sample data plus a fitted variogram
 * (e.g. from {@link fitVariogram}) to build the model without manually spreading variogram fields.
 */
export interface OrdinaryKrigingFromFittedOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
  fittedVariogram: FittedVariogram;
  /** If set, overrides the fitted variogram nugget when building the model (e.g. for UI-tuned sigma²). */
  nuggetOverride?: number;
}

/**
 * Options for {@link BinomialKriging.fromFittedVariogram}: count data plus a fitted variogram
 * (e.g. from fitting on logits or reusing ordinary-fit params) to build the model.
 */
export interface BinomialKrigingFromFittedVariogramOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  fittedVariogram: FittedVariogram;
  /** If set, overrides the fitted variogram nugget when building the model. */
  nuggetOverride?: number;
}

/**
 * Options for {@link BinomialKriging.fromFittedVariogramWithPrior}: count data, fitted variogram,
 * and Beta prior to build a binomial kriging model with a prior.
 */
export interface BinomialKrigingFromFittedVariogramWithPriorOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  fittedVariogram: FittedVariogram;
  prior: BinomialPriorParams;
  /** If set, overrides the fitted variogram nugget when building the model. */
  nuggetOverride?: number;
}

/**
 * Stable error codes for UI-friendly handling. When present, `KrigingError.code` is one of these.
 * Not every error has a code; the library may add new codes in minor releases.
 */
export type KrigingErrorCode =
  | "not_loaded"
  | "model_freed"
  | "mismatched_arrays"
  | "invalid_variogram"
  | "invalid_bins"
  | "singular_covariance"
  | "too_few_points"
  | "unknown_variogram"
  | "invalid_input"
  | "backend_unavailable"
  | "internal_error"
  | "unknown_family"
  | "unknown_trend"
  | "unknown_estimator"
  | "insufficient_data"
  | "callback_aborted";

// ---------------------------------------------------------------------------
// Spatio-temporal kriging types
// ---------------------------------------------------------------------------

/**
 * Family of space-time variogram models.
 *
 * - `"separable"` — `C(h_s, h_t) = C_s(h_s) · C_t(h_t) / C_s(0)` (product of normalized marginals).
 * - `"productSum"` — `C(h_s, h_t) = k1·C_s(h_s)·C_t(h_t) + k2·C_s(h_s) + k3·C_t(h_t)` with
 *   `k1 ≥ 0`, `k2 ≥ 0`, `k3 ≥ 0` and `k1 + k2 + k3 > 0`.
 */
export type SpaceTimeVariogramFamily = "separable" | "productSum";

/**
 * Fully-specified space-time variogram. Spatial and temporal marginals are ordinary
 * 2-D variograms (e.g. `{ variogramType: "exponential", nugget, sill, range }`). The
 * discriminated union forces `k1/k2/k3` to be supplied iff `family === "productSum"`.
 */
export type SpaceTimeVariogramParams =
  | {
      family: "separable";
      spatial: VariogramParams;
      temporal: VariogramParams;
    }
  | {
      family: "productSum";
      spatial: VariogramParams;
      temporal: VariogramParams;
      k1: number;
      k2: number;
      k3: number;
    };

/**
 * Drift basis for space-time universal kriging.
 *
 * Spatial components are scalar projections `(a, b)` of the coordinate (`(lat, lon)` for
 * geographic, `(x, y)` for projected), time is a scalar `t`.
 *
 * - `"constant"` — `[1]`; equivalent to ordinary space-time kriging.
 * - `"linearInTime"` — `[1, t]`.
 * - `"quadraticInTime"` — `[1, t, t²]`.
 * - `"linearInSpace"` — `[1, a, b]`.
 * - `"linearInSpaceAndTime"` — `[1, a, b, t]`.
 * - `"quadraticInSpaceAndTime"` — `[1, a, b, t, a², a·b, b², t², a·t, b·t]`.
 */
export type SpaceTimeUniversalTrend =
  | "constant"
  | "linearInTime"
  | "quadraticInTime"
  | "linearInSpace"
  | "linearInSpaceAndTime"
  | "quadraticInSpaceAndTime";

/** Common base for geographic space-time kriging model options. */
export interface SpaceTimeOrdinaryKrigingOptions {
  /** Latitudes in degrees. */
  lats: NumericArrayInput;
  /** Longitudes in degrees. */
  lons: NumericArrayInput;
  /** Sample times (scalar; any monotone unit — e.g. days, seconds). */
  times: NumericArrayInput;
  /** Sample values (same length as lats/lons/times). */
  values: NumericArrayInput;
  /** Space-time variogram parameters. */
  variogram: SpaceTimeVariogramParams;
}

/** Options for building a space-time simple kriging model with known mean. */
export interface SpaceTimeSimpleKrigingOptions extends SpaceTimeOrdinaryKrigingOptions {
  mean: number;
}

/** Options for building a space-time universal kriging model. */
export interface SpaceTimeUniversalKrigingOptions extends SpaceTimeOrdinaryKrigingOptions {
  trend: SpaceTimeUniversalTrend;
}

/** Options for building a space-time binomial kriging model (count data). */
export interface SpaceTimeBinomialKrigingOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: SpaceTimeVariogramParams;
}

/** Options for building a projected (planar) space-time ordinary kriging model. */
export interface SpaceTimeProjectedOrdinaryKrigingOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** Azimuth of the major axis, in degrees counter-clockwise from +x. */
  majorAngleDeg: number;
  /** Ratio of minor to major range, in `(0, 1]`. */
  rangeRatio: number;
}

/**
 * Options for {@link SpaceTimeOrdinaryKriging.fromFitted}: sample data plus a
 * {@link FittedSpaceTimeVariogram} (e.g. from {@link fitSpaceTimeVariogram}) to
 * build the model without manually spreading family / spatial / temporal / k
 * fields.
 */
export interface SpaceTimeOrdinaryKrigingFromFittedOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  fittedVariogram: FittedSpaceTimeVariogram;
}

/** Options for {@link SpaceTimeSimpleKriging.fromFitted}. */
export interface SpaceTimeSimpleKrigingFromFittedOptions
  extends SpaceTimeOrdinaryKrigingFromFittedOptions {
  mean: number;
}

/** Options for {@link SpaceTimeUniversalKriging.fromFitted}. */
export interface SpaceTimeUniversalKrigingFromFittedOptions
  extends SpaceTimeOrdinaryKrigingFromFittedOptions {
  trend: SpaceTimeUniversalTrend;
}

/** Options for {@link SpaceTimeBinomialKriging.fromFitted}. */
export interface SpaceTimeBinomialKrigingFromFittedOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  fittedVariogram: FittedSpaceTimeVariogram;
}

/** Options for {@link SpaceTimeProjectedOrdinaryKriging.fromFitted}. */
export interface SpaceTimeProjectedOrdinaryKrigingFromFittedOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  fittedVariogram: FittedSpaceTimeVariogram;
  /** Azimuth of the major axis, in degrees counter-clockwise from +x. */
  majorAngleDeg: number;
  /** Ratio of minor to major range, in `(0, 1]`. */
  rangeRatio: number;
}

/**
 * Options for {@link computeEmpiricalSpaceTimeVariogram}. Produces a 2-D empirical
 * variogram binned simultaneously in space and time.
 */
export interface ComputeEmpiricalSpaceTimeVariogramOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  /** Maximum spatial distance (km) for binning; defaults to half the largest pair distance. */
  maxSpatialDistance?: number;
  /** Maximum temporal lag (time units) for binning; defaults to half the largest pair lag. */
  maxTemporalLag?: number;
  /** Number of spatial bins (required, ≥ 1). */
  nSpatialBins: number;
  /** Number of temporal bins (required, ≥ 1). */
  nTemporalBins: number;
  /** Empirical estimator: `"classical"` (Matheron, default) or `"cressie-hawkins"` (robust). */
  estimator?: EmpiricalEstimator;
}

/**
 * Empirical space-time variogram: row-major flat arrays indexed by
 * `i = spatialBin * nTemporalBins + temporalBin`.
 */
export interface EmpiricalSpaceTimeVariogramResult {
  nSpatialBins: number;
  nTemporalBins: number;
  /** Mean spatial lag per bin (length = `nSpatialBins * nTemporalBins`). */
  spatialLags: Float64Array;
  /** Mean temporal lag per bin. */
  temporalLags: Float64Array;
  /** Mean semivariance per bin. */
  semivariances: Float64Array;
  /** Pair count per bin. */
  nPairs: Float64Array;
}

/** Options for {@link fitSpaceTimeVariogram}. */
export interface FitSpaceTimeVariogramOptions extends ComputeEmpiricalSpaceTimeVariogramOptions {
  /** Space-time family to fit: `"separable"` or `"productSum"`. */
  family: SpaceTimeVariogramFamily;
  /** Parametric model for the spatial marginal (e.g. `"exponential"`). */
  spatialModel: VariogramTypeName;
  /** Parametric model for the temporal marginal (e.g. `"exponential"`). */
  temporalModel: VariogramTypeName;
}

/**
 * Result of fitting a space-time variogram. Shape mirrors
 * {@link SpaceTimeVariogramParams} so you can pass `fit.fit` directly as a
 * kriging model's `variogram` option.
 */
export type FittedSpaceTimeVariogram =
  | {
      family: "separable";
      spatial: VariogramParams;
      temporal: VariogramParams;
      /** Sum-of-squared-errors of the final fit against the empirical variogram. */
      residuals: number;
    }
  | {
      family: "productSum";
      spatial: VariogramParams;
      temporal: VariogramParams;
      /** Product coefficient. */
      k1: number;
      /** Spatial-marginal coefficient. */
      k2: number;
      /** Temporal-marginal coefficient. */
      k3: number;
      /** Sum-of-squared-errors of the final fit against the empirical variogram. */
      residuals: number;
    };

/**
 * Combined empirical + parametric fit result returned by {@link fitSpaceTimeVariogram}.
 */
export interface FitSpaceTimeVariogramResult {
  empirical: EmpiricalSpaceTimeVariogramResult;
  fit: FittedSpaceTimeVariogram;
}

/**
 * Options for {@link leaveOneOutSpaceTime}. Space-time ordinary kriging CV over geographic
 * coordinates with a scalar time axis.
 */
export interface LeaveOneOutSpaceTimeOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  values: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
}

/** Options for {@link kFoldSpaceTime}. */
export interface KFoldSpaceTimeOptions extends LeaveOneOutSpaceTimeOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/** Options for {@link leaveOneOutSpaceTimeSimple}. */
export interface LeaveOneOutSpaceTimeSimpleOptions extends LeaveOneOutSpaceTimeOptions {
  /** Known constant mean used by simple ST kriging inside each fold. */
  mean: number;
}

/** Options for {@link kFoldSpaceTimeSimple}. */
export interface KFoldSpaceTimeSimpleOptions extends LeaveOneOutSpaceTimeSimpleOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/** Options for {@link leaveOneOutSpaceTimeUniversal}. */
export interface LeaveOneOutSpaceTimeUniversalOptions extends LeaveOneOutSpaceTimeOptions {
  /** Polynomial drift basis for universal ST kriging. */
  trend: SpaceTimeUniversalTrend;
}

/** Options for {@link kFoldSpaceTimeUniversal}. */
export interface KFoldSpaceTimeUniversalOptions extends LeaveOneOutSpaceTimeUniversalOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link leaveOneOutSpaceTimeBinomial}. Stations with `trials[i] === 0` are
 * treated as unobservable and carry `NaN` observed fields; summaries skip them.
 */
export interface LeaveOneOutSpaceTimeBinomialOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  times: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** Optional Beta(alpha, beta) prior; defaults to Beta(1, 1) when omitted. */
  prior?: BinomialPriorParams;
}

/** Options for {@link kFoldSpaceTimeBinomial}. */
export interface KFoldSpaceTimeBinomialOptions extends LeaveOneOutSpaceTimeBinomialOptions {
  /** Number of folds, must satisfy `2 ≤ k ≤ n`. */
  k: number;
}

/**
 * Options for {@link conditionalSimulateSpaceTime}. Space-time SGS returns one sample per
 * target in input order; deterministic for a given `seed`.
 */
export interface ConditionalSimulateSpaceTimeOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningTimes: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  targetTimes: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * Options for {@link conditionalSimulateManySpaceTime}. Mirrors
 * {@link ConditionalSimulateManyOptions} for the space-time ordinary variant.
 */
export interface ConditionalSimulateManySpaceTimeOptions
  extends Omit<ConditionalSimulateSpaceTimeOptions, "seed"> {
  /** Number of independent realizations to draw (must be >= 1). */
  nRealizations: number;
  /** Seed for the first realization. Successive realizations use `baseSeed + k`. */
  baseSeed?: number | bigint;
}

/** Options for {@link conditionalSimulateSpaceTimeSimple}. */
export interface ConditionalSimulateSpaceTimeSimpleOptions extends ConditionalSimulateSpaceTimeOptions {
  /** Known constant mean used by simple ST kriging inside the simulation loop. */
  mean: number;
}

/** Options for {@link conditionalSimulateSpaceTimeUniversal}. */
export interface ConditionalSimulateSpaceTimeUniversalOptions extends ConditionalSimulateSpaceTimeOptions {
  /** Polynomial drift basis for universal ST kriging. */
  trend: SpaceTimeUniversalTrend;
}

/**
 * Options for {@link conditionalSimulateSpaceTimeBinomial}. Simulation happens on the logit
 * scale; results are returned on both logit and prevalence scales.
 */
export interface ConditionalSimulateSpaceTimeBinomialOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningTimes: NumericArrayInput;
  successes: IntegerArrayInput;
  trials: IntegerArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  targetTimes: NumericArrayInput;
  variogram: SpaceTimeVariogramParams;
  /** Optional Beta(alpha, beta) prior; defaults to Beta(1, 1) when omitted. */
  prior?: BinomialPriorParams;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

// ---------------------------------------------------------------------------
// 3-D kriging, variography, and sequential Gaussian simulation
// ---------------------------------------------------------------------------

/**
 * GSLib-convention anisotropy parameters for the 3-D models. All 3-D classes and
 * functions accept this shape; omit it (or pass the identity `{ ang1: 0, ang2: 0,
 * ang3: 0, anis1: 1, anis2: 1 }`) for isotropic behavior.
 *
 * Angles are in degrees and follow GSLib's `setrot` convention: `ang1` is the
 * azimuth of the major axis (clockwise from north), `ang2` the dip, `ang3` the
 * plunge/rake. `anis1` is the minor/major range ratio and `anis2` the
 * vertical/major range ratio, both in `(0, 1]`.
 */
export interface Anisotropy3DParams {
  /** Azimuth of the major axis in degrees (clockwise from north). */
  ang1: number;
  /** Dip of the major axis in degrees. */
  ang2: number;
  /** Third rotation (plunge/rake) in degrees. */
  ang3: number;
  /** Minor/major range ratio in (0, 1]. */
  anis1: number;
  /** Vertical/major range ratio in (0, 1]. */
  anis2: number;
}

/**
 * Result of a single 3-D kriging prediction. In addition to the value and
 * variance, the 3-D solver reports conditioning diagnostics: the covariance
 * matrix condition number and whether nugget inflation was applied to rescue
 * an ill-conditioned system.
 */
export interface Prediction3D {
  value: number;
  variance: number;
  /** Condition number of the kriging system's covariance matrix. */
  conditionNumber: number;
  /** True when the solver inflated the nugget to stabilize an ill-conditioned system. */
  usedNuggetInflation: boolean;
}

/**
 * Batch 3-D kriging output as parallel typed arrays (avoids per-point object
 * allocation). `usedNuggetInflation` holds `0` / `1` flags, one per target.
 */
export interface Batch3DArrayOutput {
  values: Float64Array;
  variances: Float64Array;
  conditionNumbers: Float64Array;
  /** Per-target nugget-inflation flags as 0/1 values. */
  usedNuggetInflation: Float64Array;
}

/**
 * Search-neighborhood restriction for a 3-D kriging model. Distances are
 * anisotropic Euclidean in coordinate units (not kilometers). When both fields
 * are given, the intersection applies (k-nearest within radius).
 */
export interface Neighborhood3DOptions {
  /** Keep only the `k` closest samples at each prediction location. */
  maxNeighbors?: number;
  /** Keep only samples within this (anisotropic) distance in coordinate units. */
  maxRadius?: number;
}

/**
 * Options for constructing an {@link OrdinaryKriging3D} model. Coordinates are
 * arbitrary Cartesian `(x, y, z)` values (e.g. meters) with z positive upward;
 * the variogram range must be in the same linear units.
 */
export interface OrdinaryKriging3DOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  zs: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** GSLib anisotropy parameters; omit for isotropic. */
  anisotropy?: Anisotropy3DParams;
  /** Optional kd-tree search neighborhood; omit to use all samples. */
  neighborhood?: Neighborhood3DOptions;
}

/**
 * Options for constructing a {@link SimpleKriging3D} model (known global mean).
 */
export interface SimpleKriging3DOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  zs: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** Known mean used for the residual kriging system. */
  mean: number;
  /** GSLib anisotropy parameters; omit for isotropic. */
  anisotropy?: Anisotropy3DParams;
}

/**
 * Options for constructing a {@link UniversalKriging3D} model. The drift basis
 * is linear, `[1, x, y, z]` (the only basis exposed in v1).
 */
export interface UniversalKriging3DOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  zs: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** GSLib anisotropy parameters; omit for isotropic. */
  anisotropy?: Anisotropy3DParams;
}

/**
 * Options for {@link computeDirectionalVariogram3D}: gamv-style lag-centred
 * binning with a cone + bandwidth direction filter.
 *
 * For an omnidirectional variogram pass `azimuthToleranceDeg: 90`,
 * `dipToleranceDeg: 90`, and leave the bandwidths at their (effectively
 * unbounded) defaults; the engine detects this and double-counts pairs to
 * match GSLib `gamv`.
 */
export interface DirectionalVariogram3DOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  zs: NumericArrayInput;
  values: NumericArrayInput;
  /** Lag separation distance (gamv `xlag`), in coordinate units. */
  lagDistance: number;
  /** Half-window around each lag centre (gamv `xltol`). Defaults to `lagDistance / 2`. */
  lagTolerance?: number;
  /** Number of lags. */
  nLags: number;
  /** Azimuth of the direction vector in degrees (GSLib convention, clockwise from north). */
  azimuthDeg: number;
  /** Half-angle azimuth tolerance in degrees (default 22.5). */
  azimuthToleranceDeg?: number;
  /** Dip of the direction vector in degrees. */
  dipDeg: number;
  /** Half-angle dip tolerance in degrees (default 22.5). */
  dipToleranceDeg?: number;
  /** Horizontal bandwidth (gamv `bandwh`), coordinate units. Defaults to unbounded. */
  horizontalBandwidth?: number;
  /** Vertical bandwidth (gamv `bandwd`), coordinate units. Defaults to unbounded. */
  verticalBandwidth?: number;
}

/**
 * Result of {@link computeDirectionalVariogram3D}: parallel arrays aligned by
 * lag index. Empty lags are omitted, so the length is the number of non-empty
 * bins (≤ `nLags`).
 */
export interface DirectionalVariogram3DResult {
  distances: Float64Array;
  semivariances: Float64Array;
  /** Pair counts per non-empty lag. */
  nPairs: Uint32Array;
}

/**
 * One axis worth of experimental variogram, as inputs to the 3-D spherical
 * fitting functions: parallel arrays of bin distances, semivariances, and pair
 * counts (the counts weight the loss). Typically the output of
 * {@link computeDirectionalVariogram3D} along a major / minor / vertical axis.
 */
export interface AxisVariogramInput {
  distances: NumericArrayInput;
  semivariances: NumericArrayInput;
  nPairs: NumericArrayInput;
}

/**
 * Options for {@link fitSpherical3DJoint}: three axis-aligned experimental
 * variograms fitted jointly to one anisotropic spherical model.
 */
export interface FitSpherical3DOptions {
  major: AxisVariogramInput;
  minor: AxisVariogramInput;
  vertical: AxisVariogramInput;
}

/**
 * Options for {@link fitSpherical3DTwoStage}. Stage 1 fits nugget, sill, and
 * vertical range on the vertical experimental alone; stage 2 holds those and
 * fits the horizontal ranges.
 */
export interface FitSpherical3DTwoStageOptions extends FitSpherical3DOptions {
  /**
   * Sample variance of the underlying data; anchors the stage-1 sill so a
   * not-yet-plateaued vertical curve cannot drive the fit degenerate. Pass 0
   * (or omit) to leave the sill fully free.
   */
  dataVariance?: number;
}

/**
 * Options for {@link fitSpherical3DFixedNugget}: refit with the nugget held at
 * a user-supplied value (e.g. read off the vertical's short-lag intercept).
 */
export interface FitSpherical3DFixedNuggetOptions
  extends FitSpherical3DOptions {
  /** Nugget value to hold fixed during the fit. */
  nugget: number;
}

/**
 * A fitted 3-D anisotropic spherical variogram: one nugget and sill with
 * per-axis ranges. Drives 3-D kriging via {@link Anisotropy3DParams} (range
 * ratios) plus a spherical {@link VariogramParams} with `range: rangeMajor`.
 */
export interface FittedSpherical3D {
  nugget: number;
  sill: number;
  rangeMajor: number;
  rangeMinor: number;
  rangeVertical: number;
  /** Sum of weighted squared residuals at the optimum. */
  residuals: number;
}

/** Options for {@link fitSpherical1D}: one axis worth of experimental variogram. */
export type FitSpherical1DOptions = AxisVariogramInput;

/** A fitted 1-D spherical variogram (single range; see {@link fitSpherical1D}). */
export interface FittedSpherical1D {
  nugget: number;
  sill: number;
  range: number;
  residuals: number;
}

/**
 * Regular 3-D simulation grid: `nx × ny × nz` cells with the given origin and
 * spacing. Realization arrays are laid out x-fastest: linear index
 * `i + nx * j + nx * ny * k` for cell `(i, j, k)`.
 */
export interface Grid3DOptions {
  nx: number;
  ny: number;
  nz: number;
  originX: number;
  originY: number;
  originZ: number;
  spacingX: number;
  spacingY: number;
  spacingZ: number;
}

/**
 * Per-realization callback for {@link gaussianSimulation3D}. Receives the
 * realization index and the full grid as a fresh `Float64Array` (safe to
 * retain). Return `true` to abort the remaining realizations; any other
 * return value continues.
 */
export type Realization3DCallback = (
  realizationIndex: number,
  grid: Float64Array
) => boolean | void;

/**
 * Options for {@link gaussianSimulation3D}: 3-D sequential Gaussian simulation
 * conditioned on the samples, streaming one realization at a time through
 * {@link GaussianSimulation3DOptions.onRealization}. Cells whose kriging
 * system fails are marked `NaN` rather than silently filled.
 */
export interface GaussianSimulation3DOptions {
  xs: NumericArrayInput;
  ys: NumericArrayInput;
  zs: NumericArrayInput;
  values: NumericArrayInput;
  variogram: VariogramParams;
  /** GSLib anisotropy parameters; omit for isotropic. */
  anisotropy?: Anisotropy3DParams;
  grid: Grid3DOptions;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Number of realizations to stream (must be >= 1). */
  nRealizations: number;
  /**
   * When true, realizations are returned as raw normal scores instead of
   * back-transformed data-space values. Default false.
   */
  scoreSpace?: boolean;
  /** Called once per realization, in realization-index order. */
  onRealization: Realization3DCallback;
}

/**
 * Result of {@link gaussianSimulation3D}. `aborted` is true when the callback
 * stopped the stream early by returning `true`.
 */
export interface GaussianSimulation3DResult {
  aborted: boolean;
}

// ---------------------------------------------------------------------------
// Cokriging (multi-variable interpolation & cosimulation)
// ---------------------------------------------------------------------------

/**
 * A collocated secondary variable's global characterization: its `mean`, standard deviation
 * (`stdDev`), and the primary–secondary `correlation` at lag zero. Build one from paired
 * samples with {@link collocatedSecondaryFromPaired}.
 */
export interface CollocatedSecondary {
  mean: number;
  stdDev: number;
  correlation: number;
}

/** Options for {@link cokrigeCollocated}. */
export interface CokrigeCollocatedOptions {
  /** Primary sample latitudes (degrees). */
  lats: NumericArrayInput;
  /** Primary sample longitudes (degrees). */
  lons: NumericArrayInput;
  /** Primary sample values. */
  values: NumericArrayInput;
  /** Primary variogram. */
  variogram: VariogramParams;
  /** Known primary mean `m₁` (simple-cokriging). */
  primaryMean: number;
  /** Secondary characterization (dense; known at every target). */
  secondary: CollocatedSecondary;
  /** Target latitudes (degrees). */
  targetLats: NumericArrayInput;
  /** Target longitudes (degrees). */
  targetLons: NumericArrayInput;
  /** Collocated secondary value at each target (same length as `targetLats`). */
  targetSecondaryValues: NumericArrayInput;
}

/** Options for {@link collocatedCosimulate}. */
export interface CollocatedCosimulateOptions {
  conditioningLats: NumericArrayInput;
  conditioningLons: NumericArrayInput;
  conditioningValues: NumericArrayInput;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  /** Collocated secondary value at each target. */
  targetSecondaryValues: NumericArrayInput;
  variogram: VariogramParams;
  primaryMean: number;
  secondary: CollocatedSecondary;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/**
 * One LMC basic structure's correlation basis: either a pure nugget (`nugget: true`) or the
 * normalized covariance of a `variogram`. Exactly one form must be given.
 */
export interface CorrelationBasisSpec {
  /** Pure nugget structure (`ρ(h) = 1` at `h = 0`, else `0`). */
  nugget?: boolean;
  /** Normalized covariance of this variogram (its own sill/nugget cancel). */
  variogram?: VariogramParams;
}

/** One structure of a Linear Model of Coregionalization. */
export interface CoregionalizationStructureSpec {
  basis: CorrelationBasisSpec;
  /** Row-major `p × p` positive-semidefinite sill matrix. */
  sills: number[][];
}

/** A Linear Model of Coregionalization: a sum of basic structures. */
export interface CoregionalizationSpec {
  structures: CoregionalizationStructureSpec[];
}

/** Isotopic multi-variable data: all variables sampled at the same locations. */
export interface IsotopicMultiVariableData {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  /** `variables[v]` is the column of values for variable `v` at the shared coordinates. */
  variables: NumericArrayInput[];
}

/** One variable's samples in the heterotopic layout. */
export interface VariableSamples {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  values: NumericArrayInput;
}

/** Heterotopic multi-variable data: each variable has its own locations. */
export interface HeterotopicMultiVariableData {
  perVariable: VariableSamples[];
}

/** Multi-variable data, either isotopic or heterotopic. */
export type MultiVariableData =
  | IsotopicMultiVariableData
  | HeterotopicMultiVariableData;

/** Cokriging estimator: ordinary, or simple with one mean per variable. */
export type CokrigingKindSpec =
  | { type: "ordinary" }
  | { type: "simple"; means: number[] };

/** Options for {@link cokrige}. */
export interface CokrigeOptions {
  data: MultiVariableData;
  coregionalization: CoregionalizationSpec;
  kind: CokrigingKindSpec;
  /** Which variable to predict (0-based). */
  targetVariable: number;
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
}

/** Options for {@link cosimulateMultiVariable}. */
export interface CosimulateMultiVariableOptions {
  data: MultiVariableData;
  coregionalization: CoregionalizationSpec;
  /** Known mean per variable (simple cokriging). */
  means: number[];
  targetLats: NumericArrayInput;
  targetLons: NumericArrayInput;
  /** RNG seed for reproducibility (defaults to `0n`). */
  seed?: number | bigint;
  /** Optional permutation of `0..nTargets` giving the visit order. */
  targetOrder?: ArrayLike<number> | Uint32Array;
}

/** Result of {@link cosimulateMultiVariable}. */
export interface CosimulationResult {
  nVariables: number;
  nTargets: number;
  /** `samples[v][i]` is the simulated value of variable `v` at target `i`. */
  samples: number[][];
}

/** Options for {@link computeCrossVariogram}. */
export interface CrossVariogramOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  variables: NumericArrayInput[];
  nBins: number;
  maxDistance?: number;
}

/** Result of {@link computeCrossVariogram}. */
export interface CrossVariogramResult {
  nVariables: number;
  distances: number[];
  nPairs: number[];
  /** `gamma[bin]` is the flattened row-major `p × p` cross-semivariance matrix at that lag. */
  gamma: number[][];
}

/** Options for {@link fitLmc}. */
export interface FitLmcOptions {
  lats: NumericArrayInput;
  lons: NumericArrayInput;
  variables: NumericArrayInput[];
  nBins: number;
  maxDistance?: number;
  /** The fixed basic structures (ranges/shapes chosen up front); their sills are fitted. */
  bases: CorrelationBasisSpec[];
  /** Maximum Goulard–Voltz sweeps (defaults to 200). */
  maxIterations?: number;
  /** Relative-change stopping tolerance (defaults to 1e-7). */
  tolerance?: number;
}

/** Result of {@link fitLmc}. */
export interface LmcFitResult {
  /** The fitted, admissible coregionalization — ready to feed to {@link cokrige}. */
  coregionalization: CoregionalizationSpec;
  /** Final weighted sum of squared residuals. */
  residual: number;
  /** Number of Goulard–Voltz sweeps performed. */
  iterations: number;
}
