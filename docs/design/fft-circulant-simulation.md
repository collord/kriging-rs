# Design: FFT circulant-embedding simulation for kriging-rs

Status: draft. Written 2026-11-06; conditional-simulation plan added 2026-11-18.

## Summary

We will add a circulant-embedding (FFT) Gaussian simulator for regular 2-D planar and 3-D grids,
plus conditioning by kriging, as a faster and exact alternative to sequential Gaussian simulation
(SGS). The method follows Dietrich & Newsam (SIAM J. Sci. Comput. 18(4), 1997). It reproduces the
variogram exactly at every lag and costs one FFT per pair of realizations.

On a 256 × 256 grid we expect a few milliseconds per realization pair, against roughly 0.5–1 s per
realization for the current 3-D SGS with 24 neighbours (estimates, to be confirmed by the M1
benchmark). The new path is additive: existing SGS APIs stay, and the new one is offered where its
assumptions hold (stationary, regular grid, planar or 3-D coordinates).

A related defect is fixed alongside: the lat/lon and projected 2-D SGS refit a full kriging system
on every step, which makes them O(N⁴) and unusable beyond a few hundred targets.

## Goals and non-goals

**Goals (v1)**

- Unconditional simulation of a stationary Gaussian field on a regular 2-D planar grid and a
  regular 3-D grid (`Grid3D`).
- Conditional simulation by kriging, honouring sample data exactly, with simple-kriging (known
  mean) and ordinary-kriging variants.
- Any covariance model in `VariogramModel` except `Power` (intrinsic, no covariance), with nugget.
  Nested structures via `NestedVariogram` (spectra add).
- Geometric anisotropy: `Anisotropy2D` (CCW from +x) and `Anisotropy3D` (GSLib convention),
  matching the existing kriging paths.
- Normal-score workflow identical to 3-D SGS: simulate in score space, back-transform with
  `NormalScoreTransform`.
- Reproducible output for a given seed, on native and wasm.
- A WASM export and typed TS wrapper, streaming realizations through a callback like
  `gaussianSimulation3D`.

**Non-goals (v1)**

- Lat/lon grids. Haversine distance is not translation-invariant in grid-index space, so the
  embedding does not apply. Callers project first (the `examples/web-2d` template already works in
  planar coordinates).
- Scattered (non-grid) targets. Users needing those keep SGS.
- Universal kriging (polynomial drift) in the conditioning step. Deferred to v2.
- Binomial and space-time variants. Space-time with a separable or product-sum model on a regular
  space-time grid is a natural v2 (it becomes a 3-D FFT).
- GPU FFT. The existing WGSL kernel is f32-only and unrelated; revisit after v1 numbers.

## Background

The covariance matrix of a stationary field on a regular grid is block-Toeplitz. Dietrich & Newsam
embed it in a block-circulant matrix of twice the size per axis. A circulant matrix is diagonalised
by the DFT, so its eigenvalues are one FFT of its first row. If every eigenvalue is non-negative,
multiplying complex white noise by √λ and transforming gives a field whose covariance is exactly
the target on the original grid. The real and imaginary parts are two independent realizations.

| Path | Where | Cost per realization | Covariance reproduction |
| --- | --- | --- | --- |
| Circulant embedding (proposed) | new | O(d log d), d = embedded cells; 2 realizations per FFT | Exact |
| 3-D SGS, kd-tree, `max_neighbors = 24` | `src/simulation_3d/sgs_3d.rs` | O(N (log N + k³)) | Approximate; neighbourhood truncation weakens long lags |
| 2-D / space-time SGS, all points | `src/simulation.rs` (`conditional_simulate*`) | O(N⁴) total (dense LU of the growing system each step) | Exact, but impractical beyond a few hundred targets |

The conditional extension is Dietrich & Newsam, Water Resour. Res. 32 (1996), and the standard
conditioning-by-kriging identity (Journel & Huijbregts 1978). Wood & Chan (1994) give the general
padding strategy when the minimal embedding is not non-negative.

## Algorithm

The simulator is built once per (grid, variogram, anisotropy), then draws realization pairs from
it. All set-up arithmetic runs in f64 regardless of the `Real` feature.

**Set-up (once)**

1. **Choose the embedding size per axis.** For grid size nₐ and spacing Δₐ, pick
   Mₐ ≥ max(2(nₐ − 1), nₐ + ⌈p · aₐ / Δₐ⌉), rounded up to the next 2ᵃ 3ᵇ 5ᶜ size. Here aₐ is the
   effective range along that axis after anisotropy, and p is the padding factor (default 1.0).
   For models with finite support (spherical, cubic) p = 1 is sufficient by Corollary 1 of the
   paper.
2. **Fill the first row.** For each embedded index j, map each axis to a signed lag ℓₐ = jₐ if
   jₐ < Mₐ/2, else jₐ − Mₐ. Set c[j] = C(h) with h = (ℓₓΔₓ, ℓᵧΔᵧ, ℓ_zΔ_z) and C evaluated on the
   anisotropic distance (`Anisotropy2D::distance`, `Anisotropy3D::transform_lag`). At the midpoint
   jₐ = Mₐ/2 the two signed lags differ under rotated anisotropy, so average them (the paper's φ
   term, eq. 22). Zero lag gets the total sill, so the nugget lands on the diagonal and raises
   every eigenvalue by the same amount.
3. **Eigenvalues.** λ = FFT(c). λ is real because c is symmetric under j → −j. Assert the
   imaginary part is below 1e-9 · max|λ|.
4. **Check non-negativity.** Let λ_min = min λ. If λ_min ≥ −ε · max λ (ε = 1e-10), clamp the
   negatives to 0. Otherwise double p on the offending axes and repeat from step 1, up to the
   `max_embedding_cells` budget. If the budget runs out, return `EmbeddingNotPsd` with
   λ_min / max λ, unless the caller opted into `allow_approximate`. In that case, clamp and report
   the fraction of variance discarded.
5. **Store** √(λ / d) as an f64 (or `Real`) buffer of length d = ∏ Mₐ.

**Per realization pair**

1. Draw ξ = ξ₁ + iξ₂ with ξ₁, ξ₂ ~ N(0, I) of length d, from an RNG seeded by
   `splitmix64_mix(seed, pair_index)`, as in the parallel 3-D SGS.
2. Compute e = FFT(√(λ/d) ∘ ξ).
3. Realization 2k is Re(e) and 2k + 1 is Im(e), each restricted to the original nₓ × nᵧ (× n_z)
   corner. For an odd count, the last imaginary part is dropped.

```math
S = \tfrac{1}{d}\, F \Lambda F^{H},\quad \lambda = F c,\quad e = F\left(\sqrt{\Lambda / d}\;\xi\right),\quad \operatorname{Re} e,\ \operatorname{Im} e \sim \mathcal{N}(0, S)\ \text{independently}
```

**Multi-dimensional FFT.** `rustfft` only provides 1-D transforms. A 2-D or 3-D FFT is done as 1-D
passes along each axis, with a strided gather/scatter into a scratch line per pass. Plans are
cached per axis length in the simulator. The layout is x-fastest (`i + nx·j + nx·ny·k`), matching
`Grid3D::linear_index`.

## Conditioning by kriging

Conditional realizations come from an unconditional one plus a kriged correction. Because the data
are snapped to grid nodes, every term is a convolution with the covariance, so the correction
reuses the embedding's FFTs and costs no per-cell kriging.

```math
z_c(x) = z_s(x) + \mu + \sum_i \beta_i\, C(x - u_i),\qquad K\beta = z(u) - z_s(u) - \mu\mathbf{1}
```

Here z_s is an unconditional realization, uᵢ are the data nodes, and K is the n × n data
covariance (nugget on the diagonal). For simple kriging μ = 0 in score space (or the known mean).
For ordinary kriging μ is the generalised-least-squares mean of the residuals,
μ = 1ᵀK⁻¹r / 1ᵀK⁻¹1 with r = z(u) − z_s(u). This is the dual form of the bordered OK system.

**Steps**

1. **Snap data to nodes** (set-up). Each datum goes to its nearest grid node (cell centre). When
   several data share a node, keep the one closest to the node centre (GSLib `sgsim` convention)
   and report the count dropped. A datum more than half a cell outside the grid is an error.
   Snapping moves a datum at most half a cell diagonal; with spacing well under the range this is
   negligible, and the docs will say so.
2. **Normal scores** (set-up). Fit `NormalScoreTransform` on the kept data and transform, exactly
   as `SgsModel3D::new` does. Simulation runs in score space with a score-space variogram, and
   back-transform is the last step.
3. **Factorize K once** (set-up) with Cholesky through the existing robust solver
   (`src/kriging/solver.rs`, `SolverConfig`). Ordinary kriging reuses the same factor plus the
   precomputed vector K⁻¹1, as the solver already does for its Lagrangian.
4. **Per realization:** read z_s at the data nodes, solve for β (and μ) by back-substitution (n²),
   scatter β onto a zeroed embedded grid, and compute the correction as FFT⁻¹(λ ∘ FFT(β-grid)).
   Lags between two in-grid nodes never wrap on the torus, so this equals the sum exactly.
5. **Back-transform** each cell, or return score space when `output_space = ScoreSpace`.

Cost per conditional realization: about 1.5 FFTs of size d (two correction FFTs shared across the
pair, plus half of the unconditional FFT) and O(n²). Data honour exactly at their nodes because
K and the convolution use the same covariance (see C4 below).

**Limits.** The global data system bounds n. Cholesky is n³/3 flops and the factor holds n² f64
values: at n = 2,000 that is about 0.3 s and 32 MB, at n = 5,000 about 4 s and 200 MB in
single-threaded wasm (estimates). v1 caps n at 5,000 on native and 2,000 on wasm32, with a clear
`TooManySamples` error. A moving-neighbourhood correction is the v2 route beyond that.

## Conditional simulation: implementation plan

The method above is settled; this section lists the work to build it (milestone M3). Six work
items, C1–C6, each its own commit with tests written first. C1 and C2 have no dependency on the
FFT engine and can start in parallel with M1/M2.

### C1. A reusable factored data system (`src/kriging/solver.rs`)

`factor_with_retries` and `build_covariance_block` are private today, and the public solvers
re-factor K for every target. Conditioning needs K factored once and solved thousands of times.

- Make `factor_with_retries` `pub(crate)`.
- Add a `pub(crate) struct FactoredDataSystem { chol, k_inv_one, one_k_inv_one, used_inflation,
  cond_proxy }` built from an n × n `DMatrix<f64>`.
- Methods: `solve_simple(&self, r: &[f64], out: &mut [f64])` and
  `solve_ordinary(&self, r: &[f64], out: &mut [f64]) -> f64` (returns μ). No allocation per call.
- Expose `used_inflation`. When the solver inflates the diagonal, the data are no longer honoured
  exactly, so `ConditionalFftModel` reports `honours_data_exactly = false` in its info (and the TS
  `info` mirrors it).

Tests: SK and OK weights from `FactoredDataSystem` match `solve_simple_kriging_3d` /
`solve_ordinary_kriging_3d` at a probe target to 1e-10 (f64); μ equals the OK mean; inflation is
reported on a deliberately singular K (duplicate points).

### C2. Data snapping (`src/simulation_fft/conditional.rs`)

- `fn snap_to_grid(coords, values, grid) -> Result<SnappedData, FftSimError>` where
  `SnappedData { node: Vec<usize>, values: Vec<Real>, dropped: usize, max_offset_cells: f64 }`.
- Nearest cell centre per axis; on a shared node keep the datum closest to the centre (ties: first
  in input order, so results are deterministic).
- More than half a cell outside the grid on any axis → `SampleOutsideGrid { index }`.
- Works for both front ends: 2-D data come from `ProjectedDataset` (nz = 1), 3-D data from
  `PlanarDataset3D`. Accept an internal `enum ConditioningData { Planar2D(..), Planar3D(..) }`
  rather than forcing 2-D callers through a 3-D dataset.
- Coordinates must already be centred by the same origin as the grid (`PlanarOrigin` /
  `Origin3D` in the wasm layer); the Rust API takes grid and data in one frame and does not
  re-centre.

Tests: exact node hit, half-cell offsets on each axis, duplicates in one cell (`dropped = 1`,
closest kept), outside-grid error, nz = 1.

### C3. Score space and the data covariance matrix

- Fit `NormalScoreTransform` on the kept (post-snap) values; transform to scores.
- Report `score_sill = C(0)` of the supplied score-space variogram in info. A sill far from 1
  means the caller fitted the variogram in data space by mistake. The TS layer warns when
  |sill − 1| > 0.2; the Rust layer only reports.
- Build K from the circulant covariance, not the variogram: K_ij = c̃[lag(uᵢ − uⱼ)], where
  c̃ = FFT⁻¹(λ_clamped) is computed once at conditional set-up and dropped after K is built. When
  the embedding was clamped (`allow_approximate`), K then matches the field's actual covariance
  and data are still honoured exactly. Without clamping c̃ equals c to about 1e-10.

Tests: K from c̃ equals K from `build_covariance_block` to 1e-10 on an unclamped embedding;
honouring still holds with a clamped one.

### C4. Covariance convolution on the simulator (`src/simulation_fft/unconditional.rs`)

- `pub(crate) fn convolve_pair(&self, b: &mut [Complex<Real>])`: forward FFT, multiply by λ/d
  (= the stored √(λ/d) squared, so no extra buffer), inverse FFT, in place.
- Pack two realizations' β into one complex grid (β₁ + iβ₂). λ is real and c is even, so the
  real and imaginary parts of the output are the two corrections. One FFT pair serves two
  realizations.
- Only data nodes are non-zero in the input, and only the original grid corner is read from the
  output.

Tests: against a direct Σ βᵢ C̃(x − uᵢ) on a small grid with 5 data, to 1e-6 (f32) / 1e-10 (f64);
Re and Im outputs match two separate single-input convolutions.

### C5. The conditional driver (`ConditionalFftModel`)

Per realization pair:

1. Draw the unconditional pair (Re, Im) into the working buffer.
2. Gather z_s at the data nodes for both; form residuals r₁, r₂ (minus μ for OK).
3. Solve β₁, β₂ with `FactoredDataSystem` (and μ₁, μ₂ for OK).
4. Scatter β₁ + iβ₂ into the zeroed correction buffer; `convolve_pair`.
5. Each output cell = z_s + correction (+ μ); restrict to the grid corner; back-transform per
   `SgsOutputSpace`; hand to the callback. An odd count skips the second member.

- Serial `stream` for wasm; `stream_parallel` on native with rayon over pairs. Each worker owns
  its two complex buffers, so cap the worker count at
  `max_embedding_cells × 16 B × workers ≤ memory budget` (default: four workers).
- Seeds per pair use `splitmix64_mix(seed, pair_index)`, so the parallel and serial output are
  identical per index.
- `TooManySamples` when n exceeds the cap (5,000 native, 2,000 wasm32) is checked before
  factoring.

Tests: see C6.

### C6. Validation tests

- Honouring: |z_c(uᵢ) − z(uᵢ)| < 1e-5 in score space at every data node, every realization, SK
  and OK, 2-D and 3-D.
- SK parity (3-D): over 1,000 realizations, the ensemble mean and variance in score space match
  `SimpleKrigingModel3D` (global neighbourhood, same data snapped to nodes) within Monte Carlo
  tolerance (4σ/√N on the mean, 10% on the variance).
- OK parity: the same against ordinary kriging (3-D model in 3-D, `ProjectedKrigingModel` in 2-D).
- Far from data, the ensemble variance returns to the sill and the ensemble variogram matches the
  model (reuses the unconditional statistical harness).
- Nugget invariant: extend `larger_nugget_raises_kriging_variance_in_every_path` so a larger
  nugget raises the conditional ensemble variance at cells next to data.
- Clamped embedding (`allow_approximate`): honouring still holds (guards C3).
- Determinism: same seed bit-identical on one platform; parallel equals serial per index.
- Data-space round trip: back-transformed realizations stay within the NST's tail bounds, and the
  ensemble at a data node equals the datum.

### Conditioning: what is deliberately left for v2

- Off-node data without snapping (local-kriging placement of the unconditional value at the true
  location).
- Moving-neighbourhood correction for n above the cap.
- Universal-kriging conditioning (polynomial drift), binomial and space-time conditioning.

## Rust API and module layout

A new module, `src/simulation_fft/`, holds one dimension-agnostic engine (always three axes, nz = 1
for 2-D) behind thin 2-D and 3-D front ends. It reuses `Grid3D`, `PlanarDataset3D`,
`ProjectedDataset`, `NormalScoreTransform`, `Rng` and the kriging solver rather than adding
parallel types.

| File | Contents |
| --- | --- |
| `mod.rs` | Public re-exports, module docs |
| `fftnd.rs` | Axis-pass N-D FFT over `rustfft`, plan cache, scratch buffers |
| `embedding.rs` | `CirculantEmbedding`: size search, first row, eigenvalues, PSD check, `EmbeddingInfo` |
| `unconditional.rs` | Pair sampler, `convolve_pair`, streaming and parallel drivers |
| `conditional.rs` | Data snapping, NST, K from c̃, `ConditionalFftModel` |
| `covariance.rs` | `StationaryCovariance` trait for `VariogramModel` and `NestedVariogram`, plus the anisotropic lag-to-distance adapters |

```rust
pub struct EmbeddingOptions {
    pub padding: f64,               // default 1.0 (ranges of padding per axis)
    pub max_embedding_cells: usize, // default 1 << 24 native, 1 << 22 on wasm32
    pub psd_tolerance: f64,         // default 1e-10, relative to max eigenvalue
    pub allow_approximate: bool,    // default false: clamp large negatives and report
}

pub struct EmbeddingInfo {
    pub dims: [usize; 3],
    pub cells: usize,
    pub padding_used: [f64; 3],
    pub min_eigen_ratio: f64,       // min λ / max λ before clamping
    pub clamped_variance_fraction: f64,
}

pub struct CirculantSimulator { /* √(λ/d), dims, grid, plans */ }

impl CirculantSimulator {
    pub fn new_2d(grid: PlanarGrid2D, cov: impl StationaryCovariance,
                  anisotropy: Anisotropy2D, opts: EmbeddingOptions) -> Result<Self, FftSimError>;
    pub fn new_3d(grid: Grid3D, cov: impl StationaryCovariance,
                  anisotropy: Anisotropy3D, opts: EmbeddingOptions) -> Result<Self, FftSimError>;
    pub fn info(&self) -> &EmbeddingInfo;
    pub fn stream<F>(&self, seed: u64, n: usize, on: F) -> Result<(), FftSimError>
        where F: FnMut(usize, &[Real]) -> Result<(), FftSimError>;
    #[cfg(not(target_arch = "wasm32"))]
    pub fn stream_parallel<F>(&self, seed: u64, n: usize, on: F) -> Result<(), FftSimError>
        where F: FnMut(usize, &[Real]) -> Result<(), FftSimError> + Send;
}

pub struct ConditionalFftModel { /* simulator, snapped nodes, NST, FactoredDataSystem */ }

pub struct ConditioningInfo {
    pub n_used: usize,
    pub dropped_samples: usize,
    pub max_offset_cells: f64,
    pub score_sill: f64,
    pub honours_data_exactly: bool, // false if the solver inflated K's diagonal
}

impl ConditionalFftModel {
    pub fn new_2d(sim: CirculantSimulator, data: ProjectedDataset,
                  kind: ConditioningKind, solver: SolverConfig) -> Result<Self, FftSimError>;
    pub fn new_3d(sim: CirculantSimulator, data: PlanarDataset3D,
                  kind: ConditioningKind, solver: SolverConfig) -> Result<Self, FftSimError>;
    pub fn info(&self) -> &ConditioningInfo;
    pub fn stream<F>(&self, seed: u64, n: usize, space: SgsOutputSpace, on: F)
        -> Result<(), FftSimError> where F: FnMut(usize, &[Real]) -> Result<(), FftSimError>;
}

pub enum ConditioningKind { Simple { mean: Real }, Ordinary }
```

Callbacks use the existing SGS shape, `FnMut(usize, &[Real]) -> Result<(), _>`, so an `Err`
aborts the stream, as in `gaussian_simulation_3d_stream`.

**Errors.** `FftSimError` has `InvalidInput`, `UnsupportedVariogram` (Power),
`EmbeddingNotPsd { min_eigen_ratio, cells_tried }`, `EmbeddingTooLarge { cells, budget }`,
`SampleOutsideGrid { index }`, `TooManySamples { n, max }`, `Kriging(KrigingError)` and
`CallbackAborted`. The WASM layer maps each to a stable `code` through `coded_err`.

**2-D grid type.** No Rust 2-D grid struct exists today; the wasm layer builds projected grid
coordinates from bounds and cell counts (`build_grid_coords` in `src/wasm/mod.rs`). Add
`PlanarGrid2D { nx, ny, origin, spacing }` with cell centres and row 0 = south, matching projected
`predictGrid`.

**Dependency.** Add `rustfft` 6.x (pure Rust, builds for wasm32, SIMD on native). Measure the wasm
size change in M1. If it exceeds about 150 KB gzipped, put the module behind a `fft` feature that
`wasm` enables.

## WASM and TypeScript surface

The embedding set-up is the expensive, reusable part, so JS gets a model object with `free()` /
`[Symbol.dispose]`. It also gets one-shot helpers whose options mirror `gaussianSimulation3D`, so
switching engines is a one-word change.

**Raw exports** (new `src/wasm/fft_sim.rs`, following AGENTS.md step 2):

- `#[wasm_bindgen] struct WasmFftSimulator` with constructors `new_2d(...)` and `new_3d(...)`. They
  take flat sample arrays (empty for unconditional), the anisotropy scalars, `variogram: JsValue`
  through `variogram_model_from_js`, the grid scalars, `padding`, `max_cells`,
  `allow_approximate`, `conditioning` (0 = simple, 1 = ordinary) and `mean`.
- Coordinates are centred with `Origin3D::centroid` (3-D) or `PlanarOrigin` (2-D) in f64 before
  narrowing to `Real`. The grid origin is shifted by the same centre.
- Methods: `info() -> JsValue` (`EmbeddingInfo` plus `ConditioningInfo` when conditioned), and
  `stream(seed, n, score_space, on_realization: &Function)`, which is synchronous and in index
  order, like `wasm_gaussian_simulation_3d`.
- Errors go through `kriging_err_to_js` / `coded_err` with the codes `embedding_not_psd`,
  `embedding_too_large`, `unsupported_variogram`, `sample_outside_grid`, `too_many_samples` and
  `mismatched_arrays`.

**TypeScript** (`npm/kriging-rs-wasm/src/simulate-fft.ts`, types in `types.ts`, exports in
`index.ts`):

```ts
export interface FftEmbeddingOptions {
  padding?: number;           // default 1
  maxCells?: number;          // default 2 ** 22 (4.2 M cells)
  allowApproximate?: boolean; // default false
}

export interface FftSimulation3DOptions
  extends Omit<GaussianSimulation3DOptions, "xs" | "ys" | "zs" | "values"> {
  xs?: NumericArrayInput; ys?: NumericArrayInput; zs?: NumericArrayInput;
  values?: NumericArrayInput;           // omit all four for unconditional
  conditioning?: "simple" | "ordinary"; // default "simple" (score-space mean 0)
  embedding?: FftEmbeddingOptions;
}

export interface FftSimulation2DOptions {
  xs?: NumericArrayInput; ys?: NumericArrayInput; values?: NumericArrayInput;
  variogram: VariogramParams;
  anisotropy?: { majorAngleDeg: number; rangeRatio: number };
  grid: PredictGridProjectedOptions;  // xMin..yMax, xCells, yCells; row 0 = south
  seed?: number | bigint;
  nRealizations: number;
  scoreSpace?: boolean;
  conditioning?: "simple" | "ordinary";
  embedding?: FftEmbeddingOptions;
  onRealization: Realization3DCallback; // same (index, Float64Array) => boolean | void
}

export function createFftSimulator3D(o: Omit<FftSimulation3DOptions, "seed" | "nRealizations" | "onRealization" | "scoreSpace">): FftSimulator;
export function createFftSimulator2D(o: Omit<FftSimulation2DOptions, "seed" | "nRealizations" | "onRealization" | "scoreSpace">): FftSimulator;
export function fftSimulation3D(o: FftSimulation3DOptions): GaussianSimulation3DResult;
export function fftSimulation2D(o: FftSimulation2DOptions): GaussianSimulation3DResult;

export interface FftSimulator {
  // dims, cells, minEigenRatio, clampedVarianceFraction; plus nUsed, droppedSamples,
  // maxOffsetCells, scoreSill, honoursDataExactly when conditioned
  readonly info: FftSimulatorInfo;
  simulate(o: { seed?: number | bigint; nRealizations: number; scoreSpace?: boolean;
                onRealization: Realization3DCallback }): GaussianSimulation3DResult;
  free(): void;
  [Symbol.dispose](): void;
}
```

Every wrapper calls the raw module inside `wrapThrown`, so failures surface as `KrigingError` with
a `code`. The npm README gets a "Fast grid simulation (FFT)" section covering when to choose it
over SGS, the lat/lon restriction and the snapping caveat. `examples/web-2d` gains an engine
toggle, with the work running in its existing Web Worker.

## Numerics, memory and performance

Memory, not time, is the binding limit. The embedded grid is 4× the cells in 2-D and 8× in 3-D
before any padding, and a 3-D grid of 2M cells already needs about 300 MB in the default f32 build.

**Precision.**

- Set-up (first row, eigenvalue FFT, PSD check) always runs in f64. Smooth models (Gaussian,
  Matérn with large ν, cubic) have eigenvalues that decay to about 1e-16 · max λ. In f32 that tail
  is noise and would be misread as indefinite.
- Per-realization FFTs run in `Real`. In f32 the round-off is about 1e-7 · log₂ d relative to the
  field's standard deviation, roughly 3e-6 at d = 2²⁴, which is far below Monte Carlo noise.
- Determinism is promised per platform only. `rustfft` picks SIMD kernels per target, so native
  and wasm can differ in the last bits. Tests compare across targets with a tolerance.

**Memory per embedded cell** (f32 build): √λ 4 B, a working complex buffer 8 B, plus 8 B for the
conditioning correction buffer. Set-up adds a transient 16 B (f64 complex). Computing λ with a
real-to-complex transform, exploiting the even symmetry of c, halves that transient and is a
later optimisation. Conditioning also holds the n² f64 factor of K (32 MB at n = 2,000).

| Grid | Embedded cells (p = 0) | Steady, unconditional | Steady, conditional | Set-up peak |
| --- | --- | --- | --- | --- |
| 256 × 256 | 512 × 512 = 0.26 M | 3 MB | 5 MB | 4 MB |
| 1024 × 1024 | 2048 × 2048 = 4.2 M | 50 MB | 84 MB | 67 MB |
| 100 × 100 × 50 | 200 × 200 × 100 = 4.0 M | 48 MB | 80 MB | 64 MB |
| 200 × 200 × 50 | 400 × 400 × 100 = 16 M | 192 MB | 320 MB | 256 MB |

Padding for long ranges or smooth models multiplies these numbers. Default `max_embedding_cells`
is therefore 2²⁴ on native and 2²² (4.2 M) on wasm32. Callers can raise either.

**Time.** An FFT costs about 5 d log₂ d flops. At 1–2 GFLOP/s effective in scalar wasm, and 5–10
on native, that gives the estimates below. M1 replaces them with measurements from `benches/`.

| Embedded cells | wasm, per pair | native, per pair | 3-D SGS, per realization (same grid, k = 24) |
| --- | --- | --- | --- |
| 0.26 M | 15–25 ms | 3–5 ms | 0.5–1 s |
| 4.2 M | 0.3–0.5 s | 50–100 ms | 5–10 s |
| 16 M | 1–2 s | 0.2–0.4 s | 20–40 s |

Conditioning adds two FFTs per pair plus the O(n²) solve, roughly tripling the per-pair time.

## Testing and validation

Exactness can be tested deterministically, without Monte Carlo: the inverse FFT of the clamped
eigenvalues must give back the covariance on every in-grid lag. That check is the core regression
guard. Statistical tests then cover the sampler; the conditioning tests are listed under C6. Each
test is written before its code, per `.cursor/rules/rust-tdd.mdc`, and runs under both
`cargo test` and `cargo test --features f64`.

**Deterministic (Rust unit tests)**

- `fftnd`: matches a naive DFT on small odd and even shapes (6 × 5 × 3, 8 × 1 × 1, 1 × 7 × 4).
- Embedding exactness: IFFT(λ) equals c within 1e-10 for exponential, spherical, Gaussian and
  nested models, with and without rotated anisotropy.
- Theorem 2 cases: the minimal embedding for exponential and spherical is PSD with no padding, for
  n in 8..256 and range/Δ in 1..50.
- Gaussian with no nugget triggers padding growth. An exhausted budget returns
  `EmbeddingTooLarge`. `allow_approximate` reports a non-zero `clamped_variance_fraction`.
- `Power` returns `UnsupportedVariogram`.
- Determinism: the same seed is bit-identical on one platform, different seeds differ, and
  parallel output matches serial per index (mirroring the existing
  `parallel_same_seed_per_realization_is_bit_identical`).

**Statistical (Rust, seeded, tolerance from the ensemble size)**

- Unconditional, 2,000 realizations on 64 × 64: the cell mean is within 4σ/√N of 0, the variance
  is within 5% of the sill, and the empirical variogram is within 5% of the model at lags 1 to the
  range.
- The real and imaginary members of a pair have correlation below 0.05.
- Anisotropy conventions: directional variograms along the major and minor axes match
  `Anisotropy2D::distance` (CCW from +x) and `Anisotropy3D` (GSLib).
- Parity with SGS: on the same 3-D grid, the FFT ensemble variogram is no further from the model
  than the SGS ensemble at every lag, and closer beyond the neighbourhood radius.

**JS and performance**

- vitest cases in `tests/integration.test.ts` for `fftSimulation2D`, `fftSimulation3D` and
  `createFftSimulator*`: shapes, callback order and abort, `info`, `free()`, and each error
  `code`. This is the only check that the wasm-shapes signatures match, so it covers argument
  order explicitly.
- Add both 2-D and 3-D FFT runs to the `npm run verify` smoke script.
- Add `benches/bench_fft_sim.rs`, run on the same grids as `bench_3d_sgs_streaming.rs`, and a
  browser row in `benches/BROWSER_BENCHMARKS.md`.

## Related fix: 2-D SGS refits the full system each step

Every SGS variant in `src/simulation.rs` rebuilds its kriging model from all conditioning plus all
previously simulated points on every step. That means a dense LU of an (n + k + 1)² system per
target, for example `OrdinaryKrigingModel::new` (line 267 of `src/simulation.rs`) calling `.lu()`
in `src/kriging/ordinary.rs`. The total cost is O(N⁴), so 10,000 targets means 10,000
factorizations of up to 10,000 × 10,000.

Affected: ordinary, simple, universal and projected SGS; collocated cosimulation and full
cosimulation; the binomial variants; all space-time variants; and the `_many` wrappers that call
them.

**Fix.** Give each loop a search neighbourhood, as the 3-D SGS already has:

1. Add `max_neighbors: Option<usize>` (default 24) and `max_radius: Option<Real>` to
   `SimulationOptions`. Keep `None` meaning the current global behaviour, for small problems and
   existing tests.
2. Keep a growing nearest-neighbour index over conditioning plus simulated points. Use the
   existing `kiddo` dependency for planar and 3-D-embedded lat/lon (unit-sphere xyz, so chord
   ordering matches Haversine). Space-time scales time into a third axis.
3. At each step, solve a local system with the k neighbours. Use the per-target local path that
   `with_neighborhood(Some(...))` already provides, rather than constructing a new model.
4. Default the visit order to a seeded random path when `target_order` is `None`. Today it is
   input order, which biases a neighbourhood-limited SGS toward artefacts along rows.

This is independent of the FFT work and can ship first. It is a behaviour change: default results
differ from today's for the same seed, so it gets a CHANGELOG entry and the npm README notes the
new options.

## Milestones and rollout

The work ships as PRs against `main-3d`. M0 is independent and can merge first; C1 and C2 can
start alongside M1; the rest is sequential. No dates are set yet.

| Milestone | Deliverable | Exit criterion | Depends on |
| --- | --- | --- | --- |
| M0 | Neighbourhood + random path for 2-D / space-time SGS | 10,000-target 2-D SGS completes in under 30 s native; existing tests pass with `max_neighbors: None` | none |
| M1 | `fftnd.rs`, `embedding.rs`, `rustfft` dependency, bench skeleton | Naive-DFT and embedding-exactness tests pass in f32 and f64; wasm size change measured | none |
| M2 | Unconditional `CirculantSimulator` (2-D, 3-D), serial + parallel streaming, `convolve_pair` (C4) | Statistical tests pass; benchmark against 3-D SGS recorded | M1 |
| M3 | `ConditionalFftModel` (SK, OK): C1–C3, C5, C6 | All C6 tests pass in f32 and f64 | M2 (C1, C2 need only M0's branch point) |
| M4 | `src/wasm/fft_sim.rs`, TS wrappers, types, exports, vitest, smoke | `npm run verify` green; every error code covered | M3 |
| M5 | npm README section, `examples/web-2d` engine toggle, CHANGELOG, browser bench row | `examples/web-2d` builds and `npm run check` passes | M4 |

Each PR runs the full CI matrix from AGENTS.md: fmt, both clippy feature sets, tests for f32 and
f64, the wasm build and `npm run verify`. The new API is additive, so M2 to M5 need no
deprecation. M0 changes default SGS output and is called out in the CHANGELOG.

## Open questions and risks

The main risk is memory: long ranges or smooth models can force padding that does not fit in a
browser. The design handles it with a clear error rather than a silent approximation.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Padding exceeds the cell budget (long range relative to grid, Gaussian or high-ν Matérn) | `EmbeddingTooLarge` in wasm for realistic 3-D grids | Error message suggests a coarser grid, a small nugget or `allowApproximate`; `info` reports the needed size |
| Snapping bias when spacing is near the range or data are dense | Data slightly displaced; one datum per cell | Document it; report `droppedSamples` and `maxOffsetCells`; v2 local-kriging placement of off-node data |
| Global conditioning capped (5,000 native, 2,000 wasm) | Large datasets cannot condition | `TooManySamples` error pointing to SGS; v2 moving-neighbourhood correction |
| Solver inflates K's diagonal on near-duplicate data | Data honoured only approximately | `honoursDataExactly: false` in info; snapping already removes exact duplicates |
| Score-space variogram fitted in data space by mistake | Wrong variance in realizations | `scoreSill` in info; TS warns when it is far from 1 |
| Users apply it to non-stationary data (trend) | Biased realizations | README guidance; v2 universal-kriging conditioning |
| `rustfft` speed in wasm without SIMD | Timings at the slow end of the estimates | Measure in M1; check whether its wasm `simd128` path is usable in the release-wasm profile |
| M0 changes default SGS output | Downstream snapshot tests shift | CHANGELOG entry; `max_neighbors: None` and explicit `target_order` restore the old behaviour |

**Open questions**

- [ ] Should the one-shot JS helpers also offer a collected `Float64Array[]` result, or stay
  callback-only like `gaussianSimulation3D`?
- [ ] Default conditioning in score space: simple (mean 0, matches 3-D SGS) or ordinary?
- [ ] Should duplicate data in one cell be averaged instead of keeping the closest? (Averaging
  suits repeated measurements at one location; keeping the closest matches GSLib.)
- [ ] Put the module behind an `fft` cargo feature, or keep `rustfft` as a plain dependency?
  Decide from the M1 size measurement.
- [ ] Is cross-platform bit-identity (native vs wasm) a requirement for any consumer?
- [ ] Should M0 also switch the 2-D SGS default visit order to random, or leave that opt-in?
- [ ] Is intrinsic embedding for the `Power` model (Stein 2002) worth a later milestone?

## Sources

- C. R. Dietrich and G. N. Newsam, "Fast and exact simulation of stationary Gaussian processes
  through circulant embedding of the covariance matrix," SIAM J. Sci. Comput. 18(4):1088–1107,
  1997.
- Other references (Dietrich & Newsam 1996; Wood & Chan 1994; Journel & Huijbregts 1978; Stein
  2002) are cited from that paper and from memory; check details before relying on them.
