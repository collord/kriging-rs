# kriging-rs (3-D fork)

Geostatistical kriging library with WASM support.

**This fork extends [`m-murphy/kriging-rs`](https://github.com/m-murphy/kriging-rs)
with first-class 3-D support**: 3-D coordinates, GSLib-style anisotropy
(rotation + ratios), 3-D directional variograms, 3-D ordinary / simple /
universal kriging, and 3-D sequential Gaussian simulation with a
streaming realization API. The 2-D path is unchanged. See the
[3-D quick example](#3-d-quick-example) and the
[3-D fork section](#3-d-fork-additions) below.

[Documentation](https://docs.rs/kriging-rs)

## 3-D quick example

```rust
use kriging_rs::{
    Anisotropy3D, Coord3D, OrdinaryKrigingModel3D, PlanarDataset3D,
    VariogramModel, VariogramType,
};

# fn main() -> Result<(), kriging_rs::KrigingError> {
let coords = vec![
    Coord3D::new(0.0, 0.0, 0.0),
    Coord3D::new(10.0, 0.0, 0.0),
    Coord3D::new(0.0, 10.0, 0.0),
    Coord3D::new(0.0, 0.0, 10.0),
];
let values = vec![1.0, 2.0, 3.0, 4.0];
let dataset = PlanarDataset3D::new(coords, values)?;
let variogram = VariogramModel::new(0.01, 2.0, 10.0, VariogramType::Exponential)?;
let model = OrdinaryKrigingModel3D::new(
    dataset,
    Anisotropy3D::identity(),
    variogram,
)?;
let prediction = model.predict(Coord3D::new(5.0, 5.0, 5.0))?;
println!("{} ± {}", prediction.value, prediction.variance.sqrt());
# Ok(())
# }
```

GSLib-style anisotropy:

```rust
# use kriging_rs::{from_gslib, GslibAnisotropy};
// Major axis along azimuth 30°, dipping 60° downward, with ratios 0.5/0.3.
let aniso = from_gslib(GslibAnisotropy {
    ang1: 30.0, ang2: 60.0, ang3: 0.0,
    anis1: 0.5, anis2: 0.3,
}).unwrap();
```

Streaming 3-D SGS:

```rust
# use kriging_rs::{Anisotropy3D, Coord3D, Grid3D, PlanarDataset3D, SgsModel3D,
#     VariogramModel, VariogramType, gaussian_simulation_3d_stream};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let dataset = PlanarDataset3D::new(
#     vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(10.0, 0.0, 0.0)],
#     vec![1.0, 2.0],
# )?;
# let variogram = VariogramModel::new(0.01, 1.0, 5.0, VariogramType::Exponential)?;
let model = SgsModel3D::new(dataset, Anisotropy3D::identity(), variogram)?;
let grid = Grid3D::new(
    50, 50, 25,
    Coord3D::new(0.0, 0.0, 0.0),
    Coord3D::new(2.0, 2.0, 1.0),
)?;

// Stream realizations; the engine reuses its buffer between calls and
// retains nothing between realizations.
gaussian_simulation_3d_stream(&model, &grid, 42, 100, |idx, grid_values| {
    // Accumulate running statistics, write to disk, hand to UI -- caller's choice.
    println!("realization {idx}: {} cells", grid_values.len());
    Ok(())
})?;
# Ok(())
# }
```

## Installation

```toml
[dependencies]
kriging-rs = "0.1"
```

Or `cargo add kriging-rs`.

## Usage

```rust
use kriging_rs::{GeoCoord, GeoDataset, OrdinaryKrigingModel, VariogramModel, VariogramType};

let coords = vec![
    GeoCoord::try_new(0.0, 0.0)?,
    GeoCoord::try_new(0.0, 1.0)?,
    GeoCoord::try_new(1.0, 0.0)?,
];
let values = vec![1.0, 2.0, 1.5];
let dataset = GeoDataset::new(coords, values)?;
let variogram = VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential)?;
let model = OrdinaryKrigingModel::new(dataset, variogram)?;
let prediction = model.predict(GeoCoord::try_new(0.3, 0.3)?)?;
println!("{:?}", prediction.value);
```

## Features

- Ordinary, simple, universal, and binomial kriging for 2-D spatial interpolation
- **Spatio-temporal kriging** (ordinary / simple / universal / binomial) over a 2-D spatial
  axis and a scalar time axis, with separable and product-sum space-time variograms
- Empirical and parametric 2-D and space-time variogram fitting
- Leave-one-out and K-fold cross-validation for every 2-D and space-time variant (continuous
  residuals plus dual-scale logit+prevalence for binomial)
- Sequential Gaussian simulation (conditional simulation) for every 2-D and space-time variant,
  deterministic for a given RNG seed
- Variogram models: spherical, exponential, Gaussian, cubic, stable, Matérn, power, hole-effect
  (stable and Matérn accept an optional shape parameter)
- Geographic coordinates with Haversine distances and planar `(x, y)` coordinates with 2-D
  anisotropy — both usable in the spatial and spatio-temporal paths
- Optional WASM bindings for browser applications
- `Real` abstraction defaults to `f32` for compute paths
- Optional cross-platform GPU capability path via `wgpu`
- **Binomial (prevalence) default:** empirical-Bayes `Beta(1,1)` (or user prior) → logit
  working values → **ordinary kriging with per-site logit observation variance** (calibrated
  binomial) on the covariance diagonal, then `logistic` to prevalence; *not* a full
  binomial-likelihood field model. Build diagnostics are always returned on the Rust side
  ([`BinomialBuildNotes`]) and exposed in WASM as `getBuildNotes()`. See
  [benches/BROWSER_BENCHMARKS.md](benches/BROWSER_BENCHMARKS.md) for a large
  browser-representative prediction workload

Build with `--features wasm` for browser; see below for GPU.

## 3-D fork additions

This fork adds first-class 3-D support as parallel modules alongside
upstream's 2-D path. The 2-D code is unchanged in v1; a const-generic
unification that subsumes both is planned for v2.

**New types**:

- `Coord3D` — right-handed Cartesian with z-positive-up.
- `PlanarDataset3D` — coord/value dataset with shape validation.
- `Anisotropy3D` — internal rotation-matrix + diagonal-stretch
  representation. GSLib I/O via `from_gslib` /  `to_gslib`; all GSLib
  semantics (clockwise-from-north azimuth, dip-positive-down,
  truncated-π DEG2RAD) live exclusively in `interop::gslib_anisotropy`.
- `Grid3D` — regular 3-D simulation grid.
- `NormalScoreTransform` — empirical-CDF NST for SGS pre/post-processing.
- `KdTree3D` / `MutableKdTree3D` — anisotropy-aware kd-trees over
  `kiddo`. Coordinates are pre-transformed into the anisotropy
  ellipsoid's principal frame at construction so the tree's Euclidean
  distance *is* the anisotropic distance.

**New kriging models** (all share the same robust solver and optional
`Neighborhood3D` filter):

- `OrdinaryKrigingModel3D`
- `SimpleKrigingModel3D`
- `UniversalKrigingModel3D` (linear trend basis `[1, x, y, z]` in v1;
  quadratic and arbitrary callbacks deferred to v2)

**SGS**:

- `gaussian_simulation_3d_stream(...)` — streaming realization API.
  Each realization is yielded to a closure and **discarded**; engine
  retains nothing between calls. Native callers can use
  `gaussian_simulation_3d_stream_parallel(...)` for rayon-based
  across-realization parallelism.
- `SgsOutputSpace::ScoreSpace` to skip NST back-transform when the
  caller needs raw normal scores (e.g. for validation against
  score-space OK predictions).

**Cross-validation**:

- `cv_3d::leave_one_out_ordinary_3d` / `leave_one_out_simple_3d` /
  `leave_one_out_universal_3d_linear`. Reuses upstream's metric-
  agnostic `CvResidual` / `CvSummary` types.

**Variogram**:

- `compute_empirical_variogram_3d` — omnidirectional, parallelized.
- `compute_directional_variogram_3d` + `DirectionFilter3D` — GSLib
  `gamv`-compatible directional with cone + bandwidth filter and
  lag-centred binning. Bitwise pair-set parity with `gamv`.
- Existing 2-D variogram model and fitting code (spherical,
  exponential, Gaussian, etc.) work unchanged on 3-D empirical output.

**WASM**:

- `WasmOrdinaryKriging3D`, `WasmSimpleKriging3D`,
  `WasmUniversalKriging3D` mirror upstream's `WasmOrdinaryKriging`
  shape — `fromArrays(...)` static constructor taking flat
  `Float64Array`s, `predict` / `predictBatch` methods returning
  `Object { value(s), variance(s), conditionNumber(s),
  usedNuggetInflation }`.
- `gaussianSimulation3D(...)` — streaming SGS taking a JS callback;
  the callback fires once per realization in index order with a
  fresh `Float64Array` of the grid values.

GSLib anisotropy parameters `(ang1, ang2, ang3, anis1, anis2)` are
the boundary form for all wasm constructors; pass `(0, 0, 0, 1, 1)`
for identity. See [`npm/kriging-rs-wasm/scripts/smoke-3d.mjs`](npm/kriging-rs-wasm/scripts/smoke-3d.mjs)
for an end-to-end browser-path example.

### Parity with reference implementations

Validation fixtures live under `tests/parity_3d/fixtures/`:

| Fixture | Source | Validates |
|---|---|---|
| `setrot_reference/` | GSLib `dsetrot` Fortran | `Anisotropy3D` matrix matches GSLib bitwise (modulo truncated-π) |
| `skgstat_3d/` | scikit-gstat 1.0.23 | Omnidirectional variogram pair counts + γ̂ |
| `gamv_3d/` | GSLib `gamv` binary | Directional variogram pair-set bitwise parity |
| `textbook_ok3d/` `textbook_sk3d/` `textbook_uk3d/` | Hand-rolled numpy OK/SK/UK | 1000-target algorithmic-tier kriging parity |

### Performance

3-D kriging predictions are parallel across targets on native
builds (rayon). 3-D SGS is sequential within a realization
(algorithmic constraint) and parallel across realizations via
`gaussian_simulation_3d_stream_parallel`.

Measured single-realization SGS throughput on x86_64 macOS:

| Mode | Throughput |
|---|---|
| Native release | ~80 K cells/sec |
| WASM (single-threaded) | ~38 K cells/sec |
| Native parallel (4 cores, across realizations) | ~310 K cells/sec |

See `benches/bench_3d_kdtree.rs` and `benches/bench_3d_sgs_streaming.rs`.
The [`docs/m0-findings.md`](../docs/m0-findings.md) file in the
workspace tracks all per-milestone findings including the empirical
feasibility-at-N analysis.

### Deviations from `m-murphy/kriging-rs`

The 3-D fork is **additive** — every 2-D API is preserved. New 3-D
modules live in `src/*_3d.rs` files (`coord_3d.rs`, `anisotropy_3d.rs`,
`planar_dataset_3d.rs`, `kriging/ordinary_3d.rs`, etc.). A v2
const-generic unification will merge them with the 2-D paths; v1
keeps them parallel for review/PR simplicity.

**Conventions worth knowing**:

- 3-D kriging variance follows upstream's "latent field" convention:
  `σ² = C(0) − λ·k₀ − μ` where `C(0)` is the partial sill (without
  the nugget). Add the nugget back to recover textbook "standard"
  kriging variance.
- The robust 3-D solver uses regular Cholesky on the n×n covariance
  block (not LU like upstream's 2-D) with nugget-inflation retry on
  factorization failure. Pivoted Cholesky was specified in v3 but
  not available in nalgebra; nugget inflation rescues the cases
  pivoting would have detected.

## Repository layout

Root is the Rust crate. `npm/kriging-rs-wasm/` is the TypeScript/WASM npm package. `www/` is a browser demo (see [www/README.md](www/README.md)).

## WASM and npm package

Build WASM:

```bash
wasm-pack build --target web -- --features wasm
```

The TypeScript/npm facade lives in `npm/kriging-rs-wasm`. See that package’s README for install, verify, and batch/typed-array APIs.

Browser demo: [www/README.md](www/README.md).

## Development

Install [pre-commit](https://pre-commit.com/) and run `pre-commit install` so fmt and clippy run before each commit and match CI.

```bash
cargo test
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

## GPU

Features: `gpu` (async WebGPU via `wgpu` on native + web, including GPU-assisted RHS covariance for batch prediction) and `gpu-blocking` (native blocking helpers via `pollster`).

```bash
cargo run --example gpu_probe --features "gpu,gpu-blocking"
```

GPU batch prediction APIs are on `OrdinaryKrigingModel` / `BinomialKrigingModel` (Rust) and the WASM types (with `wasm,gpu`). See examples and the npm README for details.

## Performance

Run `cargo bench` for current numbers; see [bench-results/README.md](bench-results/README.md) for logging and comparison. A **browser-oriented** (large grid, mixed trial counts) binomial *prediction* benchmark and workload description is in [benches/BROWSER_BENCHMARKS.md](benches/BROWSER_BENCHMARKS.md) (`bench_binomial_browser_representative`).

## License

Licensed under MIT.
