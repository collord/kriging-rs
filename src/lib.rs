//! Geostatistical kriging library with ordinary and binomial kriging, variogram fitting,
//! and optional WASM and GPU support.
//!
//! This crate provides spatial interpolation via ordinary kriging and prevalence-surface
//! estimation via binomial kriging. **Default binomial kriging** is empirical-Bayes
//! (Beta-prior) logit on the link, **calibrated** ordinary kriging with per-site logit
//! observation variance on the covariance diagonal, and logistic back to prevalence; see
//! [`BinomialKrigingModel`] and [`BinomialBuildNotes`] (not a full binomial-likelihood
//! spatial model). The crate also includes empirical variogram computation, parametric model
//! fitting, and Haversine-based geographic coordinates. Build with `wasm` for browser
//! bindings or `gpu` for GPU-accelerated batch prediction.
//!
//! # Scope
//!
//! - **Geographic by default.** Coordinates are `(latitude, longitude)` in degrees and the
//!   default distance is Haversine (great-circle, kilometers). This works globally, including
//!   near the poles and across the antimeridian, to the precision allowed by the selected
//!   floating-point type (see the `f64` feature below for double precision).
//! - **2-D spatial fields and 2+1-D spatio-temporal fields.** The root `kriging` module
//!   covers surfaces that are functions of two spatial coordinates; the [`spacetime`] module
//!   adds an additional scalar time axis with the same four kriging families (ordinary,
//!   simple, universal, binomial). There is no 3-D (volumetric) kriging in this crate.
//! - **Planar data is supported via [`projected`].** `ProjectedCoord` + Euclidean distance +
//!   [`Anisotropy2D`] let you krige `(x, y)` data in any linear unit (meters, grid cells,
//!   pixels). Convert between lat/lon and planar with [`ProjectedCoord::equirectangular`]
//!   for small areas where the sphere-vs-plane distortion is negligible. The spatio-temporal
//!   models are generic over the spatial geometry and accept both geographic and projected
//!   coordinates via the [`spacetime::SpatialMetric`] trait.
//!
//! # WASM initialization
//!
//! When using the `kriging-rs-wasm` npm wrapper, you must call and `await` `init()` once
//! before constructing any model. The wrapper guards against the JS-glue-before-WASM race
//! (it only marks the module ready after the underlying instantiation resolves) — but if
//! you build your own bindings, make sure the WebAssembly instance is fully available
//! before calling constructors, otherwise you will see cryptic `TypeError`s from
//! `wasm_bindgen`.
//!
//! # Quick example
//!
//! ```rust
//! use kriging_rs::{GeoCoord, GeoDataset, OrdinaryKrigingModel, VariogramModel, VariogramType};
//!
//! # fn main() -> Result<(), kriging_rs::KrigingError> {
//! let coords = vec![
//!     GeoCoord::try_new(0.0, 0.0)?,
//!     GeoCoord::try_new(0.0, 1.0)?,
//!     GeoCoord::try_new(1.0, 0.0)?,
//! ];
//! let values = vec![1.0, 2.0, 1.5];
//! let dataset = GeoDataset::new(coords, values)?;
//! let variogram = VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential)?;
//! let model = OrdinaryKrigingModel::new(dataset, variogram)?;
//! let prediction = model.predict(GeoCoord::try_new(0.3, 0.3)?)?;
//! # let _ = prediction.value;
//! # Ok(())
//! # }
//! ```
//!
//! # Module organization
//!
//! - [`kriging`] — Ordinary kriging ([`OrdinaryKrigingModel`], [`Prediction`]) and binomial
//!   kriging ([`BinomialKrigingModel`], [`BinomialObservation`], etc.) for spatial interpolation
//!   and prevalence surfaces.
//! - [`variogram`] — Empirical variogram ([`compute_empirical_variogram`]), fitting
//!   ([`fit_variogram`]), and parametric models ([`VariogramModel`], [`VariogramType`]).
//! - [`spacetime`] — Spatio-temporal kriging (ordinary, simple, universal, binomial) with
//!   separable and product-sum space-time variograms, empirical + parametric fitting, and
//!   generic spatial metrics (geographic or projected).
//! - [`distance`] — [`GeoCoord`] and Haversine distance.
//! - [`geo_dataset`] — Coordinate–value datasets ([`GeoDataset`]).
//! - [`error`] — [`KrigingError`].
//! - `wasm` (optional, `wasm` feature) — Browser-facing WASM bindings.
//! - `gpu` (optional, `gpu` feature) — GPU backend and batch covariance helpers.
//!
//! # Features
//!
//! Default build has no WASM or GPU. Enable with:
//!
//! - `wasm` — WASM bindings and browser support.
//! - `gpu` — WebGPU-based batch prediction (native and web).
//! - `gpu-blocking` — Blocking GPU helpers on native (includes `gpu`).

/// Floating-point type used for coordinates, values, and variogram parameters.
///
/// Defaults to `f32`. Enable the `f64` Cargo feature to switch to double precision at the
/// cost of ~2x memory and slightly slower math; useful for dense station layouts where the
/// `f32` kriging matrix is close to numerically singular. Incompatible with the `gpu`
/// feature because WGSL shaders hard-code `f32`.
#[cfg(not(feature = "f64"))]
pub type Real = f32;
#[cfg(feature = "f64")]
pub type Real = f64;

#[cfg(all(feature = "f64", feature = "gpu"))]
compile_error!(
    "features `f64` and `gpu` are mutually exclusive: WGSL shaders currently only support f32"
);

pub mod aggregate;
pub mod anisotropy_3d;
pub mod coord_3d;
pub mod cv;
pub mod distance;
pub mod error;
pub mod geo_dataset;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod interop;
pub mod kriging;
/// Dense linear-system helpers used internally by kriging models. Not part of the stable
/// public API — use the higher-level `*KrigingModel` types to predict values; direct solver
/// access is a crate-internal detail that may be removed in a future release.
pub(crate) mod matrix;
pub mod neighborhood;
pub mod planar_dataset_3d;
pub mod projected;
pub mod simulation;
pub mod spacetime;
pub mod utils;

pub use utils::{Probability, clamp_probability, logistic, logit, logit_clamped};
pub mod variogram;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use anisotropy_3d::Anisotropy3D;
pub use interop::gslib_anisotropy::{GslibAnisotropy, from_gslib, to_gslib};
pub use interop::gslib_directional::GslibDirection;
pub use variogram::{
    DirectionFilter3D, DirectionalConfig3D, compute_directional_variogram_3d,
};
pub use coord_3d::{Coord3D, Orientation, euclidean_distance_3d, euclidean_distance_3d_squared};
pub use distance::GeoCoord;
pub use error::KrigingError;
pub use geo_dataset::GeoDataset;
pub use planar_dataset_3d::PlanarDataset3D;
#[cfg(feature = "gpu")]
pub use gpu::{GpuBackend, GpuSupport, build_rhs_covariances_gpu, detect_gpu_support, gpu_square};
pub use kriging::binomial::{
    BINOMIAL_CALIBRATION_VERSION, BinomialBuildNotes, BinomialCalibratedResult, BinomialFit,
    BinomialKrigingModel, BinomialObservation, BinomialPrediction, BinomialPrior,
    HeteroskedasticBinomialConfig, build_binomial_observations_dropping_zero_trials,
    indices_of_zero_trials, logit_observation_variance_empirical_bayes,
};
pub use kriging::diagnostics::{Prediction3D, SolverFailure};
pub use kriging::ordinary::{Neighborhood, OrdinaryKrigingModel, Prediction};
pub use kriging::ordinary_3d::{Neighborhood3D, OrdinaryKrigingModel3D};
pub use kriging::simple_3d::SimpleKrigingModel3D;
pub use kriging::universal_3d::{Trend3D, UniversalKrigingModel3D};
pub use kriging::simple::SimpleKrigingModel;
pub use kriging::solver::{
    SolverConfig, solve_ordinary_kriging_3d, solve_simple_kriging_3d,
    solve_universal_kriging_3d_linear,
};
pub use kriging::universal::{UniversalKrigingModel, UniversalTrend};
pub use projected::{
    Anisotropy2D, BinomialProjectedKrigingModel, DirectionalConfig, ProjectedBinomialFit,
    ProjectedBinomialObservation, ProjectedCoord, ProjectedDataset, ProjectedKrigingModel,
    compute_directional_empirical_variogram, euclidean_distance, euclidean_distance_squared,
};
pub use variogram::fitting::{FitResult, fit_variogram};
pub use variogram::models::{VariogramModel, VariogramType};
pub use variogram::nested::NestedVariogram;
pub use variogram::{
    EmpiricalEstimator, PositiveReal, VariogramConfig, compute_empirical_variogram,
    compute_empirical_variogram_3d, compute_empirical_variogram_binomial_calibrated,
};
