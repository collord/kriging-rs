//! Variogram computation, fitting, and parametric models.
//!
//! Compute an empirical variogram with [`compute_empirical_variogram`], fit a parametric model
//! with [`fit_variogram`], and build a [`VariogramModel`] (see [`VariogramType`]) for use with
//! kriging. Supported model types include spherical, exponential, Gaussian, cubic, stable, and Matérn.

pub mod directional_3d;
pub mod empirical;
pub mod experimental_3d;
pub mod fitting;
pub mod models;
pub mod nested;

pub use empirical::{
    EmpiricalEstimator, EmpiricalVariogram, PositiveReal, VariogramConfig,
    compute_empirical_variogram, compute_empirical_variogram_binomial_calibrated,
};
pub use directional_3d::{
    DirectionFilter3D, DirectionalConfig3D, compute_directional_variogram_3d,
};
pub use experimental_3d::compute_empirical_variogram_3d;
pub use fitting::{FitResult, fit_variogram};
pub use models::{VariogramModel, VariogramType};
pub use nested::NestedVariogram;
