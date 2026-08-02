//! Variogram computation, fitting, and parametric models.
//!
//! Compute an empirical variogram with [`compute_empirical_variogram`], fit a parametric model
//! with [`fit_variogram`], and build a [`VariogramModel`] (see [`VariogramType`]) for use with
//! kriging. Supported model types include spherical, exponential, Gaussian, cubic, stable, Matérn,
//! power, hole-effect, and confluent hypergeometric (a two-shape Matérn generalization with
//! polynomial tails).

pub mod directional_3d;
pub mod empirical;
pub mod experimental_3d;
pub mod fitting;
pub mod models;
pub mod nested;
pub mod spec;

pub use directional_3d::{
    DirectionFilter3D, DirectionalConfig3D, compute_directional_variogram_3d,
};
pub use empirical::{
    EmpiricalEstimator, EmpiricalVariogram, PositiveReal, VariogramConfig,
    compute_empirical_variogram, compute_empirical_variogram_binomial_calibrated,
};
pub use experimental_3d::compute_empirical_variogram_3d;
pub use fitting::{
    FitResult, Spherical3DJointFit, fit_spherical_3d_joint, fit_spherical_3d_two_stage,
    fit_spherical_3d_with_fixed_nugget, fit_variogram,
};
pub use models::{VariogramModel, VariogramType};
pub use nested::NestedVariogram;
pub use spec::VariogramSpec;
