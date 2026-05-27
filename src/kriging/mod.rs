//! Kriging models for spatial interpolation and prevalence surfaces.
//!
//! - **Ordinary kriging** ([`ordinary`]): [`crate::OrdinaryKrigingModel`] and [`crate::Prediction`] for
//!   interpolating a continuous spatial field from point observations.
//! - **Binomial kriging** ([`binomial`]): [`crate::BinomialKrigingModel`], [`crate::BinomialObservation`],
//!   and related types for prevalence surfaces. The default path is empirical-Bayes
//!   logit + ordinary kriging. [`crate::HeteroBinomialFit`] documents an experimental
//!   heteroskedastic variant.

pub mod binomial;
pub mod diagnostics;
pub mod ordinary;
pub mod simple;
pub mod solver;
pub mod universal;
