//! Cokriging: interpolation and simulation using more than one correlated variable.
//!
//! Cokriging improves estimation of a **primary** variable by exploiting its cross-correlation
//! with one or more **secondary** variables. This module is being built in tiers:
//!
//! - **Available now — collocated cokriging & cosimulation (Markov Model 1).**
//!   [`CollocatedCokrigingModel`] predicts a sparsely sampled primary from a densely sampled
//!   secondary that is known at every target, using only the primary [`VariogramModel`] and a
//!   single collocated correlation `ρ` (see [`SecondaryVariable`]). The MM1 reduction keeps a
//!   prediction as cheap as simple kriging. [`collocated_cosimulate`](crate::simulation::collocated_cosimulate)
//!   draws conditional realizations of the primary honoring the collocated secondary.
//!
//! - **Foundations for full cokriging (in place; solver to come).** The cross-covariance model
//!   [`Coregionalization`] (a Linear Model of Coregionalization with positive-semidefinite
//!   [`SillMatrix`] sill matrices over [`CorrelationBasis`] structures) and the multivariate
//!   [`MultiVariableDataset`] are the inputs a full **block cokriging** solver (ordinary/simple
//!   cokriging over an arbitrary number of variables, with cross-variograms) will consume. MM1
//!   collocated cokriging is the special case of a single intrinsic structure with a `2 × 2`
//!   sill matrix, so the two APIs describe the same family of models.
//!
//! ## Choosing an approach
//!
//! Reach for **collocated** cokriging when the secondary is exhaustively known (a covariate
//! raster, a remotely sensed field) and the Markov screening assumption is reasonable — it
//! needs no cross-variogram fitting. The full **coregionalization** path is for genuinely
//! multivariate data where you want to model the cross-structure explicitly (and, later, krige
//! or cosimulate several variables jointly, including the heterotopic case).

pub mod collocated;
pub mod coregionalization;
pub mod dataset;

pub use collocated::{CollocatedCokrigingModel, SecondaryVariable};
pub use coregionalization::{
    Coregionalization, CoregionalizationStructure, CorrelationBasis, SillMatrix,
};
pub use dataset::MultiVariableDataset;
