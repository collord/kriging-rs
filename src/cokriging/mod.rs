//! Cokriging: interpolation and simulation using more than one correlated variable.
//!
//! Cokriging improves estimation of a **primary** variable by exploiting its cross-correlation
//! with one or more **secondary** variables. Two complementary approaches are available:
//!
//! - **Collocated cokriging & cosimulation (Markov Model 1).**
//!   [`CollocatedCokrigingModel`] predicts a sparsely sampled primary from a densely sampled
//!   secondary that is known at every target, using only the primary [`VariogramModel`] and a
//!   single collocated correlation `ρ` (see [`SecondaryVariable`]). The MM1 reduction keeps a
//!   prediction as cheap as simple kriging. [`collocated_cosimulate`](crate::simulation::collocated_cosimulate)
//!   draws conditional realizations of the primary honoring the collocated secondary. Reach for
//!   this when the secondary is exhaustively known and the Markov screening assumption is
//!   reasonable — it needs no cross-variogram fitting.
//!
//! - **Full block cokriging over a Linear Model of Coregionalization.** [`CokrigingModel`]
//!   performs simple or ordinary cokriging of any target variable using **all** variables'
//!   data, driven by a [`Coregionalization`] — a sum of [`CorrelationBasis`] structures each
//!   weighted by a positive-semidefinite [`SillMatrix`], giving every cross-covariance
//!   `C_ij(h)`. The block system is factorized once and reused across target variables and
//!   locations. It reduces exactly to univariate kriging when there is one variable, and ignores
//!   an uncorrelated secondary. Variables may share locations (isotopic —
//!   [`MultiVariableDataset`], via [`CokrigingModel::new`]) or be sampled at *different* places
//!   (heterotopic — [`MultiVariableSamples`], via [`CokrigingModel::new_heterotopic`]); the
//!   latter is how a sparse primary borrows strength from a dense secondary. MM1 collocated
//!   cokriging is the special case of a single intrinsic structure with a `2 × 2` sill matrix,
//!   so the two approaches describe one family of models.
//!
//! - **Multivariate sequential cosimulation.**
//!   [`cosimulate`](crate::simulation::cosimulate) draws joint conditional realizations of all
//!   variables honoring the full LMC — simulating each variable at each node from its cokriging
//!   distribution against the data and everything simulated so far (heterotopic during a node's
//!   sweep, which is why it builds on [`CokrigingModel::new_heterotopic`]).
//!
//! - **LMC fitting.** [`fit_lmc`] chooses a coregionalization's sill matrices to match empirical
//!   cross-variograms ([`compute_empirical_cross_variogram`]) by the Goulard–Voltz algorithm —
//!   a cyclic weighted least-squares update with a positive-semidefinite projection each sweep,
//!   so the result is admissible by construction. You choose the basic structures (ranges); it
//!   fits their sills.

pub mod collocated;
pub mod coregionalization;
pub mod dataset;
pub mod fitting;
pub mod model;

pub use collocated::{CollocatedCokrigingModel, SecondaryVariable};
pub use coregionalization::{
    Coregionalization, CoregionalizationStructure, CorrelationBasis, SillMatrix,
};
pub use dataset::{MultiVariableDataset, MultiVariableSamples};
pub use fitting::{
    EmpiricalCrossVariogram, LmcFit, LmcFitOptions, compute_empirical_cross_variogram, fit_lmc,
};
pub use model::{CokrigingKind, CokrigingModel};
