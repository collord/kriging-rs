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
//!   `C_ij(h)`. The `p·n × p·n` block system is factorized once and reused across target
//!   variables and locations. It reduces exactly to univariate kriging when there is one
//!   variable, and ignores an uncorrelated secondary. Inputs come as a [`MultiVariableDataset`]
//!   (isotopic — all variables sampled at the same locations). MM1 collocated cokriging is the
//!   special case of a single intrinsic structure with a `2 × 2` sill matrix, so the two APIs
//!   describe one family of models.
//!
//! ## Not yet
//!
//! - **Heterotopic data** (variables sampled at *different* locations) — the block assembly is
//!   currently square (isotopic). Heterotopic support is the enabler for the next item.
//! - **Multivariate sequential cosimulation** (joint SGS over several variables) — needs the
//!   heterotopic path, because simulated nodes are heterotopic during a within-node sweep.
//! - **LMC auto-fitting** (e.g. Goulard–Voltz) — for now, supply the [`Coregionalization`]
//!   directly; its sill matrices are admissibility-checked at construction.

pub mod collocated;
pub mod coregionalization;
pub mod dataset;
pub mod model;

pub use collocated::{CollocatedCokrigingModel, SecondaryVariable};
pub use coregionalization::{
    Coregionalization, CoregionalizationStructure, CorrelationBasis, SillMatrix,
};
pub use dataset::MultiVariableDataset;
pub use model::{CokrigingKind, CokrigingModel};
