//! Solver diagnostics and 3-D prediction types.
//!
//! `Prediction3D` is the 3-D analogue of upstream's [`crate::kriging::ordinary::Prediction`],
//! extended with a condition-number proxy and a flag for whether the solver
//! used nugget inflation. 2-D `Prediction` is unchanged in v1 — the
//! const-generic unification at v2 M1 will reconcile.

use crate::Real;

/// Outcome of a single kriging system solve.
///
/// `value` and `variance` carry the same meaning as upstream's 2-D
/// `Prediction`; the additional fields surface solver behavior so callers
/// (e.g. SGS in M11) can detect when the simulation is operating near the
/// conditioning limit and react.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction3D {
    /// Predicted value at the target location.
    pub value: Real,
    /// Kriging variance at the target location.
    pub variance: Real,
    /// Condition-number proxy: `max(diag(L)) / min(diag(L))` after Cholesky
    /// on the n×n covariance block, where `L` is the lower-triangular factor.
    /// This is a coarse upper-bound-friendly indicator (it monotonically
    /// tracks `sqrt(cond(K))` but is not the true 2-norm condition number).
    /// Useful for thresholding "is this system near-singular?" but not for
    /// precise numerical analysis.
    pub condition_number: f64,
    /// `true` if the solver invoked one or more rounds of nugget inflation
    /// to obtain a non-singular factorization.
    pub used_nugget_inflation: bool,
}

/// Why the solver failed (only relevant when the prediction returns Err).
#[derive(Debug, Clone, PartialEq)]
pub enum SolverFailure {
    /// Even after `max_retries` rounds of nugget inflation the covariance
    /// block did not factor cleanly. Most commonly caused by exact
    /// duplicate sample locations or a degenerate variogram model.
    NonSingularEvenAfterInflation {
        attempts: usize,
        max_diagonal_inflation: f64,
    },
    /// The covariance block factored but its condition-number proxy exceeds
    /// the caller-provided threshold and inflation did not bring it back
    /// under bound.
    PoorlyConditioned {
        condition_number: f64,
        threshold: f64,
    },
    /// The kriging weights derived from a successful solve included NaN or
    /// Inf. Should not happen if the factorization succeeded; surfaces as
    /// a defensive check.
    NonFiniteWeights,
}

impl std::fmt::Display for SolverFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverFailure::NonSingularEvenAfterInflation {
                attempts,
                max_diagonal_inflation,
            } => write!(
                f,
                "covariance block remained singular after {attempts} \
                 nugget-inflation attempts (max diagonal inflation: {max_diagonal_inflation:e})"
            ),
            SolverFailure::PoorlyConditioned {
                condition_number,
                threshold,
            } => write!(
                f,
                "covariance block factored but condition-number proxy \
                 ({condition_number:e}) exceeds threshold ({threshold:e})"
            ),
            SolverFailure::NonFiniteWeights => write!(
                f,
                "solver produced non-finite weights after a successful factorization"
            ),
        }
    }
}

impl std::error::Error for SolverFailure {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_failure_display_messages_are_diagnostic() {
        let f1 = SolverFailure::NonSingularEvenAfterInflation {
            attempts: 5,
            max_diagonal_inflation: 1e-3,
        };
        let s = format!("{f1}");
        assert!(s.contains("5"));
        assert!(s.contains("1e-3") || s.contains("1.0e-3") || s.contains("0.001"));

        let f2 = SolverFailure::PoorlyConditioned {
            condition_number: 1e12,
            threshold: 1e10,
        };
        let s = format!("{f2}");
        assert!(s.contains("1e12") || s.contains("1.0e12"));

        let f3 = SolverFailure::NonFiniteWeights;
        assert!(format!("{f3}").contains("non-finite"));
    }
}
