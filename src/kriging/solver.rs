//! Robust ordinary-kriging solver.
//!
//! Solves the bordered OK system
//!
//! ```text
//!   [ K   e ] [ λ ]   [ k0 ]
//!   [ eᵀ  0 ] [ μ ] = [ 1  ]
//! ```
//!
//! via Schur complement on the `n × n` covariance block `K`. `K` is
//! symmetric and (in the absence of duplicate samples) strictly positive
//! definite, so Cholesky is the natural factorization. The Lagrangian
//! border is handled outside the factorization with two right-hand-sides
//! reusing the same `L`.
//!
//! ## Robustness strategy (v3 §F)
//!
//! 1. Try Cholesky on the n × n covariance block.
//! 2. If the factorization fails or the diagonal-ratio condition proxy
//!    exceeds the caller-provided threshold, **inflate the diagonal nugget**
//!    by a factor `(1 + ε)` and retry. `ε` starts at `1e-8` and increases
//!    geometrically until either a successful factorization is found or
//!    `max_retries` rounds have been spent.
//! 3. On exhaustion, return [`SolverFailure::NonSingularEvenAfterInflation`].
//!
//! ## Deliberate deviation from v3
//!
//! v3 §F specifies *pivoted* Cholesky (LAPACK `dpstrf`). nalgebra 0.34
//! ships only plain Cholesky, and adding an LAPACK dependency breaks
//! WASM portability. For v1 we use plain Cholesky and detect rank
//! deficiency by `Cholesky::new` returning `None`. The information lost
//! by skipping pivoting is "which row/column caused the failure" — we
//! don't surface that to callers. The smallest intervention that meets
//! v3 §F's stated robustness behavior; pivoting can be added later if a
//! real workload demands it.

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::kriging::diagnostics::{Prediction3D, SolverFailure};
use crate::variogram::models::VariogramModel;

/// Tuning knobs for the robust solver.
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    /// Maximum allowable diagonal-ratio condition-number proxy. If the
    /// initial Cholesky factors but the proxy exceeds this, the solver
    /// attempts nugget inflation. The default of `1e10` is conservative
    /// for f64; tighten on noisy datasets where you'd rather error than
    /// trust a near-singular solve.
    pub condition_threshold: f64,
    /// Maximum number of nugget-inflation retries. The default of `5`
    /// gives a final inflation factor of `(1 + 1e-8 · 10^4) ≈ 1.0001`,
    /// which is enough to rescue almost any merely-ill-conditioned system
    /// without distorting the variogram model.
    pub max_retries: usize,
    /// Initial diagonal-inflation epsilon, applied as `K' = K · (1 + ε)`
    /// on the diagonal entries only. Default `1e-8`.
    pub initial_epsilon: f64,
    /// Multiplier on `ε` between retries. Default `10.0`.
    pub epsilon_scale: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            condition_threshold: 1e10,
            max_retries: 5,
            initial_epsilon: 1e-8,
            epsilon_scale: 10.0,
        }
    }
}

/// One pass of "try Cholesky on the n × n covariance block." Returns the
/// factorization and the diagonal-ratio condition proxy on success.
fn try_factor(k_block: &DMatrix<f64>) -> Option<(Cholesky<f64, Dyn>, f64)> {
    let cholesky = Cholesky::new(k_block.clone())?;
    let l = cholesky.l_dirty();
    let n = l.nrows();
    if n == 0 {
        return Some((cholesky, 1.0));
    }
    let mut max_diag = f64::NEG_INFINITY;
    let mut min_diag = f64::INFINITY;
    for i in 0..n {
        let d = l[(i, i)].abs();
        if d > max_diag {
            max_diag = d;
        }
        if d < min_diag {
            min_diag = d;
        }
    }
    let cond_proxy = if min_diag > 0.0 {
        max_diag / min_diag
    } else {
        f64::INFINITY
    };
    Some((cholesky, cond_proxy))
}

/// Build the n × n covariance block for a set of (anisotropy-aware) sample
/// locations, evaluated through the given variogram model.
fn build_covariance_block(
    samples: &[Coord3D],
    anisotropy: &Anisotropy3D,
    variogram: &VariogramModel,
) -> DMatrix<f64> {
    let n = samples.len();
    let mut k = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let d = anisotropy.anisotropic_distance(samples[i], samples[j]);
            let c = variogram.covariance(d as Real) as f64;
            k[(i, j)] = c;
            if i != j {
                k[(j, i)] = c;
            }
        }
    }
    k
}

/// Solve a single ordinary-kriging system at the given target.
///
/// Builds the covariance block, factors with Cholesky (retrying with
/// inflated nugget on failure or poor conditioning), and returns the
/// predicted value, kriging variance, and solver diagnostics. The
/// `Prediction3D` returned describes the full outcome; failures surface
/// as `Err(SolverFailure)`.
pub fn solve_ordinary_kriging_3d(
    samples: &[Coord3D],
    values: &[Real],
    target: Coord3D,
    anisotropy: &Anisotropy3D,
    variogram: &VariogramModel,
    config: &SolverConfig,
) -> Result<Prediction3D, SolverFailure> {
    assert_eq!(
        samples.len(),
        values.len(),
        "samples and values must have matching length"
    );
    let n = samples.len();

    // Build invariant pieces of the system once.
    let mut k_block = build_covariance_block(samples, anisotropy, variogram);
    let k0: DVector<f64> = DVector::from_iterator(
        n,
        samples
            .iter()
            .map(|s| variogram.covariance(anisotropy.anisotropic_distance(*s, target) as Real) as f64),
    );
    let ones: DVector<f64> = DVector::from_element(n, 1.0);
    let sill_at_zero = variogram.covariance(0.0) as f64;

    // Attempt 0: bare Cholesky.
    let mut attempt = 0usize;
    let mut current_epsilon = 0.0_f64;
    let mut used_inflation = false;
    let (chol, cond_proxy) = loop {
        match try_factor(&k_block) {
            Some((c, cp)) if cp <= config.condition_threshold => break (c, cp),
            Some((_, cp)) if attempt >= config.max_retries => {
                return Err(SolverFailure::PoorlyConditioned {
                    condition_number: cp,
                    threshold: config.condition_threshold,
                });
            }
            None if attempt >= config.max_retries => {
                return Err(SolverFailure::NonSingularEvenAfterInflation {
                    attempts: attempt,
                    max_diagonal_inflation: current_epsilon,
                });
            }
            _ => {
                // Cholesky failed or condition exceeded threshold; inflate.
                used_inflation = true;
                let next_epsilon = if attempt == 0 {
                    config.initial_epsilon
                } else {
                    current_epsilon * config.epsilon_scale
                };
                // Inflate diagonal entries by (1 + next_epsilon).
                for i in 0..n {
                    k_block[(i, i)] *= 1.0 + next_epsilon;
                }
                current_epsilon = next_epsilon;
                attempt += 1;
            }
        }
    };

    // Schur-complement solves: u = K^-1 e, v = K^-1 k0.
    let u = chol.solve(&ones);
    let v = chol.solve(&k0);

    let e_dot_u: f64 = ones.dot(&u);
    if e_dot_u.abs() < f64::EPSILON {
        return Err(SolverFailure::NonFiniteWeights);
    }
    let e_dot_v: f64 = ones.dot(&v);
    let mu = (e_dot_v - 1.0) / e_dot_u;
    let lambda = &v - &u * mu;

    // Check finiteness defensively.
    if !mu.is_finite() || lambda.iter().any(|x| !x.is_finite()) {
        return Err(SolverFailure::NonFiniteWeights);
    }

    // Predicted value: λ · z.
    let value: f64 = lambda
        .iter()
        .zip(values.iter())
        .map(|(l, z)| l * (*z as f64))
        .sum();

    // Kriging variance: σ² = C(0) - λ · k0 - μ.
    // Pin to >= 0 because rounding can push slightly negative.
    let variance = (sill_at_zero - lambda.dot(&k0) - mu).max(0.0);

    Ok(Prediction3D {
        value: value as Real,
        variance: variance as Real,
        condition_number: cond_proxy,
        used_nugget_inflation: used_inflation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn isotropic_exponential() -> VariogramModel {
        VariogramModel::new(0.0, 1.0, 10.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn solves_well_conditioned_2x2_system_and_reports_low_condition() {
        // Two samples, well-separated, with the target between them.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
        ];
        let values = vec![1.0 as Real, 3.0 as Real];
        let target = Coord3D::new(5.0, 0.0, 0.0);
        let pred = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .expect("well-conditioned system should solve");
        // OK with a symmetric configuration: weights are 0.5/0.5, so the
        // prediction is the average of the two values = 2.0.
        assert_relative_eq!(pred.value as f64, 2.0, epsilon = 1e-4);
        assert!(pred.variance >= 0.0, "variance must be non-negative");
        assert!(
            pred.condition_number < 100.0,
            "well-conditioned 2x2 should have a small condition proxy, got {}",
            pred.condition_number
        );
        assert!(
            !pred.used_nugget_inflation,
            "well-conditioned system should not need inflation"
        );
    }

    #[test]
    fn exact_match_at_sample_recovers_sample_value() {
        // Target equals one of the sample locations -> kriging should
        // interpolate exactly to that sample's value (modulo nugget,
        // which we set to zero).
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
        ];
        let values = vec![1.0 as Real, 3.0 as Real, 5.0 as Real];
        let target = samples[1]; // exact match with second sample
        let pred = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .expect("exact-match system should solve");
        assert_relative_eq!(pred.value as f64, 3.0, epsilon = 1e-3);
        // Variance at a sample location should be ~0 (no nugget).
        assert!(
            (pred.variance as f64).abs() < 1e-3,
            "variance at sample should be ~0, got {}",
            pred.variance
        );
    }

    #[test]
    fn duplicate_samples_trigger_nugget_inflation() {
        // Two samples at the same location with different values: K has
        // a duplicated row/column, making it exactly singular. Nugget
        // inflation should rescue it.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
        ];
        let values = vec![1.0 as Real, 1.5 as Real, 3.0 as Real];
        let target = Coord3D::new(5.0, 0.0, 0.0);
        let pred = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .expect("duplicate samples should be rescued by nugget inflation");
        assert!(
            pred.used_nugget_inflation,
            "duplicate samples should require nugget inflation"
        );
        assert!(pred.value.is_finite(), "value should be finite");
    }

    #[test]
    fn solver_returns_error_when_no_retries_are_allowed() {
        // Singular K with `max_retries = 0`: bare Cholesky fails and the
        // solver has no budget to inflate, so it must surface the failure.
        // (With even one retry available, diagonal inflation rescues this
        // case; see `duplicate_samples_trigger_nugget_inflation`. The
        // point of this test is the cap, not the singularity.)
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(0.0, 0.0, 0.0),
        ];
        let values = vec![1.0 as Real, 2.0 as Real, 3.0 as Real];
        let target = Coord3D::new(1.0, 1.0, 1.0);
        let no_retry_config = SolverConfig {
            max_retries: 0,
            ..SolverConfig::default()
        };
        let result = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &no_retry_config,
        );
        assert!(
            matches!(
                result,
                Err(SolverFailure::NonSingularEvenAfterInflation { .. })
                    | Err(SolverFailure::PoorlyConditioned { .. })
            ),
            "expected solver failure with max_retries=0, got {:?}",
            result,
        );
    }

    #[test]
    fn anisotropy_changes_prediction() {
        // Same samples, same values, same target — but two different
        // anisotropies should produce different predictions because the
        // covariance distances change.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(0.0, 0.0, 10.0),
            Coord3D::new(10.0, 0.0, 0.0),
        ];
        let values = vec![1.0 as Real, 5.0 as Real, 3.0 as Real];
        let target = Coord3D::new(0.0, 0.0, 5.0);

        let iso = Anisotropy3D::identity();
        let z_stretched = Anisotropy3D::from_rotation_matrix(
            nalgebra::Matrix3::identity(),
            nalgebra::Vector3::new(1.0, 1.0, 10.0),
        )
        .unwrap();

        let pred_iso = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &iso,
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        let pred_aniso = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &z_stretched,
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        assert!(
            (pred_iso.value as f64 - pred_aniso.value as f64).abs() > 0.01,
            "anisotropy should meaningfully change the prediction: iso={}, aniso={}",
            pred_iso.value,
            pred_aniso.value,
        );
    }

    #[test]
    fn condition_threshold_respected_when_inflation_cannot_recover() {
        // Build a system that's well-conditioned but our threshold is
        // pathologically tight; solver should err with PoorlyConditioned
        // rather than silently inflate.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(0.001, 0.0, 0.0), // near-collinear with first
            Coord3D::new(10.0, 0.0, 0.0),
        ];
        let values = vec![1.0 as Real, 1.5 as Real, 3.0 as Real];
        let target = Coord3D::new(5.0, 0.0, 0.0);
        let tight_config = SolverConfig {
            condition_threshold: 1.5, // ridiculous
            max_retries: 0,
            ..SolverConfig::default()
        };
        let result = solve_ordinary_kriging_3d(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &tight_config,
        );
        assert!(
            matches!(result, Err(SolverFailure::PoorlyConditioned { .. })),
            "expected PoorlyConditioned with tight threshold, got {:?}",
            result,
        );
    }
}
