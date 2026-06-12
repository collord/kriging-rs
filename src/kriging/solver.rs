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

use nalgebra::{Cholesky, DMatrix, DVector, Dyn, Matrix4, Vector4};

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

/// Factor the n×n covariance block with Cholesky, retrying with diagonal
/// nugget inflation on failure or poor conditioning. Returns the
/// factorization, the condition-number proxy, and whether inflation was
/// used. Shared across OK, SK, and UK solvers.
fn factor_with_retries(
    k_block: &mut DMatrix<f64>,
    config: &SolverConfig,
) -> Result<(Cholesky<f64, Dyn>, f64, bool), SolverFailure> {
    let n = k_block.nrows();
    let mut attempt = 0usize;
    let mut current_epsilon = 0.0_f64;
    let mut used_inflation = false;
    loop {
        match try_factor(k_block) {
            Some((c, cp)) if cp <= config.condition_threshold => {
                return Ok((c, cp, used_inflation));
            }
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
                used_inflation = true;
                let next_epsilon = if attempt == 0 {
                    config.initial_epsilon
                } else {
                    current_epsilon * config.epsilon_scale
                };
                for i in 0..n {
                    k_block[(i, i)] *= 1.0 + next_epsilon;
                }
                current_epsilon = next_epsilon;
                attempt += 1;
            }
        }
    }
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
        samples.iter().map(|s| {
            variogram.covariance(anisotropy.anisotropic_distance(*s, target) as Real) as f64
        }),
    );
    let ones: DVector<f64> = DVector::from_element(n, 1.0);
    let sill_at_zero = variogram.covariance(0.0) as f64;

    let (chol, cond_proxy, used_inflation) = factor_with_retries(&mut k_block, config)?;

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

/// Solve a single simple-kriging system at the given target.
///
/// SK assumes a known global mean `m` and **does not enforce
/// `Σλ = 1`** — there's no Lagrangian. The system is `K λ = k₀`, the
/// predicted value is `m + λᵀ(z − m·1)`, and the kriging variance is
/// `C(0) − λᵀk₀`.
///
/// Same robustness strategy as OK (Cholesky on K with nugget-inflation
/// retries on failure or poor conditioning).
pub fn solve_simple_kriging_3d(
    samples: &[Coord3D],
    values: &[Real],
    mean: Real,
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

    let mut k_block = build_covariance_block(samples, anisotropy, variogram);
    let k0: DVector<f64> = DVector::from_iterator(
        n,
        samples.iter().map(|s| {
            variogram.covariance(anisotropy.anisotropic_distance(*s, target) as Real) as f64
        }),
    );
    let sill_at_zero = variogram.covariance(0.0) as f64;

    let (chol, cond_proxy, used_inflation) = factor_with_retries(&mut k_block, config)?;

    // SK has no Lagrangian: lambda = K^-1 k0 directly.
    let lambda = chol.solve(&k0);
    if lambda.iter().any(|x| !x.is_finite()) {
        return Err(SolverFailure::NonFiniteWeights);
    }

    // ẑ = m + Σ λᵢ (zᵢ − m).
    let m_f64 = mean as f64;
    let residual_value: f64 = lambda
        .iter()
        .zip(values.iter())
        .map(|(l, z)| l * ((*z as f64) - m_f64))
        .sum();
    let value = m_f64 + residual_value;

    // σ² = C(0) − λᵀk₀. No μ term in SK.
    let variance = (sill_at_zero - lambda.dot(&k0)).max(0.0);

    Ok(Prediction3D {
        value: value as Real,
        variance: variance as Real,
        condition_number: cond_proxy,
        used_nugget_inflation: used_inflation,
    })
}

/// Universal-kriging linear-trend basis: `[1, x, y, z]`. v1 supports
/// only this basis (per v3 §"UK trend functions in v1"); quadratic and
/// arbitrary callbacks are deferred to v2.
const UK_LINEAR_TREND_DIM: usize = 4;

fn linear_trend_row(p: Coord3D) -> [f64; UK_LINEAR_TREND_DIM] {
    [1.0, p.x as f64, p.y as f64, p.z as f64]
}

/// Solve a single universal-kriging system at the given target with the
/// linear trend basis `[1, x, y, z]`.
///
/// UK system:
///
/// ```text
///   [ K   F ] [λ]   [k₀]
///   [ Fᵀ  0 ] [μ] = [f₀]
/// ```
///
/// where `F` is `n × 4` with each row the trend basis evaluated at the
/// sample, and `f₀` is the trend basis at the target. Solved via Schur
/// complement on `K`:
///
/// - `U = K⁻¹ F` (4 Cholesky solves reusing the same factorization).
/// - `v = K⁻¹ k₀` (one more solve).
/// - `μ = (FᵀU)⁻¹ (Fᵀv − f₀)`. The 4×4 system is solved directly with
///   `nalgebra::Matrix4`.
/// - `λ = v − Uμ`.
///
/// Variance: `σ² = C(0) − λᵀk₀ − μᵀf₀`.
///
/// Returns [`SolverFailure::NonFiniteWeights`] if the 4×4 trend system
/// (`FᵀU`) is singular — this happens when sample geometry doesn't span
/// the trend basis (e.g. all samples in a horizontal plane → z-column
/// of F is constant → singular Fᵀ K⁻¹ F).
pub fn solve_universal_kriging_3d_linear(
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
    if n < UK_LINEAR_TREND_DIM {
        // Not enough samples to even define the trend (need at least 4
        // non-degenerate samples for the [1, x, y, z] basis).
        return Err(SolverFailure::NonFiniteWeights);
    }

    let mut k_block = build_covariance_block(samples, anisotropy, variogram);
    let k0: DVector<f64> = DVector::from_iterator(
        n,
        samples.iter().map(|s| {
            variogram.covariance(anisotropy.anisotropic_distance(*s, target) as Real) as f64
        }),
    );
    let sill_at_zero = variogram.covariance(0.0) as f64;

    // F is n x 4: each row is the trend basis at sample i.
    let mut f_mat = DMatrix::<f64>::zeros(n, UK_LINEAR_TREND_DIM);
    for (i, s) in samples.iter().enumerate() {
        let row = linear_trend_row(*s);
        for j in 0..UK_LINEAR_TREND_DIM {
            f_mat[(i, j)] = row[j];
        }
    }
    let f0_arr = linear_trend_row(target);
    let f0 = Vector4::from(f0_arr);

    let (chol, cond_proxy, used_inflation) = factor_with_retries(&mut k_block, config)?;

    // U = K^-1 F (column-by-column Cholesky solves).
    let mut u_mat = DMatrix::<f64>::zeros(n, UK_LINEAR_TREND_DIM);
    for j in 0..UK_LINEAR_TREND_DIM {
        let col = f_mat.column(j).into_owned();
        let solved = chol.solve(&col);
        u_mat.set_column(j, &solved);
    }
    // v = K^-1 k0.
    let v_vec = chol.solve(&k0);

    // Fᵀ U is the 4x4 trend system; Fᵀ v − f0 is its RHS.
    // We extract to fixed-size Matrix4/Vector4 for direct inversion.
    let ftu_dyn = f_mat.transpose() * &u_mat; // 4x4 in DMatrix
    let mut ftu = Matrix4::<f64>::zeros();
    for i in 0..UK_LINEAR_TREND_DIM {
        for j in 0..UK_LINEAR_TREND_DIM {
            ftu[(i, j)] = ftu_dyn[(i, j)];
        }
    }
    let ftv_dyn: DVector<f64> = f_mat.transpose() * &v_vec; // 4-vector
    let mut ftv = Vector4::<f64>::zeros();
    for i in 0..UK_LINEAR_TREND_DIM {
        ftv[i] = ftv_dyn[i];
    }
    let rhs_mu = ftv - f0;

    // mu = (Fᵀ U)⁻¹ (Fᵀ v − f0). 4x4 LU is exact enough.
    let mu_vec = match ftu.lu().solve(&rhs_mu) {
        Some(m) => m,
        None => return Err(SolverFailure::NonFiniteWeights),
    };

    // λ = v − U μ.
    // u_mat * mu_vec is an n-vector. Do it in DMatrix to stay shapely.
    let mu_dyn = DVector::from_iterator(UK_LINEAR_TREND_DIM, mu_vec.iter().copied());
    let lambda = &v_vec - &(u_mat * mu_dyn);
    if lambda.iter().any(|x| !x.is_finite()) || mu_vec.iter().any(|x| !x.is_finite()) {
        return Err(SolverFailure::NonFiniteWeights);
    }

    // Predicted value: λ · z.
    let value: f64 = lambda
        .iter()
        .zip(values.iter())
        .map(|(l, z)| l * (*z as f64))
        .sum();

    // Variance: σ² = C(0) − λᵀk₀ − μᵀf₀.
    let lambda_dot_k0 = lambda.dot(&k0);
    let mu_dot_f0: f64 = mu_vec.iter().zip(f0_arr.iter()).map(|(m, f)| m * f).sum();
    let variance = (sill_at_zero - lambda_dot_k0 - mu_dot_f0).max(0.0);

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
        let samples = vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(10.0, 0.0, 0.0)];
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

    // ---------- Simple-kriging solver tests ----------

    #[test]
    fn sk_predicts_global_mean_far_from_samples() {
        // Far from any sample, all kriging weights should collapse to ~0
        // and the prediction should be the global mean.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
        ];
        let values = vec![1.0 as Real, 2.0 as Real, 3.0 as Real];
        let mean: Real = 5.0;
        let far_target = Coord3D::new(1000.0, 1000.0, 1000.0);
        let pred = solve_simple_kriging_3d(
            &samples,
            &values,
            mean,
            far_target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        assert_relative_eq!(pred.value as f64, mean as f64, epsilon = 1e-3);
        // Variance approaches C(0) at infinity (no information).
        let sill = isotropic_exponential().covariance(0.0) as f64;
        assert_relative_eq!(pred.variance as f64, sill, epsilon = 1e-3);
    }

    #[test]
    fn sk_recovers_sample_value_at_sample_location() {
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
        ];
        let values = vec![1.0 as Real, 3.0 as Real, 5.0 as Real];
        let pred = solve_simple_kriging_3d(
            &samples,
            &values,
            2.0, // arbitrary mean; SK should still hit the sample
            samples[1],
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        assert_relative_eq!(pred.value as f64, 3.0, epsilon = 1e-3);
        assert!(
            (pred.variance as f64).abs() < 1e-3,
            "SK variance at sample should be ~0, got {}",
            pred.variance,
        );
    }

    #[test]
    fn sk_with_nonzero_mean_uses_residuals() {
        // Two samples at known locations with values v1, v2. With mean m,
        // SK on a target between them should be approximately
        // m + (residual average). Just check it's between v1 and v2 and
        // shifted toward the mean.
        let samples = vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(20.0, 0.0, 0.0)];
        let values = vec![1.0 as Real, 3.0 as Real];
        let mean: Real = 10.0;
        let target = Coord3D::new(10.0, 0.0, 0.0);
        let pred = solve_simple_kriging_3d(
            &samples,
            &values,
            mean,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        // Range = 10 and target is 10 units from each sample, so the
        // covariance between target and each sample is small
        // (exp(-3) ~ 0.05). SK weights are small; prediction is
        // dominated by the mean with a small residual pull toward the
        // sample values (which sit below the mean). Expect a value
        // pulled slightly below the mean.
        let v = pred.value as f64;
        let m = mean as f64;
        assert!(v < m, "SK value {v} should be below the mean {m}");
        assert!(
            v > m - 2.0,
            "SK value {v} should be only mildly pulled below mean {m} \
             (residual magnitude ~8 but tiny weights)",
        );
    }

    // ---------- Universal-kriging (linear trend) solver tests ----------

    fn three_d_dataset_for_uk() -> (Vec<Coord3D>, Vec<Real>) {
        // 8 corners of a unit cube + center: span the 3-D linear basis.
        let coords = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
            Coord3D::new(10.0, 10.0, 0.0),
            Coord3D::new(0.0, 0.0, 10.0),
            Coord3D::new(10.0, 0.0, 10.0),
            Coord3D::new(0.0, 10.0, 10.0),
            Coord3D::new(10.0, 10.0, 10.0),
            Coord3D::new(5.0, 5.0, 5.0),
        ];
        // A linear trend: v = 1 + 2x + 3y + 4z + small noise.
        let values: Vec<Real> = coords
            .iter()
            .map(|c| 1.0 + 2.0 * c.x + 3.0 * c.y + 4.0 * c.z)
            .collect();
        (coords, values)
    }

    #[test]
    fn uk_recovers_linear_trend_exactly_on_noiseless_data() {
        // Noiseless linear-trend data: UK should reproduce the trend
        // perfectly at any target inside or near the convex hull.
        let (coords, values) = three_d_dataset_for_uk();
        let target = Coord3D::new(3.0, 7.0, 2.5);
        let expected = 1.0 + 2.0 * 3.0 + 3.0 * 7.0 + 4.0 * 2.5;
        let pred = solve_universal_kriging_3d_linear(
            &coords,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        // UK should hit the linear trend within solver noise.
        assert_relative_eq!(pred.value as f64, expected, epsilon = 5e-3);
    }

    #[test]
    fn uk_recovers_sample_value_at_sample_location() {
        let (coords, values) = three_d_dataset_for_uk();
        let pred = solve_universal_kriging_3d_linear(
            &coords,
            &values,
            coords[3], // (10, 10, 0)
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        )
        .unwrap();
        assert_relative_eq!(pred.value as f64, values[3] as f64, epsilon = 1e-3);
        assert!(
            (pred.variance as f64).abs() < 1e-2,
            "UK variance at sample should be ~0, got {}",
            pred.variance,
        );
    }

    #[test]
    fn uk_errors_when_samples_dont_span_trend_basis() {
        // All samples coplanar in z=0 → F[:, 3] is constant zero → 4x4
        // trend system is singular.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
            Coord3D::new(10.0, 10.0, 0.0),
        ];
        let values = vec![1.0 as Real, 2.0 as Real, 3.0 as Real, 4.0 as Real];
        let target = Coord3D::new(5.0, 5.0, 5.0);
        let result = solve_universal_kriging_3d_linear(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        );
        assert!(
            matches!(result, Err(SolverFailure::NonFiniteWeights)),
            "expected NonFiniteWeights for trend-degenerate sample set, got {:?}",
            result,
        );
    }

    #[test]
    fn uk_errors_when_too_few_samples_for_basis() {
        // Linear basis is 4-dimensional; with fewer than 4 samples the
        // system is underdetermined.
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
        ];
        let values = vec![1.0 as Real, 2.0 as Real, 3.0 as Real];
        let target = Coord3D::new(5.0, 5.0, 5.0);
        let result = solve_universal_kriging_3d_linear(
            &samples,
            &values,
            target,
            &Anisotropy3D::identity(),
            &isotropic_exponential(),
            &SolverConfig::default(),
        );
        assert!(matches!(result, Err(SolverFailure::NonFiniteWeights)));
    }
}
