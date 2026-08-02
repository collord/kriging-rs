//! Empirical cross-variograms and Linear Model of Coregionalization (LMC) fitting.
//!
//! Fitting an LMC has two steps:
//!
//! 1. [`compute_empirical_cross_variogram`] estimates the auto- and cross-variograms
//!    `γ_ij(h)` from an (isotopic) [`MultiVariableDataset`], binned by lag exactly like the
//!    univariate [`compute_empirical_variogram`](crate::compute_empirical_variogram)
//!    (classical estimator).
//!
//! 2. [`fit_lmc`] chooses the **sill matrices** of a set of *fixed* basic structures (their
//!    ranges/shapes are picked in advance) so the model `γ_ij(h) = Σ_k B_k[i,j]·g_k(h)` matches
//!    the empirical curves, using the **Goulard–Voltz** algorithm: a cyclic weighted
//!    least-squares update of each `B_k` followed by a projection onto the positive-semidefinite
//!    cone. The PSD projection guarantees the fitted [`Coregionalization`] is admissible.
//!
//! Only the sills are fitted; to search over ranges, fit for several candidate range sets and
//! keep the lowest-residual result.

use nalgebra::DMatrix;

use crate::Real;
use crate::distance::haversine_distance;
use crate::error::KrigingError;
use crate::variogram::empirical::VariogramConfig;

use super::coregionalization::{
    Coregionalization, CoregionalizationStructure, CorrelationBasis, SillMatrix,
};
use super::dataset::MultiVariableDataset;

/// Empirical auto- and cross-variograms of `p` variables, binned by lag.
///
/// `gamma(bin, i, j)` is `γ_ij(h)` at the bin's mean distance; it is symmetric in `i, j`.
#[derive(Debug, Clone)]
pub struct EmpiricalCrossVariogram {
    n_variables: usize,
    distances: Vec<Real>,
    n_pairs: Vec<usize>,
    /// Per bin: a row-major `p × p` symmetric matrix of cross-semivariances.
    gamma: Vec<Vec<Real>>,
}

impl EmpiricalCrossVariogram {
    /// Construct directly from per-bin cross-variogram matrices (e.g. to supply externally
    /// computed values). `gamma[b]` must be a row-major `p × p` buffer.
    pub fn new(
        n_variables: usize,
        distances: Vec<Real>,
        n_pairs: Vec<usize>,
        gamma: Vec<Vec<Real>>,
    ) -> Result<Self, KrigingError> {
        if n_variables == 0 {
            return Err(KrigingError::InvalidInput(
                "cross-variogram needs at least one variable".to_string(),
            ));
        }
        if distances.is_empty() {
            return Err(KrigingError::FittingError(
                "cross-variogram has no lag bins".to_string(),
            ));
        }
        if distances.len() != n_pairs.len() || distances.len() != gamma.len() {
            return Err(KrigingError::DimensionMismatch(
                "distances, n_pairs and gamma must have equal length".to_string(),
            ));
        }
        for g in &gamma {
            if g.len() != n_variables * n_variables {
                return Err(KrigingError::DimensionMismatch(format!(
                    "each gamma matrix must have {}² entries",
                    n_variables
                )));
            }
        }
        Ok(Self {
            n_variables,
            distances,
            n_pairs,
            gamma,
        })
    }

    /// Number of variables `p`.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    /// Number of populated lag bins.
    #[inline]
    pub fn n_bins(&self) -> usize {
        self.distances.len()
    }

    /// Mean lag distances (one per bin).
    #[inline]
    pub fn distances(&self) -> &[Real] {
        &self.distances
    }

    /// Pair counts per bin.
    #[inline]
    pub fn n_pairs(&self) -> &[usize] {
        &self.n_pairs
    }

    /// Cross-semivariance `γ_ij` at `bin`.
    #[inline]
    pub fn gamma(&self, bin: usize, i: usize, j: usize) -> Real {
        self.gamma[bin][i * self.n_variables + j]
    }
}

/// Compute the empirical auto- and cross-variograms from an isotopic multivariate dataset,
/// binned by lag (classical estimator: `γ_ij(h) = 1/(2N) Σ ΔZ_i·ΔZ_j`).
pub fn compute_empirical_cross_variogram(
    dataset: &MultiVariableDataset,
    config: &VariogramConfig,
) -> Result<EmpiricalCrossVariogram, KrigingError> {
    let coords = dataset.coords();
    let n = coords.len();
    let p = dataset.n_variables();
    let cols: Vec<&[Real]> = (0..p).map(|v| dataset.variable(v)).collect();
    let n_bins = config.n_bins.get();

    // Bin width from the configured or observed maximum distance (mirrors the univariate path).
    let max_dist = match config.max_distance {
        Some(m) => m.get(),
        None => {
            let mut observed: Real = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    observed = observed.max(haversine_distance(coords[i], coords[j]));
                }
            }
            if observed <= 0.0 {
                return Err(KrigingError::FittingError(
                    "max distance must be positive".to_string(),
                ));
            }
            observed
        }
    };
    let bin_width = max_dist / n_bins as Real;

    let mut dist_sums = vec![0.0 as Real; n_bins];
    let mut counts = vec![0usize; n_bins];
    let mut cross = vec![vec![0.0 as Real; p * p]; n_bins];

    for a in 0..n {
        for b in (a + 1)..n {
            let d = haversine_distance(coords[a], coords[b]);
            if d > max_dist {
                continue;
            }
            let mut bin = (d / bin_width).floor() as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            counts[bin] += 1;
            dist_sums[bin] += d;
            let bucket = &mut cross[bin];
            for i in 0..p {
                let dzi = cols[i][a] - cols[i][b];
                for j in 0..p {
                    let dzj = cols[j][a] - cols[j][b];
                    bucket[i * p + j] += dzi * dzj;
                }
            }
        }
    }

    let mut distances = Vec::new();
    let mut n_pairs = Vec::new();
    let mut gamma = Vec::new();
    for bin in 0..n_bins {
        if counts[bin] == 0 {
            continue;
        }
        let nf = counts[bin] as Real;
        distances.push(dist_sums[bin] / nf);
        n_pairs.push(counts[bin]);
        gamma.push(cross[bin].iter().map(|s| s / (2.0 * nf)).collect());
    }
    if distances.is_empty() {
        return Err(KrigingError::FittingError(
            "no pairs in selected distance range".to_string(),
        ));
    }
    EmpiricalCrossVariogram::new(p, distances, n_pairs, gamma)
}

/// Options controlling [`fit_lmc`].
#[derive(Debug, Clone, Copy)]
pub struct LmcFitOptions {
    /// Maximum Goulard–Voltz sweeps over the structures.
    pub max_iterations: usize,
    /// Relative change in the weighted residual below which iteration stops.
    pub tolerance: Real,
}

impl Default for LmcFitOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-7,
        }
    }
}

/// Result of an LMC fit.
#[derive(Debug, Clone)]
pub struct LmcFit {
    /// The fitted (admissible) coregionalization.
    pub coregionalization: Coregionalization,
    /// Final weighted sum of squared residuals `Σ_l w_l Σ_ij (γ_ij − model)²`.
    pub residual: Real,
    /// Number of Goulard–Voltz sweeps performed.
    pub iterations: usize,
}

/// Fit an LMC's sill matrices to empirical cross-variograms by Goulard–Voltz.
///
/// `bases` are the fixed basic structures (their ranges/shapes chosen in advance); the algorithm
/// finds each structure's positive-semidefinite sill matrix. Lags are weighted by their pair
/// count. The returned [`Coregionalization`] is admissible by construction (every sill matrix is
/// PSD-projected each sweep).
pub fn fit_lmc(
    empirical: &EmpiricalCrossVariogram,
    bases: Vec<CorrelationBasis>,
    options: LmcFitOptions,
) -> Result<LmcFit, KrigingError> {
    let p = empirical.n_variables();
    let k = bases.len();
    let l = empirical.n_bins();
    if k == 0 {
        return Err(KrigingError::InvalidInput(
            "LMC fitting needs at least one basic structure".to_string(),
        ));
    }

    // g[k][l] = 1 − ρ_k(h_l): the k-th basic structure's (unit-sill) variogram at each lag.
    let g: Vec<Vec<Real>> = bases
        .iter()
        .map(|basis| {
            empirical
                .distances()
                .iter()
                .map(|&h| 1.0 - basis.correlation(h))
                .collect()
        })
        .collect();
    // Per-lag weight = pair count (≥ 1).
    let w: Vec<Real> = empirical
        .n_pairs()
        .iter()
        .map(|&c| c.max(1) as Real)
        .collect();
    // Empirical Γ(h_l) as p×p matrices.
    let gamma: Vec<DMatrix<Real>> = (0..l)
        .map(|li| DMatrix::<Real>::from_fn(p, p, |i, j| empirical.gamma(li, i, j)))
        .collect();

    // Weighted residual of the current model.
    let wss = |b: &[DMatrix<Real>]| -> Real {
        let mut s = 0.0 as Real;
        for li in 0..l {
            let mut model = DMatrix::<Real>::zeros(p, p);
            for kk in 0..k {
                model += &b[kk] * g[kk][li];
            }
            let diff = &gamma[li] - &model;
            s += w[li] * diff.iter().map(|x| x * x).sum::<Real>();
        }
        s
    };

    let mut bmat: Vec<DMatrix<Real>> = (0..k).map(|_| DMatrix::<Real>::zeros(p, p)).collect();
    let mut prev = wss(&bmat);
    let mut iterations = 0;
    for _ in 0..options.max_iterations {
        iterations += 1;
        for kk in 0..k {
            let denom: Real = (0..l).map(|li| w[li] * g[kk][li] * g[kk][li]).sum();
            if denom <= 1e-12 {
                // Structure is flat over the sampled lags — leave it unchanged.
                continue;
            }
            // Weighted LS target for B_k from the residual with the other structures removed.
            let mut m = DMatrix::<Real>::zeros(p, p);
            for li in 0..l {
                let mut other = DMatrix::<Real>::zeros(p, p);
                for (k2, b2) in bmat.iter().enumerate() {
                    if k2 != kk {
                        other += b2 * g[k2][li];
                    }
                }
                let residual = &gamma[li] - &other;
                m += residual * (w[li] * g[kk][li]);
            }
            bmat[kk] = project_psd(&(m / denom));
        }
        let cur = wss(&bmat);
        let converged = (prev - cur).abs() <= options.tolerance * (prev.abs() + 1e-12);
        prev = cur;
        if converged {
            break;
        }
    }

    // Assemble the coregionalization from the fitted (PSD) sill matrices.
    let mut structures = Vec::with_capacity(k);
    for (basis, b) in bases.iter().zip(bmat.iter()) {
        let mut flat = Vec::with_capacity(p * p);
        for i in 0..p {
            for j in 0..p {
                flat.push(b[(i, j)]);
            }
        }
        let sill = SillMatrix::new(p, flat)?;
        structures.push(CoregionalizationStructure::new(*basis, sill));
    }
    let coregionalization = Coregionalization::new(structures)?;

    Ok(LmcFit {
        coregionalization,
        residual: prev,
        iterations,
    })
}

/// Project a symmetric matrix onto the positive-semidefinite cone (clamp eigenvalues at 0).
fn project_psd(m: &DMatrix<Real>) -> DMatrix<Real> {
    // Symmetrize to kill round-off asymmetry, then clamp negative eigenvalues.
    let sym = (m + m.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    let clamped = eig.eigenvalues.map(|v| v.max(0.0));
    &eig.eigenvectors * DMatrix::from_diagonal(&clamped) * eig.eigenvectors.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;
    use crate::variogram::empirical::EmpiricalEstimator;
    use crate::variogram::models::{VariogramModel, VariogramType};
    use approx::assert_relative_eq;
    use std::num::NonZeroUsize;

    fn exp_basis(range: Real) -> CorrelationBasis {
        CorrelationBasis::Model(
            VariogramModel::new(0.0, 1.0, range, VariogramType::Exponential).unwrap(),
        )
    }

    #[test]
    fn goulard_voltz_recovers_a_known_lmc() {
        // Build a true 2-variable LMC (nugget + exponential) and synthesize exact empirical
        // cross-variograms from it. Goulard–Voltz should recover the sill matrices.
        let range = 100.0;
        let bases = vec![CorrelationBasis::Nugget, exp_basis(range)];
        let b_nugget = [[0.5 as Real, 0.1], [0.1, 0.4]];
        let b_exp = [[3.0 as Real, 1.2], [1.2, 2.0]];
        let lags = [10.0 as Real, 25.0, 50.0, 90.0, 150.0];

        let gnug = |h: Real| 1.0 - CorrelationBasis::Nugget.correlation(h);
        let gexp = |h: Real| 1.0 - exp_basis(range).correlation(h);
        let mut gamma = Vec::new();
        for &h in &lags {
            let mut mat = vec![0.0 as Real; 4];
            for i in 0..2 {
                for j in 0..2 {
                    mat[i * 2 + j] = b_nugget[i][j] * gnug(h) + b_exp[i][j] * gexp(h);
                }
            }
            gamma.push(mat);
        }
        let empirical =
            EmpiricalCrossVariogram::new(2, lags.to_vec(), vec![100; lags.len()], gamma).unwrap();

        let fit = fit_lmc(&empirical, bases, LmcFitOptions::default()).unwrap();
        assert!(
            fit.residual < 1e-4,
            "residual should be ~0, got {}",
            fit.residual
        );
        let coreg = &fit.coregionalization;
        // The fitted model must reproduce the true cross-covariances.
        for &h in &lags {
            for i in 0..2 {
                for j in 0..2 {
                    let expected = b_nugget[i][j] * CorrelationBasis::Nugget.correlation(h)
                        + b_exp[i][j] * exp_basis(range).correlation(h);
                    assert_relative_eq!(coreg.cross_covariance(i, j, h), expected, epsilon = 1e-2);
                }
            }
        }
    }

    #[test]
    fn cross_variogram_auto_matches_univariate() {
        use crate::compute_empirical_variogram;
        use crate::geo_dataset::GeoDataset;
        let coords: Vec<GeoCoord> = (0..12)
            .map(|i| GeoCoord::try_new((i as Real) * 0.15, ((i * 7) % 11) as Real * 0.1).unwrap())
            .collect();
        let a: Vec<Real> = (0..12).map(|i| (i as Real).sin() * 3.0 + 10.0).collect();
        let b: Vec<Real> = (0..12).map(|i| (i as Real).cos() * 2.0 + 5.0).collect();
        let config = VariogramConfig {
            // Auto max distance so both paths bin identically over the full lag range.
            max_distance: None,
            n_bins: NonZeroUsize::new(6).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let multi = MultiVariableDataset::new(coords.clone(), vec![a.clone(), b]).unwrap();
        let cross = compute_empirical_cross_variogram(&multi, &config).unwrap();
        let uni =
            compute_empirical_variogram(&GeoDataset::new(coords, a).unwrap(), &config).unwrap();
        // The diagonal (auto) cross-variogram of variable 0 equals the univariate variogram.
        assert_eq!(cross.n_bins(), uni.distances.len());
        for bin in 0..cross.n_bins() {
            assert_relative_eq!(
                cross.gamma(bin, 0, 0),
                uni.semivariances[bin],
                epsilon = 1e-4
            );
        }
    }

    #[test]
    fn fit_lmc_rejects_no_structures() {
        let empirical =
            EmpiricalCrossVariogram::new(1, vec![1.0], vec![10], vec![vec![0.5]]).unwrap();
        assert!(fit_lmc(&empirical, vec![], LmcFitOptions::default()).is_err());
    }
}
