use std::num::NonZeroUsize;

use crate::Real;
use crate::distance::haversine_distance;
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;

/// A positive real number (> 0), enforced at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveReal(Real);

impl PositiveReal {
    pub fn try_new(x: Real) -> Result<Self, KrigingError> {
        if x > 0.0 && x.is_finite() {
            Ok(Self(x))
        } else {
            Err(KrigingError::FittingError(
                "value must be finite and positive".to_string(),
            ))
        }
    }

    #[inline]
    pub fn get(self) -> Real {
        self.0
    }
}

impl std::ops::Deref for PositiveReal {
    type Target = Real;
    #[inline]
    fn deref(&self) -> &Real {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct EmpiricalVariogram {
    pub distances: Vec<Real>,
    pub semivariances: Vec<Real>,
    pub n_pairs: Vec<usize>,
}

/// Which empirical variogram estimator to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmpiricalEstimator {
    /// Matheron's classical estimator: `γ̂(h) = 0.5·mean((z_i − z_j)²)`. Optimal under a
    /// Gaussian process but sensitive to outliers.
    #[default]
    Classical,
    /// Cressie–Hawkins (1980) robust estimator:
    /// `γ̂(h) = 0.5·mean(|z_i − z_j|^{1/2})⁴ / (0.457 + 0.494/N + 0.045/N²)`,
    /// where `N` is the number of pairs in the bin. Down-weights outliers.
    CressieHawkins,
}

#[derive(Debug, Clone)]
pub struct VariogramConfig {
    pub max_distance: Option<PositiveReal>,
    pub n_bins: NonZeroUsize,
    pub estimator: EmpiricalEstimator,
}

impl Default for VariogramConfig {
    fn default() -> Self {
        Self {
            max_distance: None,
            n_bins: NonZeroUsize::new(12).expect("12 != 0"),
            estimator: EmpiricalEstimator::Classical,
        }
    }
}

/// Omnidirectional empirical variogram of a geographic dataset (Haversine distance, km).
/// See [`compute_empirical_variogram_projected`](crate::compute_empirical_variogram_projected)
/// for planar `(x, y)` data.
pub fn compute_empirical_variogram(
    dataset: &GeoDataset,
    config: &VariogramConfig,
) -> Result<EmpiricalVariogram, KrigingError> {
    let coords = dataset.coords();
    empirical_variogram_with_distance(dataset.values(), config, |i, j| {
        haversine_distance(coords[i], coords[j])
    })
}

/// Shared binning + estimator core: `distance(i, j)` gives the separation of samples `i`
/// and `j`, so geographic and planar variograms differ only in the closure they pass.
pub(crate) fn empirical_variogram_with_distance(
    values: &[Real],
    config: &VariogramConfig,
    distance: impl Fn(usize, usize) -> Real,
) -> Result<EmpiricalVariogram, KrigingError> {
    let n = values.len();
    let n_bins = config.n_bins.get();
    let mut dist_sums = vec![0.0 as Real; n_bins];
    // For the Cressie–Hawkins estimator we accumulate Σ |Δz|^{1/2}; for the classical one
    // we accumulate Σ (Δz)². Both pools are averaged in the finalization step.
    let mut value_sums = vec![0.0 as Real; n_bins];
    let mut counts = vec![0usize; n_bins];
    let robust = matches!(config.estimator, EmpiricalEstimator::CressieHawkins);

    let accumulate_pair = |values_i: Real,
                           values_j: Real,
                           bin: usize,
                           d: Real,
                           dist_sums: &mut [Real],
                           value_sums: &mut [Real],
                           counts: &mut [usize]| {
        let dz = (values_i - values_j).abs();
        let g = if robust { dz.sqrt() } else { 0.5 * dz * dz };
        dist_sums[bin] += d;
        value_sums[bin] += g;
        counts[bin] += 1;
    };

    if let Some(max_dist) = config.max_distance {
        let max_dist = max_dist.get();
        let bin_width = max_dist / n_bins as Real;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = distance(i, j);
                if d > max_dist {
                    continue;
                }
                let mut bin = (d / bin_width).floor() as usize;
                if bin >= n_bins {
                    bin = n_bins - 1;
                }
                accumulate_pair(
                    values[i],
                    values[j],
                    bin,
                    d,
                    &mut dist_sums,
                    &mut value_sums,
                    &mut counts,
                );
            }
        }
    } else {
        // Two-pass streaming: first pass finds the max distance so we can size bins; second
        // pass bins pairs directly. This avoids the `O(n²)` allocation that materializing
        // every (d, z_i, z_j) tuple would otherwise require.
        let mut max_observed: Real = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = distance(i, j);
                if d > max_observed {
                    max_observed = d;
                }
            }
        }
        if max_observed <= 0.0 {
            return Err(KrigingError::FittingError(
                "max distance must be positive".to_string(),
            ));
        }
        let bin_width = max_observed / n_bins as Real;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = distance(i, j);
                let mut bin = (d / bin_width).floor() as usize;
                if bin >= n_bins {
                    bin = n_bins - 1;
                }
                accumulate_pair(
                    values[i],
                    values[j],
                    bin,
                    d,
                    &mut dist_sums,
                    &mut value_sums,
                    &mut counts,
                );
            }
        }
    }

    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for i in 0..n_bins {
        if counts[i] == 0 {
            continue;
        }
        let n_i = counts[i] as Real;
        let g = if robust {
            // Cressie–Hawkins: γ̂(h) = 0.5·(mean(|Δz|^{1/2}))⁴ / (0.457 + 0.494/N + 0.045/N²).
            let mean_sqrt = value_sums[i] / n_i;
            let numer = mean_sqrt.powi(4);
            let denom = 0.457 + 0.494 / n_i + 0.045 / (n_i * n_i);
            0.5 * numer / denom
        } else {
            value_sums[i] / n_i
        };
        distances.push(dist_sums[i] / n_i);
        semivariances.push(g);
        n_pairs.push(counts[i]);
    }

    if distances.is_empty() {
        return Err(KrigingError::FittingError(
            "no pairs in selected distance range".to_string(),
        ));
    }

    Ok(EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    })
}

/// For binomial (logit) inputs: classical empirical variogram on **adjusted** per-pair
/// contributions `max(0, ½(Δz)² − ½(σ_i²+σ_j²))` (σ² = per-site logit observation variances
/// as in the binomial model), weighted by
/// `1 / (0.5·(σ_i²+σ_j²) + ε)`.
///
/// This targets the **latent** logit field rather than inflating the empirical γ with
/// independent sampling noise on the observations. Requires
/// `per_site_logit_variance.len() == dataset.len()`. **Cressie–Hawkins** is not supported
/// (returns [`KrigingError::FittingError`]); use [`EmpiricalEstimator::Classical`].
pub fn compute_empirical_variogram_binomial_calibrated(
    dataset: &GeoDataset,
    per_site_logit_variance: &[Real],
    config: &VariogramConfig,
    rel_weight_eps: Real,
) -> Result<EmpiricalVariogram, KrigingError> {
    if !matches!(config.estimator, EmpiricalEstimator::Classical) {
        return Err(KrigingError::FittingError(
            "binomial_calibrated variogram: use Estimator::Classical only; Cressie-Hawkins is not implemented for this path".to_string(),
        ));
    }
    let n = dataset.len();
    if per_site_logit_variance.len() != n {
        return Err(KrigingError::FittingError(
            "per_site_logitVariance must have same length as dataset values".to_string(),
        ));
    }
    for &v in per_site_logit_variance {
        if !v.is_finite() || v < 0.0 {
            return Err(KrigingError::FittingError(
                "per-site logit variance must be finite and non-negative".to_string(),
            ));
        }
    }
    let coords = dataset.coords();
    let values = dataset.values();
    let n_bins = config.n_bins.get();
    let mut dist_weighted: Vec<Real> = vec![0.0; n_bins];
    let mut gamma_weighted: Vec<Real> = vec![0.0; n_bins];
    let mut weight_sums: Vec<Real> = vec![0.0; n_bins];
    let mut counts: Vec<usize> = vec![0usize; n_bins];

    let (distances, semivariances, n_pairs) = {
        if let Some(max_dist) = config.max_distance {
            let max_d = max_dist.get();
            let bin_width = max_d / n_bins as Real;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = haversine_distance(coords[i], coords[j]);
                    if d > max_d {
                        continue;
                    }
                    let mut bin = (d / bin_width).floor() as usize;
                    if bin >= n_bins {
                        bin = n_bins - 1;
                    }
                    let vi = per_site_logit_variance[i];
                    let vj = per_site_logit_variance[j];
                    let w = 1.0 / (0.5 * (vi + vj) + rel_weight_eps);
                    if !w.is_finite() {
                        continue;
                    }
                    let half_sq = 0.5 * (values[i] - values[j]) * (values[i] - values[j]);
                    let half_noise = 0.5 * (vi + vj);
                    let g = (half_sq - half_noise).max(0.0);
                    dist_weighted[bin] += w * d;
                    gamma_weighted[bin] += w * g;
                    weight_sums[bin] += w;
                    counts[bin] += 1;
                }
            }
        } else {
            let mut max_observed: Real = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = haversine_distance(coords[i], coords[j]);
                    if d > max_observed {
                        max_observed = d;
                    }
                }
            }
            if max_observed <= 0.0 {
                return Err(KrigingError::FittingError(
                    "max distance must be positive".to_string(),
                ));
            }
            let bin_width = max_observed / n_bins as Real;
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = haversine_distance(coords[i], coords[j]);
                    let mut bin = (d / bin_width).floor() as usize;
                    if bin >= n_bins {
                        bin = n_bins - 1;
                    }
                    let vi = per_site_logit_variance[i];
                    let vj = per_site_logit_variance[j];
                    let w = 1.0 / (0.5 * (vi + vj) + rel_weight_eps);
                    if !w.is_finite() {
                        continue;
                    }
                    let half_sq = 0.5 * (values[i] - values[j]) * (values[i] - values[j]);
                    let half_noise = 0.5 * (vi + vj);
                    let g = (half_sq - half_noise).max(0.0);
                    dist_weighted[bin] += w * d;
                    gamma_weighted[bin] += w * g;
                    weight_sums[bin] += w;
                    counts[bin] += 1;
                }
            }
        }

        let mut distances = Vec::new();
        let mut semivariances = Vec::new();
        let mut n_pairs = Vec::new();
        for i in 0..n_bins {
            if counts[i] == 0 || weight_sums[i] <= 0.0 {
                continue;
            }
            let wsum = weight_sums[i];
            distances.push(dist_weighted[i] / wsum);
            semivariances.push(gamma_weighted[i] / wsum);
            n_pairs.push(counts[i]);
        }
        (distances, semivariances, n_pairs)
    };

    if distances.is_empty() {
        return Err(KrigingError::FittingError(
            "no pairs in selected distance range (binomial calibrated variogram)".to_string(),
        ));
    }

    Ok(EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::GeoCoord;

    #[test]
    fn empirical_variogram_has_non_empty_bins() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 2.0, 2.0, 3.0];
        let dataset = GeoDataset::new(coords, values).unwrap();
        let out = compute_empirical_variogram(
            &dataset,
            &VariogramConfig {
                max_distance: None,
                n_bins: NonZeroUsize::new(6).unwrap(),
                estimator: EmpiricalEstimator::Classical,
            },
        )
        .expect("empirical variogram should compute");
        assert!(!out.distances.is_empty());
        assert_eq!(out.distances.len(), out.semivariances.len());
        assert_eq!(out.distances.len(), out.n_pairs.len());
    }

    #[test]
    fn cressie_hawkins_is_robust_to_a_single_outlier() {
        // A clean linear field plus one extreme outlier. The robust estimator should produce
        // smaller semivariances in each populated bin than the classical estimator because
        // the outlier's contribution is down-weighted.
        let mut coords: Vec<GeoCoord> = Vec::new();
        let mut values: Vec<Real> = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                coords.push(GeoCoord::try_new(i as Real * 0.1, j as Real * 0.1).unwrap());
                values.push((i + j) as Real); // smooth plane
            }
        }
        // Replace one value with a large outlier.
        *values.last_mut().unwrap() = 1000.0;
        let dataset = GeoDataset::new(coords, values).unwrap();

        let classical = compute_empirical_variogram(
            &dataset,
            &VariogramConfig {
                estimator: EmpiricalEstimator::Classical,
                n_bins: NonZeroUsize::new(5).unwrap(),
                ..Default::default()
            },
        )
        .expect("classical");
        let robust = compute_empirical_variogram(
            &dataset,
            &VariogramConfig {
                estimator: EmpiricalEstimator::CressieHawkins,
                n_bins: NonZeroUsize::new(5).unwrap(),
                ..Default::default()
            },
        )
        .expect("robust");
        assert_eq!(classical.semivariances.len(), robust.semivariances.len());
        // Compare the maximum bin; classical will be dominated by the outlier's squared diff.
        let max_classical = classical
            .semivariances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        let max_robust = robust
            .semivariances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        assert!(
            max_robust < max_classical * 0.5,
            "robust max={max_robust} should be much smaller than classical max={max_classical}"
        );
    }

    #[test]
    fn empirical_variogram_preserves_pair_accounting_with_fixed_max_distance() {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 0.5).unwrap(),
            GeoCoord::try_new(0.5, 0.0).unwrap(),
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let max_distance = PositiveReal::try_new(500.0).unwrap();
        let dataset = GeoDataset::new(coords, values).unwrap();
        let out = compute_empirical_variogram(
            &dataset,
            &VariogramConfig {
                max_distance: Some(max_distance),
                n_bins: NonZeroUsize::new(8).unwrap(),
                estimator: EmpiricalEstimator::Classical,
            },
        )
        .expect("empirical variogram should compute");

        let mut expected_pair_count = 0usize;
        let coords = dataset.coords();
        for i in 0..coords.len() {
            for j in (i + 1)..coords.len() {
                if haversine_distance(coords[i], coords[j]) <= max_distance.get() {
                    expected_pair_count += 1;
                }
            }
        }
        let observed_pair_count = out.n_pairs.iter().sum::<usize>();
        assert_eq!(observed_pair_count, expected_pair_count);
        assert_eq!(out.distances.len(), out.semivariances.len());
        assert_eq!(out.distances.len(), out.n_pairs.len());
        assert!(
            out.distances
                .iter()
                .all(|d| *d >= 0.0 && *d <= max_distance.get())
        );
        assert!(out.semivariances.iter().all(|g| g.is_finite() && *g >= 0.0));
    }
}
