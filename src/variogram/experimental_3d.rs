//! 3-D experimental variogram (omnidirectional).
//!
//! Pair generation iterates over all `n*(n-1)/2` unordered pairs and bins by
//! anisotropic distance. **No kd-tree filtering** — for omnidirectional
//! semivariograms over reasonably-sized point sets (n <~ 5000) the naive
//! O(n²) loop is the right call: every pair is a candidate, and the kd-tree
//! only helps when `max_distance` is much smaller than the dataset extent.
//! At M6 the directional variant adds kd-tree filtering where it earns its
//! keep.
//!
//! The output type [`EmpiricalVariogram`] is the same as upstream's 2-D
//! variant — `distances`, `semivariances`, `n_pairs` are metric-agnostic
//! and reused unchanged.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::variogram::empirical::{EmpiricalEstimator, EmpiricalVariogram, VariogramConfig};

/// Compute the 3-D omnidirectional experimental variogram of a
/// [`PlanarDataset3D`] under the given anisotropy.
///
/// Pass [`Anisotropy3D::identity`] for a purely Cartesian (isotropic)
/// variogram.
pub fn compute_empirical_variogram_3d(
    dataset: &PlanarDataset3D,
    anisotropy: &Anisotropy3D,
    config: &VariogramConfig,
) -> Result<EmpiricalVariogram, KrigingError> {
    let coords = dataset.coords();
    let values = dataset.values();
    let n = coords.len();
    let n_bins = config.n_bins.get();

    let robust = matches!(config.estimator, EmpiricalEstimator::CressieHawkins);

    let bin_width = match config.max_distance {
        Some(max_dist) => max_dist.get() / n_bins as Real,
        None => {
            // Two-pass: find max distance, then bin. Each row i's max is
            // an independent reduction so this parallelizes cleanly.
            let max_observed = row_max_distance(n, coords, anisotropy);
            if max_observed <= 0.0 {
                return Err(KrigingError::FittingError(
                    "max distance must be positive".to_string(),
                ));
            }
            // Pad outward by a tiny relative epsilon so the longest pair
            // (which equals max_observed exactly) lands inside the last bin
            // under right-exclusive binning. Without this, the two-pass
            // path silently drops the max-distance pair.
            (max_observed * (1.0 + Real::EPSILON * 4.0)) / n_bins as Real
        }
    };

    // Binning is right-exclusive: bin i covers [i*bin_width, (i+1)*bin_width).
    // Pairs at distance >= n_bins*bin_width are excluded -- this matches the
    // convention used by scikit-gstat (and most numpy-style histogram code)
    // and keeps pair-count parity with our M5 reference fixture. The trade
    // is that a pair landing exactly on the outer boundary is dropped rather
    // than clamped into the last bin; for any non-degenerate dataset this
    // affects at most a handful of pairs.
    let max_d = config
        .max_distance
        .map(|m| m.get())
        .unwrap_or(bin_width * n_bins as Real);

    // Parallelize over rows i. Each row produces its own per-bin
    // accumulator tuples, then a tree-reduction sums them. Sequential
    // builds (wasm32) use the same accumulator shape via a serial fold.
    let (dist_sums, value_sums, counts) = accumulate_pairs(
        n, coords, values, anisotropy, n_bins, bin_width, max_d, robust,
    );

    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for i in 0..n_bins {
        if counts[i] == 0 {
            continue;
        }
        let n_i = counts[i] as Real;
        let g = if robust {
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

/// Maximum anisotropic pair distance across all i<j. Native: rayon
/// reduction. WASM: serial.
#[cfg(not(target_arch = "wasm32"))]
fn row_max_distance(n: usize, coords: &[Coord3D], anisotropy: &Anisotropy3D) -> Real {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut m: Real = 0.0;
            for j in (i + 1)..n {
                let d = anisotropy.anisotropic_distance(coords[i], coords[j]) as Real;
                if d > m {
                    m = d;
                }
            }
            m
        })
        .reduce(|| 0.0, |a, b| if a > b { a } else { b })
}

#[cfg(target_arch = "wasm32")]
fn row_max_distance(n: usize, coords: &[Coord3D], anisotropy: &Anisotropy3D) -> Real {
    let mut max_observed: Real = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = anisotropy.anisotropic_distance(coords[i], coords[j]) as Real;
            if d > max_observed {
                max_observed = d;
            }
        }
    }
    max_observed
}

/// Compute per-bin (dist_sum, value_sum, count) accumulators across all
/// pairs `(i, j)` with `i < j`. Each row `i` is independent; per-row
/// partial accumulators are tree-reduced. Native: rayon. WASM: serial.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn accumulate_pairs(
    n: usize,
    coords: &[Coord3D],
    values: &[Real],
    anisotropy: &Anisotropy3D,
    n_bins: usize,
    bin_width: Real,
    max_d: Real,
    robust: bool,
) -> (Vec<Real>, Vec<Real>, Vec<usize>) {
    let identity = || {
        (
            vec![0.0 as Real; n_bins],
            vec![0.0 as Real; n_bins],
            vec![0usize; n_bins],
        )
    };
    (0..n)
        .into_par_iter()
        .fold(identity, |mut acc, i| {
            let (dist_sums, value_sums, counts) = &mut acc;
            for j in (i + 1)..n {
                let d = anisotropy.anisotropic_distance(coords[i], coords[j]) as Real;
                if d >= max_d {
                    continue;
                }
                let bin = (d / bin_width).floor() as usize;
                if bin >= n_bins {
                    continue;
                }
                let dz = (values[i] - values[j]).abs();
                let g = if robust { dz.sqrt() } else { 0.5 * dz * dz };
                dist_sums[bin] += d;
                value_sums[bin] += g;
                counts[bin] += 1;
            }
            acc
        })
        .reduce(identity, |mut a, b| {
            for k in 0..n_bins {
                a.0[k] += b.0[k];
                a.1[k] += b.1[k];
                a.2[k] += b.2[k];
            }
            a
        })
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn accumulate_pairs(
    n: usize,
    coords: &[Coord3D],
    values: &[Real],
    anisotropy: &Anisotropy3D,
    n_bins: usize,
    bin_width: Real,
    max_d: Real,
    robust: bool,
) -> (Vec<Real>, Vec<Real>, Vec<usize>) {
    let mut dist_sums = vec![0.0 as Real; n_bins];
    let mut value_sums = vec![0.0 as Real; n_bins];
    let mut counts = vec![0usize; n_bins];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = anisotropy.anisotropic_distance(coords[i], coords[j]) as Real;
            if d >= max_d {
                continue;
            }
            let bin = (d / bin_width).floor() as usize;
            if bin >= n_bins {
                continue;
            }
            let dz = (values[i] - values[j]).abs();
            let g = if robust { dz.sqrt() } else { 0.5 * dz * dz };
            dist_sums[bin] += d;
            value_sums[bin] += g;
            counts[bin] += 1;
        }
    }
    (dist_sums, value_sums, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord_3d::Coord3D;
    use crate::variogram::empirical::PositiveReal;
    use approx::assert_relative_eq;
    use std::num::NonZeroUsize;

    fn three_point_dataset() -> PlanarDataset3D {
        // Coordinates chosen so distances are exactly 1, 1, sqrt(2).
        PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(1.0, 0.0, 0.0),
                Coord3D::new(0.0, 1.0, 0.0),
            ],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap()
    }

    #[test]
    fn omnidirectional_isotropic_three_point_classical() {
        let dataset = three_point_dataset();
        let config = VariogramConfig {
            max_distance: Some(PositiveReal::try_new(2.0).unwrap()),
            n_bins: NonZeroUsize::new(2).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let ev =
            compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &config).unwrap();
        // Bin 0 covers [0, 1); bin 1 covers [1, 2]. Distances: (0,1) = 1.0
        // (lands in bin 1, since 1.0 / 1.0 = 1 = bin index), (0,2) = 1.0
        // (bin 1), (1,2) = sqrt(2) ≈ 1.414 (bin 1).
        //
        // All three pairs in bin 1. Matheron γ̂ = 0.5 * mean((z_i - z_j)²) =
        // 0.5 * ((1)² + (2)² + (1)²) / 3 = 0.5 * 6 / 3 = 1.0.
        assert_eq!(ev.n_pairs.len(), 1);
        assert_eq!(ev.n_pairs[0], 3);
        assert_relative_eq!(ev.semivariances[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn omnidirectional_no_pairs_in_range_errors() {
        let dataset = three_point_dataset();
        let config = VariogramConfig {
            max_distance: Some(PositiveReal::try_new(0.1).unwrap()),
            n_bins: NonZeroUsize::new(2).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let result = compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &config);
        assert!(matches!(result, Err(KrigingError::FittingError(_))));
    }

    #[test]
    fn anisotropy_changes_binning() {
        // With z-stretch 10x, two points separated along z by 0.1 become
        // anisotropically separated by 1.0. Easiest way to test: build a
        // dataset where the isotropic and anisotropic distances put pairs
        // into different bins.
        let dataset = PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(0.0, 0.0, 0.1), // close in z (isotropic d=0.1)
                Coord3D::new(1.0, 0.0, 0.0), // far in x (isotropic d=1.0)
            ],
            vec![10.0, 11.0, 20.0],
        )
        .unwrap();
        let config = VariogramConfig {
            max_distance: Some(PositiveReal::try_new(2.0).unwrap()),
            n_bins: NonZeroUsize::new(2).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let iso = Anisotropy3D::identity();
        let ev_iso = compute_empirical_variogram_3d(&dataset, &iso, &config).unwrap();

        let stretched = Anisotropy3D::from_rotation_matrix(
            nalgebra::Matrix3::identity(),
            nalgebra::Vector3::new(1.0, 1.0, 10.0),
        )
        .unwrap();
        let ev_aniso = compute_empirical_variogram_3d(&dataset, &stretched, &config).unwrap();

        // Mean distances must differ between iso and anisotropic.
        // (Detailed bin contents are sensitive to the bin-width arithmetic;
        // proving they're not equal is enough to show anisotropy is wired.)
        let total_iso_distance: Real = ev_iso
            .distances
            .iter()
            .zip(ev_iso.n_pairs.iter())
            .map(|(d, n)| d * (*n as Real))
            .sum();
        let total_aniso_distance: Real = ev_aniso
            .distances
            .iter()
            .zip(ev_aniso.n_pairs.iter())
            .map(|(d, n)| d * (*n as Real))
            .sum();
        assert!(
            (total_iso_distance - total_aniso_distance).abs() > 0.1,
            "expected anisotropy to change total pair distance: iso={total_iso_distance}, aniso={total_aniso_distance}",
        );
    }

    #[test]
    fn cressie_hawkins_estimator_runs_and_differs_from_classical() {
        let dataset = three_point_dataset();
        let base_config = VariogramConfig {
            max_distance: Some(PositiveReal::try_new(2.0).unwrap()),
            n_bins: NonZeroUsize::new(2).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let ev_classical =
            compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &base_config)
                .unwrap();

        let ch_config = VariogramConfig {
            estimator: EmpiricalEstimator::CressieHawkins,
            ..base_config
        };
        let ev_ch = compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &ch_config)
            .unwrap();

        // Same pair counts and bin distances, but semivariances differ.
        assert_eq!(ev_classical.n_pairs, ev_ch.n_pairs);
        for i in 0..ev_classical.semivariances.len() {
            assert!(
                (ev_classical.semivariances[i] - ev_ch.semivariances[i]).abs() > 1e-6,
                "classical and CH semivariances should differ in bin {i}"
            );
        }
    }

    #[test]
    fn two_pass_max_distance_inference_works() {
        // Without max_distance, two-pass logic should compute correct bins.
        let dataset = three_point_dataset();
        let config = VariogramConfig {
            max_distance: None,
            n_bins: NonZeroUsize::new(2).unwrap(),
            estimator: EmpiricalEstimator::Classical,
        };
        let ev =
            compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &config).unwrap();
        // All pairs should land in some bin.
        let total: usize = ev.n_pairs.iter().sum();
        assert_eq!(total, 3);
    }
}
