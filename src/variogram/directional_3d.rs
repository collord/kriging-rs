//! 3-D directional experimental variogram (GSLib `gamv` semantics).
//!
//! Differs from [`crate::variogram::experimental_3d`] in two ways that
//! together give bitwise pair-set parity with GSLib's `gamv`:
//!
//! 1. **Lag-centred binning, not equal-width.** Lag `k` (1-indexed) accepts
//!    pairs with `|d - k·xlag| <= xltol`. With `xltol < xlag/2` there are
//!    gaps between bins; with `xltol > xlag/2` bins overlap and a pair can
//!    be in multiple bins. This is the `gamv.f90` convention.
//! 2. **Direction filter applied as a 4-step cone test.** The azimuth and
//!    dip cones plus horizontal and vertical bandwidth filters; see
//!    [`DirectionFilter3D::accepts`].
//!
//! The output is the same [`EmpiricalVariogram`] type as the omnidirectional
//! variant. The Matheron estimator is implemented; Cressie-Hawkins follows
//! upstream's formula on the same per-bin accumulators.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::variogram::empirical::{EmpiricalEstimator, EmpiricalVariogram, PositiveReal};

/// Internal direction filter — built from
/// [`crate::interop::gslib_directional::GslibDirection`] at the API boundary,
/// then used inside the pair loop. All fields are precomputed so the inner
/// loop reduces to a few multiplies and compares.
#[derive(Debug, Clone, Copy)]
pub struct DirectionFilter3D {
    /// Horizontal direction unit vector, x component.
    pub uv_x_azm: f64,
    /// Horizontal direction unit vector, y component.
    pub uv_y_azm: f64,
    /// Dip direction unit vector, z component.
    pub uv_z_dec: f64,
    /// Dip direction unit vector, horizontal component.
    pub uv_h_dec: f64,
    /// Cosine of the azimuth tolerance (cone half-angle).
    pub cos_atol: f64,
    /// Cosine of the dip tolerance (cone half-angle).
    pub cos_dtol: f64,
    /// Horizontal bandwidth.
    pub bandh: f64,
    /// Vertical bandwidth.
    pub bandv: f64,
    /// `true` if `atol >= 90 deg`. Triggers gamv's omni double-count
    /// (each pair contributes twice with reversed (tail, head) ordering).
    pub omni: bool,
}

impl DirectionFilter3D {
    /// Test whether a lag vector `(dx, dy, dz)` passes the four direction
    /// checks. Returns `(accepted, dcazm, dcdec)` so the caller can branch
    /// on whether to swap (tail, head) and whether to apply the omni
    /// double-count.
    #[inline]
    pub fn accepts(&self, dx: f64, dy: f64, dz: f64, h: f64) -> Option<(f64, f64)> {
        // Horizontal projection length.
        let dxs = dx * dx;
        let dys = dy * dy;
        let dxy = (dxs + dys).max(0.0).sqrt();

        // 1) Azimuth cone.
        let dcazm = if dxy < f64::EPSILON {
            1.0
        } else {
            (dx * self.uv_x_azm + dy * self.uv_y_azm) / dxy
        };
        if dcazm.abs() < self.cos_atol {
            return None;
        }

        // 2) Horizontal bandwidth: perpendicular distance from the lag's
        //    horizontal projection to the azimuth direction.
        let band_h = self.uv_x_azm * dy - self.uv_y_azm * dx;
        if band_h.abs() > self.bandh {
            return None;
        }

        // 3) Dip cone. gamv flips the sign of dxy when dcazm is negative
        //    (so the dip is measured along the same forward direction).
        let dxy_signed = if dcazm < 0.0 { -dxy } else { dxy };
        let dcdec = if h <= f64::EPSILON {
            0.0
        } else {
            let v = (dxy_signed * self.uv_h_dec + dz * self.uv_z_dec) / h;
            if v.abs() < self.cos_dtol {
                return None;
            }
            v
        };

        // 4) Vertical bandwidth.
        let band_v = self.uv_h_dec * dz - self.uv_z_dec * dxy_signed;
        if band_v.abs() > self.bandv {
            return None;
        }

        Some((dcazm, dcdec))
    }
}

/// Configuration for the directional variogram. Mirrors gamv's parameters
/// at the binning level — `xlag` is the bin spacing, `xltol` is the half-
/// width of each lag's tolerance band.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalConfig3D {
    pub xlag: PositiveReal,
    pub xltol: PositiveReal,
    pub n_lags: usize,
    pub estimator: EmpiricalEstimator,
}

/// Compute the directional experimental variogram. Pairs are iterated over
/// all `n*(n-1)/2` unordered combinations and accepted/rejected by the
/// `filter`; accepted pairs are binned by gamv's lag-centred rule.
///
/// If `filter.omni == true`, each accepted pair contributes twice (once
/// with `(vrt, vrh) = (vi, vj)`, once swapped) — matching gamv's behavior
/// when atol >= 90. For directional cones this flag is false and each pair
/// contributes once.
pub fn compute_directional_variogram_3d(
    dataset: &PlanarDataset3D,
    filter: &DirectionFilter3D,
    config: &DirectionalConfig3D,
) -> Result<EmpiricalVariogram, KrigingError> {
    if config.n_lags == 0 {
        return Err(KrigingError::InvalidInput(
            "n_lags must be at least 1".to_string(),
        ));
    }

    let coords = dataset.coords();
    let values = dataset.values();
    let n = coords.len();

    let xlag = config.xlag.get() as f64;
    let xltol = config.xltol.get() as f64;
    let robust = matches!(config.estimator, EmpiricalEstimator::CressieHawkins);

    // gamv's pre-distance squared cap. With n_lags lag bins indexed 1..=n_lags
    // and centres at (k-1)*xlag, the last bin accepts up to
    // (n_lags-1)*xlag + xltol. Add a tiny margin.
    let max_centre = (config.n_lags.saturating_sub(1)) as f64 * xlag;
    let dismxs = ((max_centre + xltol + f64::EPSILON) * 1.0_f64).powi(2);

    let n_lags = config.n_lags;

    // Native: parallelize the i loop with rayon, per-row accumulators
    // reduced into a single result. WASM: serial.
    //
    // gamv loops j from i (inclusive of self-pairs); we loop from i+1
    // because self-pairs always have h=0 and only land in the lag-1
    // bin if xltol >= xlag/2. Excluding them keeps the count comparable
    // to mainstream geostatistics practice. The parity test strips
    // gamv's lag-1 self bin before comparing.
    let (dist_sums, value_sums, counts) = accumulate_directional_pairs(
        n, coords, values, filter, n_lags, xlag, xltol, dismxs, robust,
    );

    let mut distances = Vec::new();
    let mut semivariances = Vec::new();
    let mut n_pairs = Vec::new();
    for k in 0..config.n_lags {
        if counts[k] == 0 {
            continue;
        }
        let n_k = counts[k] as f64;
        let g = if robust {
            let mean_sqrt = value_sums[k] / n_k;
            let numer = mean_sqrt.powi(4);
            let denom = 0.457 + 0.494 / n_k + 0.045 / (n_k * n_k);
            0.5 * numer / denom
        } else {
            value_sums[k] / n_k
        };
        distances.push((dist_sums[k] / n_k) as Real);
        semivariances.push(g as Real);
        n_pairs.push(counts[k]);
    }

    if distances.is_empty() {
        return Err(KrigingError::FittingError(
            "no pairs passed the direction filter in any lag bin".to_string(),
        ));
    }

    Ok(EmpiricalVariogram {
        distances,
        semivariances,
        n_pairs,
    })
}

/// Per-row body of the directional accumulation loop. Encapsulated so
/// the rayon and serial drivers share the inner logic verbatim.
#[inline]
fn directional_row_into(
    i: usize,
    n: usize,
    coords: &[Coord3D],
    values: &[Real],
    filter: &DirectionFilter3D,
    n_lags: usize,
    xlag: f64,
    xltol: f64,
    dismxs: f64,
    robust: bool,
    dist_sums: &mut [f64],
    value_sums: &mut [f64],
    counts: &mut [usize],
) {
    let pi = coords[i];
    for j in (i + 1)..n {
        let pj = coords[j];
        let dx = (pj.x - pi.x) as f64;
        let dy = (pj.y - pi.y) as f64;
        let dz = (pj.z - pi.z) as f64;
        let hs = dx * dx + dy * dy + dz * dz;
        if hs > dismxs {
            continue;
        }
        let h = hs.max(0.0).sqrt();

        for k in 1..=n_lags {
            let centre = (k as f64 - 1.0) * xlag;
            if h >= centre - xltol && h <= centre + xltol && filter.accepts(dx, dy, dz, h).is_some()
            {
                let dz_val = (values[i] - values[j]).abs() as f64;
                let g = if robust {
                    dz_val.sqrt()
                } else {
                    0.5 * dz_val * dz_val
                };
                dist_sums[k - 1] += h;
                value_sums[k - 1] += g;
                counts[k - 1] += 1;
                if filter.omni {
                    // gamv.f90:473-484 double-count for omni mode.
                    dist_sums[k - 1] += h;
                    value_sums[k - 1] += g; // symmetric in (vi, vj) -> same g
                    counts[k - 1] += 1;
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn accumulate_directional_pairs(
    n: usize,
    coords: &[Coord3D],
    values: &[Real],
    filter: &DirectionFilter3D,
    n_lags: usize,
    xlag: f64,
    xltol: f64,
    dismxs: f64,
    robust: bool,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let identity = || {
        (
            vec![0.0_f64; n_lags],
            vec![0.0_f64; n_lags],
            vec![0usize; n_lags],
        )
    };
    (0..n)
        .into_par_iter()
        .fold(identity, |mut acc, i| {
            directional_row_into(
                i, n, coords, values, filter, n_lags, xlag, xltol, dismxs, robust, &mut acc.0,
                &mut acc.1, &mut acc.2,
            );
            acc
        })
        .reduce(identity, |mut a, b| {
            for k in 0..n_lags {
                a.0[k] += b.0[k];
                a.1[k] += b.1[k];
                a.2[k] += b.2[k];
            }
            a
        })
}

#[cfg(target_arch = "wasm32")]
fn accumulate_directional_pairs(
    n: usize,
    coords: &[Coord3D],
    values: &[Real],
    filter: &DirectionFilter3D,
    n_lags: usize,
    xlag: f64,
    xltol: f64,
    dismxs: f64,
    robust: bool,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut dist_sums = vec![0.0_f64; n_lags];
    let mut value_sums = vec![0.0_f64; n_lags];
    let mut counts = vec![0usize; n_lags];
    for i in 0..n {
        directional_row_into(
            i,
            n,
            coords,
            values,
            filter,
            n_lags,
            xlag,
            xltol,
            dismxs,
            robust,
            &mut dist_sums,
            &mut value_sums,
            &mut counts,
        );
    }
    (dist_sums, value_sums, counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord_3d::Coord3D;
    use crate::interop::gslib_directional::GslibDirection;
    use approx::assert_relative_eq;

    fn simple_dataset() -> PlanarDataset3D {
        PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(10.0, 0.0, 0.0),
                Coord3D::new(0.0, 10.0, 0.0),
                Coord3D::new(0.0, 0.0, 10.0),
            ],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn azim_east_picks_up_only_x_aligned_pair() {
        // Direction along east (azm=90) with narrow tolerance and small
        // bandwidth. Only the (0,0,0)-(10,0,0) pair should pass.
        let f = GslibDirection {
            azm_deg: 90.0,
            atol_deg: 5.0,
            bandh: 1.0,
            dip_deg: 0.0,
            dtol_deg: 5.0,
            bandv: 1.0,
        }
        .to_filter()
        .unwrap();
        let cfg = DirectionalConfig3D {
            xlag: PositiveReal::try_new(10.0).unwrap(),
            xltol: PositiveReal::try_new(5.0).unwrap(),
            n_lags: 2,
            estimator: EmpiricalEstimator::Classical,
        };
        let ev = compute_directional_variogram_3d(&simple_dataset(), &f, &cfg).unwrap();
        // Exactly one pair (the x-aligned one) at distance 10.
        assert_eq!(ev.n_pairs.len(), 1);
        assert_eq!(ev.n_pairs[0], 1);
        assert_relative_eq!(ev.distances[0] as f64, 10.0, epsilon = 1e-6);
        // gamma = 0.5 * (2-1)^2 = 0.5
        assert_relative_eq!(ev.semivariances[0] as f64, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn vertical_dip_picks_up_only_z_aligned_pair() {
        let f = GslibDirection {
            azm_deg: 0.0,
            atol_deg: 5.0,
            bandh: 1.0,
            dip_deg: 90.0,
            dtol_deg: 5.0,
            bandv: 1.0,
        }
        .to_filter()
        .unwrap();
        let cfg = DirectionalConfig3D {
            xlag: PositiveReal::try_new(10.0).unwrap(),
            xltol: PositiveReal::try_new(5.0).unwrap(),
            n_lags: 2,
            estimator: EmpiricalEstimator::Classical,
        };
        let ev = compute_directional_variogram_3d(&simple_dataset(), &f, &cfg).unwrap();
        assert_eq!(ev.n_pairs.len(), 1);
        assert_eq!(ev.n_pairs[0], 1);
        // (1,4) along z: gamma = 0.5 * (4-1)^2 = 4.5
        assert_relative_eq!(ev.semivariances[0] as f64, 4.5, epsilon = 1e-6);
    }

    #[test]
    fn omni_doubles_pair_counts_relative_to_directional() {
        // With atol=90 (omni flag set), all pairs in each lag are counted
        // twice. Build a 2-point dataset; one pair, gamma should be the
        // same whether omni doubles or not but n_pairs should differ.
        let dataset = PlanarDataset3D::new(
            vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(10.0, 0.0, 0.0)],
            vec![1.0, 2.0],
        )
        .unwrap();
        // n_lags=2 because gamv-style lag 1 is centred at 0 and our pair
        // at distance 10 lands in lag 2 (centred at xlag=10).
        let cfg = DirectionalConfig3D {
            xlag: PositiveReal::try_new(10.0).unwrap(),
            xltol: PositiveReal::try_new(5.0).unwrap(),
            n_lags: 2,
            estimator: EmpiricalEstimator::Classical,
        };
        let dir_filter = GslibDirection {
            azm_deg: 90.0,
            atol_deg: 5.0,
            bandh: 1.0,
            dip_deg: 0.0,
            dtol_deg: 90.0,
            bandv: 1e10,
        }
        .to_filter()
        .unwrap();
        let omni_filter = GslibDirection {
            azm_deg: 0.0,
            atol_deg: 90.0,
            bandh: 1e10,
            dip_deg: 0.0,
            dtol_deg: 90.0,
            bandv: 1e10,
        }
        .to_filter()
        .unwrap();
        let ev_dir = compute_directional_variogram_3d(&dataset, &dir_filter, &cfg).unwrap();
        let ev_omni = compute_directional_variogram_3d(&dataset, &omni_filter, &cfg).unwrap();
        assert_eq!(ev_dir.n_pairs[0], 1);
        assert_eq!(ev_omni.n_pairs[0], 2);
        // Semivariance unchanged: 0.5 * (1-2)^2 / N, with the doubled
        // accumulator and doubled N giving the same ratio.
        assert_relative_eq!(
            ev_dir.semivariances[0] as f64,
            ev_omni.semivariances[0] as f64,
            epsilon = 1e-6,
        );
    }

    #[test]
    fn bandwidth_filter_rejects_off_axis_pair() {
        // A pair offset slightly off the azimuth axis but within the cone
        // can be rejected by a tight bandwidth.
        let dataset = PlanarDataset3D::new(
            vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(10.0, 2.0, 0.0)],
            vec![1.0, 2.0],
        )
        .unwrap();
        let cfg = DirectionalConfig3D {
            xlag: PositiveReal::try_new(10.0).unwrap(),
            xltol: PositiveReal::try_new(5.0).unwrap(),
            n_lags: 1,
            estimator: EmpiricalEstimator::Classical,
        };
        // 22.5 deg cone half-angle accepts the pair (since arctan(2/10) ~ 11.3 deg);
        // bandh=1 rejects it (perpendicular distance ~ 2 > 1).
        let f = GslibDirection {
            azm_deg: 90.0,
            atol_deg: 22.5,
            bandh: 1.0,
            dip_deg: 0.0,
            dtol_deg: 22.5,
            bandv: 25.0,
        }
        .to_filter()
        .unwrap();
        let result = compute_directional_variogram_3d(&dataset, &f, &cfg);
        assert!(matches!(result, Err(KrigingError::FittingError(_))));
    }
}
