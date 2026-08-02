//! Collocated (simple) cokriging under the Markov Model 1 (MM1) screening hypothesis.
//!
//! When a densely sampled **secondary** variable is available at every prediction location, we
//! can improve interpolation of a sparsely sampled **primary** variable by borrowing strength
//! from the collocated secondary datum — without fitting a full cross-variogram. Under MM1 the
//! primary–secondary cross-covariance is proportional to the primary autocovariance,
//!
//! ```text
//!   C₁₂(h) = ρ · (σ₂ / σ₁) · C₁₁(h),
//! ```
//!
//! so the whole model needs only the primary [`VariogramModel`], the secondary variance, and a
//! single collocated correlation `ρ = corr(Z₁(u), Z₂(u))`.
//!
//! ## Estimator (simple cokriging, known means)
//!
//! With primary mean `m₁` and secondary mean `m₂`, at a target `u` with collocated secondary
//! value `z₂(u)`:
//!
//! ```text
//!   Z₁*(u) = m₁ + Σ_α w_α (Z₁(u_α) − m₁) + w_s (z₂(u) − m₂).
//! ```
//!
//! The `(n+1)`-dimensional collocated cokriging system reduces, via a Schur complement, to a
//! single solve against the primary simple-kriging matrix `R` plus a few scalars — so a
//! prediction costs the same as simple kriging once the model is built. Requiring `|ρ| < 1`
//! keeps the reduced system positive-definite.
//!
//! Ordinary (unknown-mean) collocated cokriging and full block cokriging over a
//! [`Coregionalization`](super::Coregionalization) are the next steps; see the module docs.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::distance::{GeoCoord, PreparedGeoCoord, haversine_distance_prepared, prepare_geo_coord};
use crate::error::KrigingError;
use crate::geo_dataset::GeoDataset;
use crate::kriging::ordinary::{Prediction, kriging_diagonal_jitter};
use crate::variogram::models::VariogramModel;

/// Characterization of the secondary variable for MM1 collocated cokriging: its global moments
/// and the collocated (lag-zero) correlation with the primary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SecondaryVariable {
    mean: Real,
    std_dev: Real,
    correlation: Real,
}

impl SecondaryVariable {
    /// Build from the secondary mean, standard deviation (`> 0`), and the collocated Pearson
    /// correlation `ρ` with the primary (`|ρ| < 1`, finite). `|ρ| = 1` is rejected: it makes
    /// the reduced system singular (the secondary would be a deterministic copy of the primary).
    pub fn new(mean: Real, std_dev: Real, correlation: Real) -> Result<Self, KrigingError> {
        if !mean.is_finite() {
            return Err(KrigingError::InvalidInput(
                "secondary mean must be finite".to_string(),
            ));
        }
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return Err(KrigingError::InvalidInput(
                "secondary standard deviation must be finite and positive".to_string(),
            ));
        }
        if !correlation.is_finite() || correlation.abs() >= 1.0 {
            return Err(KrigingError::InvalidInput(
                "collocated correlation must be finite and satisfy |ρ| < 1".to_string(),
            ));
        }
        Ok(Self {
            mean,
            std_dev,
            correlation,
        })
    }

    /// Estimate the secondary characterization from paired primary/secondary samples: the
    /// secondary sample mean and standard deviation, and the Pearson correlation between the
    /// two. Requires at least two pairs with positive variance in each variable.
    pub fn from_paired(primary: &[Real], secondary: &[Real]) -> Result<Self, KrigingError> {
        if primary.len() != secondary.len() {
            return Err(KrigingError::DimensionMismatch(
                "primary and secondary sample vectors must have equal length".to_string(),
            ));
        }
        let n = primary.len();
        if n < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        let nf = n as Real;
        let mean_p = primary.iter().copied().sum::<Real>() / nf;
        let mean_s = secondary.iter().copied().sum::<Real>() / nf;
        let (mut spp, mut sss, mut sps) = (0.0 as Real, 0.0 as Real, 0.0 as Real);
        for i in 0..n {
            let dp = primary[i] - mean_p;
            let ds = secondary[i] - mean_s;
            spp += dp * dp;
            sss += ds * ds;
            sps += dp * ds;
        }
        if spp <= 0.0 || sss <= 0.0 {
            return Err(KrigingError::InvalidInput(
                "cannot estimate correlation: a variable has zero sample variance".to_string(),
            ));
        }
        // Sample standard deviation (n−1 denominator) and Pearson correlation.
        let std_s = (sss / (nf - 1.0)).sqrt();
        let mut correlation = sps / (spp.sqrt() * sss.sqrt());
        // Clamp strictly inside (−1, 1) to keep the reduced system well-conditioned.
        let cap = 1.0 - 1e-6;
        correlation = correlation.clamp(-cap, cap);
        Self::new(mean_s, std_s, correlation)
    }

    /// Secondary mean `m₂`.
    #[inline]
    pub fn mean(&self) -> Real {
        self.mean
    }
    /// Secondary standard deviation `σ₂`.
    #[inline]
    pub fn std_dev(&self) -> Real {
        self.std_dev
    }
    /// Collocated correlation `ρ`.
    #[inline]
    pub fn correlation(&self) -> Real {
        self.correlation
    }
}

/// Fitted MM1 collocated simple cokriging model. Build with [`new`](Self::new), then
/// [`predict`](Self::predict) supplying the collocated secondary value at each target.
#[derive(Debug)]
pub struct CollocatedCokrigingModel {
    prepared_coords: Vec<PreparedGeoCoord>,
    primary_residuals: Vec<Real>,
    primary_mean: Real,
    variogram: VariogramModel,
    secondary: SecondaryVariable,
    /// Primary variance `σ₁² = C₁₁(0)`.
    sigma1_sq: Real,
    /// Secondary variance `σ₂²`.
    sigma2_sq: Real,
    /// MM1 cross scale `k = ρ·σ₂/σ₁`, so `C₁₂(h) = k·C₁₁(h)`.
    k: Real,
    /// Primary simple-kriging covariance matrix `R` (kept for reference/debugging).
    #[allow(dead_code)]
    system: DMatrix<Real>,
    /// Shared LU factorization of `R`; `Clone` just bumps the `Arc`.
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
}

impl Clone for CollocatedCokrigingModel {
    fn clone(&self) -> Self {
        Self {
            prepared_coords: self.prepared_coords.clone(),
            primary_residuals: self.primary_residuals.clone(),
            primary_mean: self.primary_mean,
            variogram: self.variogram,
            secondary: self.secondary,
            sigma1_sq: self.sigma1_sq,
            sigma2_sq: self.sigma2_sq,
            k: self.k,
            system: self.system.clone(),
            system_lu: Arc::clone(&self.system_lu),
        }
    }
}

impl CollocatedCokrigingModel {
    /// Build a collocated simple cokriging model from primary data, the primary variogram, a
    /// known primary mean `m₁`, and the secondary characterization.
    ///
    /// Errors on a degenerate primary variance (`C₁₁(0) ≤ 0`) or an unfactorable primary
    /// system.
    pub fn new(
        dataset: GeoDataset,
        variogram: VariogramModel,
        primary_mean: Real,
        secondary: SecondaryVariable,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        let prepared_coords = coords
            .iter()
            .copied()
            .map(prepare_geo_coord)
            .collect::<Vec<_>>();
        let primary_residuals: Vec<Real> = values.iter().map(|v| *v - primary_mean).collect();

        let sigma1_sq = variogram.covariance(0.0);
        if !sigma1_sq.is_finite() || sigma1_sq <= 0.0 {
            return Err(KrigingError::InvalidInput(
                "primary variogram has non-positive variance C(0); cannot cokrige".to_string(),
            ));
        }
        let sigma1 = sigma1_sq.sqrt();
        let sigma2_sq = secondary.std_dev * secondary.std_dev;
        let k = secondary.correlation * secondary.std_dev / sigma1;

        let system = build_primary_system(&prepared_coords, variogram);
        let system_lu = Arc::new(system.clone().lu());
        let probe = DVector::from_element(prepared_coords.len(), 1.0);
        if system_lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize primary cokriging system".to_string(),
            ));
        }

        Ok(Self {
            prepared_coords,
            primary_residuals,
            primary_mean,
            variogram,
            secondary,
            sigma1_sq,
            sigma2_sq,
            k,
            system,
            system_lu,
        })
    }

    /// Predict the primary variable at `target`, given the collocated secondary value there.
    pub fn predict(
        &self,
        target: GeoCoord,
        secondary_value: Real,
    ) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.prepared_coords.len(), 0.0);
        self.predict_with_rhs(target, secondary_value, &mut rhs)
    }

    /// Batch prediction. `targets` and `secondary_values` must have equal length. Parallel on
    /// native builds.
    pub fn predict_batch(
        &self,
        targets: &[GeoCoord],
        secondary_values: &[Real],
    ) -> Result<Vec<Prediction>, KrigingError> {
        if targets.len() != secondary_values.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "targets ({}) and secondary_values ({}) must have equal length",
                targets.len(),
                secondary_values.len()
            )));
        }
        let n = self.prepared_coords.len();
        #[cfg(not(target_arch = "wasm32"))]
        {
            targets
                .par_iter()
                .zip(secondary_values.par_iter())
                .map_init(
                    || DVector::<Real>::from_element(n, 0.0),
                    |rhs, (t, s)| self.predict_with_rhs(*t, *s, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(n, 0.0);
            let mut out = Vec::with_capacity(targets.len());
            for (t, s) in targets.iter().zip(secondary_values.iter()) {
                out.push(self.predict_with_rhs(*t, *s, &mut rhs)?);
            }
            Ok(out)
        }
    }

    fn predict_with_rhs(
        &self,
        target: GeoCoord,
        secondary_value: Real,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        let n = self.prepared_coords.len();
        let prepared = prepare_geo_coord(target);
        for i in 0..n {
            rhs[i] = self.variogram.covariance(haversine_distance_prepared(
                self.prepared_coords[i],
                prepared,
            ));
        }
        // Schur reduction of the (n+1) collocated system:
        //   g = R⁻¹ r,   d = rᵀg,
        //   w_s = k(σ₁² − d) / (σ₂² − k² d),   w = (1 − k·w_s) g.
        // The denominator equals σ₂²(1 − ρ²) at worst, so it is strictly positive for |ρ| < 1.
        let g = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve primary cokriging system".to_string())
        })?;
        let mut d: Real = 0.0;
        for i in 0..n {
            d += rhs[i] * g[i];
        }
        let denom = self.sigma2_sq - self.k * self.k * d;
        // Guarded: denom ≥ σ₂²(1 − ρ²) > 0, but clamp against round-off to stay finite.
        let denom = denom.max(Real::EPSILON);
        let w_s = self.k * (self.sigma1_sq - d) / denom;
        let scale = 1.0 - self.k * w_s;

        let mut residual_pred: Real = 0.0;
        for i in 0..n {
            residual_pred += scale * g[i] * self.primary_residuals[i];
        }
        let secondary_residual = secondary_value - self.secondary.mean;
        let value = self.primary_mean + residual_pred + w_s * secondary_residual;

        // Kriging variance: σ₁² − (wᵀr + w_s·k·σ₁²) with wᵀr = (1 − k·w_s)·d.
        let variance = (self.sigma1_sq - (scale * d + w_s * self.k * self.sigma1_sq)).max(0.0);
        Ok(Prediction { value, variance })
    }

    /// The secondary characterization used by this model.
    #[inline]
    pub fn secondary(&self) -> SecondaryVariable {
        self.secondary
    }
}

/// Primary simple-kriging covariance matrix `R` (`n × n`, `C₁₁(h)` with diagonal jitter).
fn build_primary_system(coords: &[PreparedGeoCoord], variogram: VariogramModel) -> DMatrix<Real> {
    let n = coords.len();
    let diag_eps = kriging_diagonal_jitter(n, variogram);
    let mut m = DMatrix::from_element(n, n, 0.0);
    for i in 0..n {
        for j in i..n {
            let mut cov = variogram.covariance(haversine_distance_prepared(coords[i], coords[j]));
            if i == j {
                cov += diag_eps;
            }
            m[(i, j)] = cov;
            m[(j, i)] = cov;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn setup() -> (Vec<GeoCoord>, Vec<Real>, VariogramModel) {
        let coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let values = vec![10.0, 12.0, 11.0, 13.0];
        let variogram = VariogramModel::new(0.01, 4.0, 300.0, VariogramType::Exponential).unwrap();
        (coords, values, variogram)
    }

    #[test]
    fn secondary_validation() {
        assert!(SecondaryVariable::new(0.0, 1.0, 0.5).is_ok());
        assert!(SecondaryVariable::new(0.0, 0.0, 0.5).is_err()); // std must be > 0
        assert!(SecondaryVariable::new(0.0, 1.0, 1.0).is_err()); // |ρ| < 1
        assert!(SecondaryVariable::new(0.0, 1.0, -1.2).is_err());
    }

    #[test]
    fn from_paired_recovers_correlation() {
        // y = 2x + const → perfect positive correlation (clamped below 1).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [12.0, 14.0, 16.0, 18.0, 20.0];
        let s = SecondaryVariable::from_paired(&x, &y).unwrap();
        assert!(s.correlation() > 0.999);
        assert_relative_eq!(s.mean(), 16.0, epsilon = 1e-6);
    }

    #[test]
    fn zero_correlation_matches_simple_kriging() {
        // With ρ = 0 the secondary is ignored, so collocated cokriging must reproduce simple
        // kriging exactly.
        use crate::kriging::simple::SimpleKrigingModel;
        let (coords, values, vg) = setup();
        let mean = 11.5;
        let sec = SecondaryVariable::new(0.0, 3.0, 0.0).unwrap();
        let ck = CollocatedCokrigingModel::new(
            GeoDataset::new(coords.clone(), values.clone()).unwrap(),
            vg,
            mean,
            sec,
        )
        .unwrap();
        let sk =
            SimpleKrigingModel::new(GeoDataset::new(coords, values).unwrap(), vg, mean).unwrap();
        for t in [
            GeoCoord::try_new(0.4, 0.6).unwrap(),
            GeoCoord::try_new(0.7, 0.2).unwrap(),
        ] {
            let a = ck.predict(t, 5.0).unwrap(); // secondary value irrelevant when ρ = 0
            let b = sk.predict(t).unwrap();
            assert_relative_eq!(a.value, b.value, epsilon = 1e-4);
            assert_relative_eq!(a.variance, b.variance, epsilon = 1e-4);
        }
    }

    #[test]
    fn secondary_pulls_prediction_and_reduces_variance() {
        let (coords, values, vg) = setup();
        let mean = 11.5;
        let sec = SecondaryVariable::new(0.0, 4.0, 0.8).unwrap();
        let ck =
            CollocatedCokrigingModel::new(GeoDataset::new(coords, values).unwrap(), vg, mean, sec)
                .unwrap();
        // A target away from data, with a strongly positive collocated secondary, should be
        // pulled above the primary mean; a strongly negative one, below.
        let t = GeoCoord::try_new(5.0, 5.0).unwrap();
        let high = ck.predict(t, 8.0).unwrap();
        let low = ck.predict(t, -8.0).unwrap();
        assert!(
            high.value > mean,
            "high secondary should pull up: {}",
            high.value
        );
        assert!(
            low.value < mean,
            "low secondary should pull down: {}",
            low.value
        );
        // A positive correlation with a known secondary cannot increase uncertainty vs. the
        // prior variance.
        assert!(high.variance <= vg.covariance(0.0) + 1e-5);
        assert!(high.variance >= 0.0);
    }

    #[test]
    fn batch_matches_single() {
        let (coords, values, vg) = setup();
        let sec = SecondaryVariable::new(1.0, 2.0, 0.5).unwrap();
        let ck =
            CollocatedCokrigingModel::new(GeoDataset::new(coords, values).unwrap(), vg, 11.5, sec)
                .unwrap();
        let targets = vec![
            GeoCoord::try_new(0.3, 0.3).unwrap(),
            GeoCoord::try_new(0.6, 0.7).unwrap(),
        ];
        let secs = vec![2.0, -1.0];
        let batch = ck.predict_batch(&targets, &secs).unwrap();
        for (i, t) in targets.iter().enumerate() {
            let single = ck.predict(*t, secs[i]).unwrap();
            assert_relative_eq!(batch[i].value, single.value, epsilon = 1e-5);
            assert_relative_eq!(batch[i].variance, single.variance, epsilon = 1e-5);
        }
        assert!(ck.predict_batch(&targets, &[1.0]).is_err());
    }
}
