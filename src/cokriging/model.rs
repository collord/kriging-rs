//! Full (block) cokriging over a [`Coregionalization`].
//!
//! Given several variables and a Linear Model of Coregionalization giving every cross-covariance
//! `C_ij(h)`, [`CokrigingModel`] predicts any chosen *target variable* at new locations using
//! **all** variables' data. Variables may be sampled at the same locations (isotopic —
//! [`MultiVariableDataset`]) or at different ones (heterotopic — [`MultiVariableSamples`]); a
//! sparse primary borrowing strength from a dense secondary is the canonical heterotopic case.
//!
//! ## System
//!
//! Flatten the data by `(variable, sample) → offset(v) + α`, where `offset(v)` is the running
//! count of earlier variables' samples. The left-hand side is the full covariance matrix of the
//! data,
//!
//! ```text
//!   A[(v,α),(w,β)] = C_vw(u^v_α − u^w_β),
//! ```
//!
//! (block `(v, w)` is `n_v × n_w`, rectangular in the heterotopic case) and, for target variable
//! `t` at `u₀`, the right-hand side is `b[(v,α)] = C_vt(u^v_α − u₀)`. `A` does **not** depend on
//! the target, so it is factorized once and reused for every target variable and location.
//!
//! - **Simple** cokriging (known per-variable means `m_v`):
//!   `Z_t*(u₀) = m_t + Σ λ_{v,α}(Z_v(u^v_α) − m_v)`, variance `C_tt(0) − λᵀb`.
//! - **Ordinary** cokriging (unknown means): add `p` unbiasedness constraints
//!   `Σ_α λ_{v,α} = δ_{vt}` with Lagrange multipliers `μ_v`, giving a `(Σn_v + p)`-system;
//!   `Z_t*(u₀) = Σ λ_{v,α} Z_v(u^v_α)`, variance `C_tt(0) − λᵀb − μ_t`.
//!
//! With a single variable this reduces exactly to ordinary/simple kriging (see the tests); with
//! a block-diagonal LMC the secondary data receive zero weight.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Dyn, linalg::LU};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::distance::{GeoCoord, PreparedGeoCoord, haversine_distance_prepared, prepare_geo_coord};
use crate::error::KrigingError;
use crate::kriging::ordinary::Prediction;

use super::coregionalization::Coregionalization;
use super::dataset::{MultiVariableDataset, MultiVariableSamples};

/// Which cokriging estimator to build.
#[derive(Debug, Clone, PartialEq)]
pub enum CokrigingKind {
    /// Simple cokriging with a known mean per variable (length must equal the number of
    /// variables).
    Simple { means: Vec<Real> },
    /// Ordinary cokriging: per-variable means are unknown and estimated implicitly via
    /// unbiasedness constraints.
    Ordinary,
}

/// A factorized block cokriging model. Build with [`new`](Self::new) (isotopic) or
/// [`new_heterotopic`](Self::new_heterotopic); predict a chosen target variable at new locations
/// with [`predict`](Self::predict) / [`predict_batch`](Self::predict_batch).
#[derive(Debug)]
pub struct CokrigingModel {
    /// Prepared sample coordinates per variable.
    per_var_prepared: Vec<Vec<PreparedGeoCoord>>,
    /// Flat-index start of each variable: `offsets[v] = Σ_{w<v} n_w`.
    offsets: Vec<usize>,
    coregionalization: Coregionalization,
    n_variables: usize,
    /// Total data unknowns `Σ_v n_v`.
    n_data: usize,
    /// Number of Lagrange constraints (`0` for simple, `p` for ordinary).
    n_constraints: usize,
    /// Per-variable base added to a prediction (`m_v` for simple, `0` for ordinary).
    base: Vec<Real>,
    /// Flattened per-datum contribution: residual `Z_v − m_v` for simple, raw `Z_v` for ordinary.
    contribution: Vec<Real>,
    system_lu: Arc<LU<Real, Dyn, Dyn>>,
    ordinary: bool,
}

impl Clone for CokrigingModel {
    fn clone(&self) -> Self {
        Self {
            per_var_prepared: self.per_var_prepared.clone(),
            offsets: self.offsets.clone(),
            coregionalization: self.coregionalization.clone(),
            n_variables: self.n_variables,
            n_data: self.n_data,
            n_constraints: self.n_constraints,
            base: self.base.clone(),
            contribution: self.contribution.clone(),
            system_lu: Arc::clone(&self.system_lu),
            ordinary: self.ordinary,
        }
    }
}

impl CokrigingModel {
    /// Build from an **isotopic** dataset (all variables sampled at the same locations).
    ///
    /// Convenience wrapper over [`new_heterotopic`](Self::new_heterotopic).
    pub fn new(
        dataset: MultiVariableDataset,
        coregionalization: Coregionalization,
        kind: CokrigingKind,
    ) -> Result<Self, KrigingError> {
        Self::new_heterotopic(dataset.to_samples(), coregionalization, kind)
    }

    /// Build from **heterotopic** samples (each variable has its own locations).
    ///
    /// Errors:
    /// - [`KrigingError::DimensionMismatch`] if the coregionalization's variable count differs
    ///   from the samples', or (simple) the means length differs from the variable count.
    /// - [`KrigingError::MatrixError`] if the block system cannot be factorized.
    pub fn new_heterotopic(
        samples: MultiVariableSamples,
        coregionalization: Coregionalization,
        kind: CokrigingKind,
    ) -> Result<Self, KrigingError> {
        let p = samples.n_variables();
        if coregionalization.n_variables() != p {
            return Err(KrigingError::DimensionMismatch(format!(
                "coregionalization has {} variables but the samples have {p}",
                coregionalization.n_variables()
            )));
        }
        let (ordinary, base) = match &kind {
            CokrigingKind::Simple { means } => {
                if means.len() != p {
                    return Err(KrigingError::DimensionMismatch(format!(
                        "simple cokriging needs one mean per variable ({p}), got {}",
                        means.len()
                    )));
                }
                (false, means.clone())
            }
            CokrigingKind::Ordinary => (true, vec![0.0 as Real; p]),
        };

        let n_per_var: Vec<usize> = (0..p).map(|v| samples.n_points(v)).collect();
        let mut offsets = Vec::with_capacity(p);
        let mut acc = 0usize;
        for &n in &n_per_var {
            offsets.push(acc);
            acc += n;
        }
        let n_data = acc;

        let per_var_prepared: Vec<Vec<PreparedGeoCoord>> = (0..p)
            .map(|v| {
                samples
                    .coords(v)
                    .iter()
                    .copied()
                    .map(prepare_geo_coord)
                    .collect()
            })
            .collect();

        // Flattened contributions: residuals (simple) or raw values (ordinary).
        let mut contribution = vec![0.0 as Real; n_data];
        for v in 0..p {
            let vals = samples.values(v);
            for (alpha, &value) in vals.iter().enumerate() {
                contribution[offsets[v] + alpha] = value - base[v];
            }
        }

        // Per-variable diagonal jitter (sill-scaled), mirroring the univariate conditioning.
        let jitter: Vec<Real> = (0..p)
            .map(|v| {
                let var = coregionalization.cross_sill(v, v).max(0.0);
                (1e-6 * var * (n_per_var[v] as Real).sqrt()).max(1e-10)
            })
            .collect();

        let n_constraints = if ordinary { p } else { 0 };
        let dim = n_data + n_constraints;
        let mut a = DMatrix::<Real>::from_element(dim, dim, 0.0);

        // Data–data covariance blocks (fill each unordered pair once, then mirror).
        for v in 0..p {
            for alpha in 0..n_per_var[v] {
                let i = offsets[v] + alpha;
                for w in 0..p {
                    for beta in 0..n_per_var[w] {
                        let j = offsets[w] + beta;
                        if j < i {
                            continue;
                        }
                        let d = haversine_distance_prepared(
                            per_var_prepared[v][alpha],
                            per_var_prepared[w][beta],
                        );
                        let mut c = coregionalization.cross_covariance(v, w, d);
                        if i == j {
                            c += jitter[v];
                        }
                        a[(i, j)] = c;
                        a[(j, i)] = c;
                    }
                }
            }
        }

        // Unbiasedness constraints for ordinary cokriging: Σ_α λ_{v,α} = δ_{vt}.
        if ordinary {
            for v in 0..p {
                for alpha in 0..n_per_var[v] {
                    a[(offsets[v] + alpha, n_data + v)] = 1.0;
                    a[(n_data + v, offsets[v] + alpha)] = 1.0;
                }
            }
        }

        let lu = a.lu();
        let probe = DVector::from_element(dim, 1.0);
        if lu.solve(&probe).is_none() {
            return Err(KrigingError::MatrixError(
                "could not factorize cokriging block system".to_string(),
            ));
        }

        Ok(Self {
            per_var_prepared,
            offsets,
            coregionalization,
            n_variables: p,
            n_data,
            n_constraints,
            base,
            contribution,
            system_lu: Arc::new(lu),
            ordinary,
        })
    }

    /// Number of variables `p`.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    /// Total number of data values across all variables.
    #[inline]
    pub fn n_data(&self) -> usize {
        self.n_data
    }

    /// Predict `target_variable` at `target`.
    ///
    /// Errors with [`KrigingError::InvalidInput`] if `target_variable >= n_variables`.
    pub fn predict(
        &self,
        target_variable: usize,
        target: GeoCoord,
    ) -> Result<Prediction, KrigingError> {
        let mut rhs = DVector::from_element(self.n_data + self.n_constraints, 0.0);
        self.predict_with_rhs(target_variable, target, &mut rhs)
    }

    /// Predict `target_variable` at many locations. Parallel on native builds.
    pub fn predict_batch(
        &self,
        target_variable: usize,
        targets: &[GeoCoord],
    ) -> Result<Vec<Prediction>, KrigingError> {
        if target_variable >= self.n_variables {
            return Err(KrigingError::InvalidInput(format!(
                "target_variable {target_variable} out of range (n_variables = {})",
                self.n_variables
            )));
        }
        let dim = self.n_data + self.n_constraints;
        #[cfg(not(target_arch = "wasm32"))]
        {
            targets
                .par_iter()
                .map_init(
                    || DVector::<Real>::from_element(dim, 0.0),
                    |rhs, t| self.predict_with_rhs(target_variable, *t, rhs),
                )
                .collect()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let mut rhs = DVector::from_element(dim, 0.0);
            let mut out = Vec::with_capacity(targets.len());
            for t in targets {
                out.push(self.predict_with_rhs(target_variable, *t, &mut rhs)?);
            }
            Ok(out)
        }
    }

    fn predict_with_rhs(
        &self,
        target_variable: usize,
        target: GeoCoord,
        rhs: &mut DVector<Real>,
    ) -> Result<Prediction, KrigingError> {
        let p = self.n_variables;
        if target_variable >= p {
            return Err(KrigingError::InvalidInput(format!(
                "target_variable {target_variable} out of range (n_variables = {p})"
            )));
        }
        let prepared = prepare_geo_coord(target);
        // RHS: cross-covariance between each datum and the target variable at u₀.
        for v in 0..p {
            for (alpha, coord) in self.per_var_prepared[v].iter().enumerate() {
                let d = haversine_distance_prepared(*coord, prepared);
                rhs[self.offsets[v] + alpha] =
                    self.coregionalization
                        .cross_covariance(v, target_variable, d);
            }
        }
        if self.ordinary {
            for v in 0..p {
                rhs[self.n_data + v] = if v == target_variable { 1.0 } else { 0.0 };
            }
        }

        let x = self.system_lu.solve(rhs).ok_or_else(|| {
            KrigingError::MatrixError("could not solve cokriging block system".to_string())
        })?;

        let mut value = self.base[target_variable];
        let mut explained: Real = 0.0;
        for idx in 0..self.n_data {
            value += x[idx] * self.contribution[idx];
            explained += x[idx] * rhs[idx];
        }
        let c_tt0 = self
            .coregionalization
            .cross_sill(target_variable, target_variable);
        let mut variance = c_tt0 - explained;
        if self.ordinary {
            // − μ_t (the Lagrange multiplier for the target-variable constraint).
            variance -= x[self.n_data + target_variable];
        }
        Ok(Prediction {
            value,
            variance: variance.max(0.0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo_dataset::GeoDataset;
    use crate::kriging::ordinary::OrdinaryKrigingModel;
    use crate::kriging::simple::SimpleKrigingModel;
    use crate::variogram::models::{VariogramModel, VariogramType};
    use approx::assert_relative_eq;

    use super::super::coregionalization::{
        CoregionalizationStructure, CorrelationBasis, SillMatrix,
    };

    fn coords() -> Vec<GeoCoord> {
        vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
            GeoCoord::try_new(0.5, 0.5).unwrap(),
        ]
    }

    /// Single-structure LMC whose auto-covariance equals a zero-nugget variogram's covariance.
    fn one_variable_lmc(sill: Real, range: Real) -> (VariogramModel, Coregionalization) {
        let vg = VariogramModel::new(0.0, sill, range, VariogramType::Exponential).unwrap();
        let lmc = Coregionalization::new(vec![CoregionalizationStructure::new(
            CorrelationBasis::Model(vg),
            SillMatrix::from_rows(vec![vec![sill]]).unwrap(),
        )])
        .unwrap();
        (vg, lmc)
    }

    fn two_variable_lmc(cross: Real, range: Real) -> Coregionalization {
        let vg = VariogramModel::new(0.0, 1.0, range, VariogramType::Exponential).unwrap();
        Coregionalization::new(vec![CoregionalizationStructure::new(
            CorrelationBasis::Model(vg),
            SillMatrix::from_rows(vec![vec![2.0, cross], vec![cross, 1.0]]).unwrap(),
        )])
        .unwrap()
    }

    #[test]
    fn single_variable_simple_matches_simple_kriging() {
        let sill = 4.0;
        let range = 250.0;
        let (vg, lmc) = one_variable_lmc(sill, range);
        let values = vec![10.0, 12.0, 11.0, 13.0, 11.5];
        let mean = 11.5;
        let ds = MultiVariableDataset::new(coords(), vec![values.clone()]).unwrap();
        let ck = CokrigingModel::new(ds, lmc, CokrigingKind::Simple { means: vec![mean] }).unwrap();
        let sk =
            SimpleKrigingModel::new(GeoDataset::new(coords(), values).unwrap(), vg, mean).unwrap();
        for t in [
            GeoCoord::try_new(0.3, 0.4).unwrap(),
            GeoCoord::try_new(0.8, 0.2).unwrap(),
        ] {
            let a = ck.predict(0, t).unwrap();
            let b = sk.predict(t).unwrap();
            assert_relative_eq!(a.value, b.value, epsilon = 1e-3);
            assert_relative_eq!(a.variance, b.variance, epsilon = 1e-2);
        }
    }

    #[test]
    fn single_variable_ordinary_matches_ordinary_kriging() {
        let sill = 3.0;
        let range = 250.0;
        let (vg, lmc) = one_variable_lmc(sill, range);
        let values = vec![10.0, 12.0, 11.0, 13.0, 11.5];
        let ds = MultiVariableDataset::new(coords(), vec![values.clone()]).unwrap();
        let ck = CokrigingModel::new(ds, lmc, CokrigingKind::Ordinary).unwrap();
        let ok = OrdinaryKrigingModel::new(GeoDataset::new(coords(), values).unwrap(), vg).unwrap();
        for t in [
            GeoCoord::try_new(0.3, 0.4).unwrap(),
            GeoCoord::try_new(0.8, 0.2).unwrap(),
        ] {
            let a = ck.predict(0, t).unwrap();
            let b = ok.predict(t).unwrap();
            assert_relative_eq!(a.value, b.value, epsilon = 1e-2);
            assert_relative_eq!(a.variance, b.variance, epsilon = 5e-2);
        }
    }

    #[test]
    fn block_diagonal_lmc_ignores_secondary() {
        let vg = VariogramModel::new(0.0, 2.0, 250.0, VariogramType::Exponential).unwrap();
        let lmc2 = Coregionalization::new(vec![CoregionalizationStructure::new(
            CorrelationBasis::Model(vg),
            SillMatrix::from_rows(vec![vec![2.0, 0.0], vec![0.0, 5.0]]).unwrap(),
        )])
        .unwrap();
        let primary = vec![10.0, 12.0, 11.0, 13.0, 11.5];
        let secondary = vec![100.0, 200.0, 50.0, 300.0, 150.0];
        let ds = MultiVariableDataset::new(coords(), vec![primary.clone(), secondary]).unwrap();
        let ck = CokrigingModel::new(ds, lmc2, CokrigingKind::Ordinary).unwrap();
        let ok =
            OrdinaryKrigingModel::new(GeoDataset::new(coords(), primary).unwrap(), vg).unwrap();
        for t in [
            GeoCoord::try_new(0.35, 0.45).unwrap(),
            GeoCoord::try_new(0.7, 0.6).unwrap(),
        ] {
            let a = ck.predict(0, t).unwrap();
            let b = ok.predict(t).unwrap();
            assert_relative_eq!(a.value, b.value, epsilon = 1e-2);
        }
    }

    #[test]
    fn correlated_secondary_reduces_variance() {
        let primary = vec![10.0, 12.0, 11.0, 13.0, 11.5];
        let secondary = vec![1.0, 2.2, 1.1, 2.9, 1.6];
        let build = |cross: Real| {
            let ds = MultiVariableDataset::new(coords(), vec![primary.clone(), secondary.clone()])
                .unwrap();
            CokrigingModel::new(ds, two_variable_lmc(cross, 250.0), CokrigingKind::Ordinary)
                .unwrap()
        };
        let t = GeoCoord::try_new(0.4, 0.55).unwrap();
        let uncorrelated = build(0.0).predict(0, t).unwrap();
        let correlated = build(1.0).predict(0, t).unwrap();
        assert!(
            correlated.variance <= uncorrelated.variance + 1e-6,
            "correlated secondary should not increase variance: {} vs {}",
            correlated.variance,
            uncorrelated.variance
        );
    }

    #[test]
    fn heterotopic_matches_isotopic_when_locations_shared() {
        // Feeding the *same* locations for both variables via the heterotopic constructor must
        // reproduce the isotopic result exactly.
        let primary = vec![10.0, 12.0, 11.0, 13.0, 11.5];
        let secondary = vec![1.0, 2.2, 1.1, 2.9, 1.6];
        let lmc = two_variable_lmc(0.9, 250.0);
        let iso = CokrigingModel::new(
            MultiVariableDataset::new(coords(), vec![primary.clone(), secondary.clone()]).unwrap(),
            lmc.clone(),
            CokrigingKind::Ordinary,
        )
        .unwrap();
        let het = CokrigingModel::new_heterotopic(
            MultiVariableSamples::new(vec![(coords(), primary), (coords(), secondary)]).unwrap(),
            lmc,
            CokrigingKind::Ordinary,
        )
        .unwrap();
        for t in [
            GeoCoord::try_new(0.4, 0.55).unwrap(),
            GeoCoord::try_new(0.7, 0.3).unwrap(),
        ] {
            let a = iso.predict(0, t).unwrap();
            let b = het.predict(0, t).unwrap();
            assert_relative_eq!(a.value, b.value, epsilon = 1e-6);
            assert_relative_eq!(a.variance, b.variance, epsilon = 1e-6);
        }
    }

    #[test]
    fn heterotopic_predicts_sparse_primary_from_dense_secondary() {
        // The point of heterotopic cokriging: a primary with few samples, a secondary with many,
        // measured at different places. A strongly correlated secondary should let us predict the
        // primary, and pull the estimate toward the secondary's local level.
        let primary_coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let primary = vec![10.0, 12.0];
        let secondary_coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        // Strong positive cross-correlation (2x2 sill matrix must stay PSD: |cross| ≤ √(2·1)).
        let lmc = two_variable_lmc(1.35, 300.0);
        let samples = MultiVariableSamples::new(vec![
            (primary_coords, primary),
            (secondary_coords.clone(), vec![1.0, 5.0, 2.0, 3.0, 4.0]),
        ])
        .unwrap();
        let ck = CokrigingModel::new_heterotopic(samples, lmc, CokrigingKind::Ordinary).unwrap();

        // Predict the primary near a location where the secondary is high vs. low.
        let near_high_secondary = ck.predict(0, GeoCoord::try_new(0.5, 0.5).unwrap()).unwrap();
        let near_low_secondary = ck.predict(0, GeoCoord::try_new(0.0, 0.0).unwrap()).unwrap();
        assert!(near_high_secondary.value.is_finite());
        assert!(near_low_secondary.value.is_finite());
        // The secondary is highest at (0.5,0.5) and lowest at (0,0), so the primary estimate
        // near the former should exceed the latter.
        assert!(
            near_high_secondary.value > near_low_secondary.value,
            "primary near high secondary ({}) should exceed near low ({})",
            near_high_secondary.value,
            near_low_secondary.value
        );
    }

    #[test]
    fn rejects_shape_mismatches() {
        let vg = VariogramModel::new(0.0, 2.0, 250.0, VariogramType::Exponential).unwrap();
        let lmc1 = Coregionalization::new(vec![CoregionalizationStructure::new(
            CorrelationBasis::Model(vg),
            SillMatrix::from_rows(vec![vec![2.0]]).unwrap(),
        )])
        .unwrap();
        let ds = MultiVariableDataset::new(coords(), vec![vec![1.0; 5], vec![2.0; 5]]).unwrap();
        assert!(CokrigingModel::new(ds, lmc1, CokrigingKind::Ordinary).is_err());

        let lmc1b = Coregionalization::new(vec![CoregionalizationStructure::new(
            CorrelationBasis::Model(vg),
            SillMatrix::from_rows(vec![vec![2.0]]).unwrap(),
        )])
        .unwrap();
        let ds1 = MultiVariableDataset::new(coords(), vec![vec![1.0; 5]]).unwrap();
        assert!(
            CokrigingModel::new(
                ds1,
                lmc1b,
                CokrigingKind::Simple {
                    means: vec![0.0, 0.0]
                }
            )
            .is_err()
        );
    }
}
