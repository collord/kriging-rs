//! 3-D universal kriging model (linear trend only in v1).
//!
//! Mirrors [`crate::kriging::ordinary_3d::OrdinaryKrigingModel3D`] but
//! wraps [`crate::kriging::solver::solve_universal_kriging_3d_linear`].
//! The trend basis is hard-wired to `[1, x, y, z]`; quadratic and
//! arbitrary-callback trends are deferred to v2 (per v3 §"UK trend
//! functions in v1") with explicit conditioning guidance — the linear
//! basis is already a 4-Lagrangian system and a quadratic basis
//! produces a 10-Lagrangian system that's much more often
//! ill-conditioned.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;
use crate::kriging::diagnostics::Prediction3D;
use crate::kriging::ordinary_3d::Neighborhood3D;
use crate::kriging::solver::{SolverConfig, solve_universal_kriging_3d_linear};
use crate::neighborhood::kdtree_3d::KdTree3D;
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::variogram::models::VariogramModel;

/// The trend basis used by [`UniversalKrigingModel3D`] in v1.
///
/// v1 hard-codes `Linear` (basis `[1, x, y, z]`). The enum exists as
/// the v2 forward-compatibility stub from v3 §"v2 forward-compatibility"
/// — `Quadratic` and `Custom(...)` variants land in v2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend3D {
    /// Linear trend `[1, x, y, z]`. Requires at least 4 samples that
    /// span the basis (not all coplanar).
    Linear,
}

/// 3-D universal kriging model.
#[derive(Debug, Clone)]
pub struct UniversalKrigingModel3D {
    coords: Vec<Coord3D>,
    values: Vec<Real>,
    variogram: VariogramModel,
    anisotropy: Anisotropy3D,
    trend: Trend3D,
    solver: SolverConfig,
    neighborhood: Option<Neighborhood3D>,
    kdtree: Option<Arc<KdTree3D>>,
}

impl UniversalKrigingModel3D {
    /// Build a model. Requires at least 4 samples (the linear-trend
    /// basis has 4 components); the solver's per-prediction failure
    /// surface also reports if the actual neighbor set doesn't span
    /// the basis at any individual target.
    pub fn new(
        dataset: PlanarDataset3D,
        anisotropy: Anisotropy3D,
        variogram: VariogramModel,
        trend: Trend3D,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        // Linear trend needs 4 samples globally; per-prediction
        // neighborhoods are checked at solve time.
        if coords.len() < 4 {
            return Err(KrigingError::InsufficientData(4));
        }
        Ok(Self {
            coords,
            values,
            variogram,
            anisotropy,
            trend,
            solver: SolverConfig::default(),
            neighborhood: None,
            kdtree: None,
        })
    }

    /// Override the solver configuration.
    pub fn with_solver_config(mut self, config: SolverConfig) -> Self {
        self.solver = config;
        self
    }

    /// Attach a neighborhood filter. Builds an anisotropy-aware kd-tree
    /// over the samples at construction. With UK and a neighborhood,
    /// every prediction's filtered neighbor set must still span the
    /// trend basis (4 samples in general position for linear); the
    /// solver returns `NonFiniteWeights` if it doesn't.
    pub fn with_neighborhood(mut self, neighborhood: Neighborhood3D) -> Self {
        self.kdtree = Some(Arc::new(KdTree3D::build(
            &self.coords,
            self.anisotropy,
        )));
        self.neighborhood = Some(neighborhood);
        self
    }

    /// Predict a single target.
    pub fn predict(&self, target: Coord3D) -> Result<Prediction3D, KrigingError> {
        let (sample_subset, value_subset) = match (&self.neighborhood, &self.kdtree) {
            (Some(nb), Some(tree)) => {
                let indices = self.select_neighbor_indices(tree, target, *nb);
                if indices.len() < 4 {
                    return Err(KrigingError::InsufficientData(4));
                }
                let mut s = Vec::with_capacity(indices.len());
                let mut v = Vec::with_capacity(indices.len());
                for i in indices {
                    s.push(self.coords[i]);
                    v.push(self.values[i]);
                }
                (s, v)
            }
            _ => (self.coords.clone(), self.values.clone()),
        };

        match self.trend {
            Trend3D::Linear => solve_universal_kriging_3d_linear(
                &sample_subset,
                &value_subset,
                target,
                &self.anisotropy,
                &self.variogram,
                &self.solver,
            ),
        }
        .map_err(|e| KrigingError::MatrixError(format!("{e}")))
    }

    /// Predict a batch of targets. Native: parallel via rayon. WASM:
    /// sequential. See [`super::ordinary_3d::OrdinaryKrigingModel3D::predict_batch`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn predict_batch(
        &self,
        targets: &[Coord3D],
    ) -> Result<Vec<Prediction3D>, KrigingError> {
        targets.par_iter().map(|t| self.predict(*t)).collect()
    }

    #[cfg(target_arch = "wasm32")]
    pub fn predict_batch(
        &self,
        targets: &[Coord3D],
    ) -> Result<Vec<Prediction3D>, KrigingError> {
        targets.iter().map(|t| self.predict(*t)).collect()
    }

    fn select_neighbor_indices(
        &self,
        tree: &KdTree3D,
        target: Coord3D,
        nb: Neighborhood3D,
    ) -> Vec<usize> {
        match (nb.max_radius, nb.max_neighbors) {
            (Some(radius), Some(k)) => tree
                .nearest_n(target, k)
                .into_iter()
                .take_while(|n| n.distance <= radius)
                .map(|n| n.index)
                .collect(),
            (Some(radius), None) => tree
                .within(target, radius)
                .into_iter()
                .map(|n| n.index)
                .collect(),
            (None, Some(k)) => tree
                .nearest_n(target, k)
                .into_iter()
                .map(|n| n.index)
                .collect(),
            (None, None) => (0..self.coords.len()).collect(),
        }
    }

    /// Borrow the trend type.
    pub fn trend(&self) -> Trend3D {
        self.trend
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn dataset_linear_trend() -> (PlanarDataset3D, Box<dyn Fn(Coord3D) -> Real>) {
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
        let trend = |c: Coord3D| 1.0 + 2.0 * c.x + 3.0 * c.y + 4.0 * c.z;
        let values: Vec<Real> = coords.iter().map(|c| trend(*c)).collect();
        (
            PlanarDataset3D::new(coords, values).unwrap(),
            Box::new(trend),
        )
    }

    fn variogram() -> VariogramModel {
        VariogramModel::new(0.0, 1.0, 10.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn predict_recovers_linear_trend_at_arbitrary_target() {
        let (dataset, trend) = dataset_linear_trend();
        let model = UniversalKrigingModel3D::new(
            dataset,
            Anisotropy3D::identity(),
            variogram(),
            Trend3D::Linear,
        )
        .unwrap();
        let target = Coord3D::new(3.0, 7.0, 2.5);
        let p = model.predict(target).unwrap();
        assert_relative_eq!(p.value as f64, trend(target) as f64, epsilon = 5e-3);
    }

    #[test]
    fn rejects_too_few_samples() {
        let dataset = PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(10.0, 0.0, 0.0),
                Coord3D::new(0.0, 10.0, 0.0),
            ],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        let result = UniversalKrigingModel3D::new(
            dataset,
            Anisotropy3D::identity(),
            variogram(),
            Trend3D::Linear,
        );
        assert!(matches!(result, Err(KrigingError::InsufficientData(4))));
    }

    #[test]
    fn predict_batch_matches_repeated_predict() {
        let (dataset, _trend) = dataset_linear_trend();
        let model = UniversalKrigingModel3D::new(
            dataset,
            Anisotropy3D::identity(),
            variogram(),
            Trend3D::Linear,
        )
        .unwrap();
        let targets = vec![
            Coord3D::new(2.0, 2.0, 2.0),
            Coord3D::new(7.0, 3.0, 1.0),
        ];
        let batch = model.predict_batch(&targets).unwrap();
        for (t, b) in targets.iter().zip(batch.iter()) {
            let single = model.predict(*t).unwrap();
            assert_eq!(single.value, b.value);
            assert_eq!(single.variance, b.variance);
        }
    }

    #[test]
    fn trend_accessor_returns_configured_trend() {
        let (dataset, _) = dataset_linear_trend();
        let model = UniversalKrigingModel3D::new(
            dataset,
            Anisotropy3D::identity(),
            variogram(),
            Trend3D::Linear,
        )
        .unwrap();
        assert_eq!(model.trend(), Trend3D::Linear);
    }
}
