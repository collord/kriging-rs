//! 3-D simple kriging model.
//!
//! Mirrors [`crate::kriging::ordinary_3d::OrdinaryKrigingModel3D`] but
//! wraps [`crate::kriging::solver::solve_simple_kriging_3d`] instead.
//! SK assumes a known global mean and **does not enforce `Σλ = 1`** —
//! the prediction is `m + λᵀ(z − m)` and the variance is
//! `C(0) − λᵀk₀` (no Lagrangian, no μ).
//!
//! Both v1 OK and SK share [`crate::kriging::ordinary_3d::Neighborhood3D`].

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;
use crate::kriging::diagnostics::Prediction3D;
use crate::kriging::ordinary_3d::Neighborhood3D;
use crate::kriging::solver::{SolverConfig, solve_simple_kriging_3d};
use crate::neighborhood::kdtree_3d::KdTree3D;
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::variogram::models::VariogramModel;

/// 3-D simple kriging model with a known global mean.
#[derive(Debug, Clone)]
pub struct SimpleKrigingModel3D {
    coords: Vec<Coord3D>,
    values: Vec<Real>,
    mean: Real,
    variogram: VariogramModel,
    anisotropy: Anisotropy3D,
    solver: SolverConfig,
    neighborhood: Option<Neighborhood3D>,
    kdtree: Option<Arc<KdTree3D>>,
}

impl SimpleKrigingModel3D {
    /// Build a model from a dataset, anisotropy, variogram, and known
    /// global mean. The mean must be a finite Real; non-finite values
    /// are rejected as `KrigingError::InvalidInput`.
    pub fn new(
        dataset: PlanarDataset3D,
        anisotropy: Anisotropy3D,
        variogram: VariogramModel,
        mean: Real,
    ) -> Result<Self, KrigingError> {
        if !mean.is_finite() {
            return Err(KrigingError::InvalidInput(format!(
                "mean must be finite, got {mean}"
            )));
        }
        let (coords, values) = dataset.into_parts();
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        Ok(Self {
            coords,
            values,
            mean,
            variogram,
            anisotropy,
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
    /// over the samples at construction.
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
                if indices.len() < 2 {
                    return Err(KrigingError::InsufficientData(2));
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

        solve_simple_kriging_3d(
            &sample_subset,
            &value_subset,
            self.mean,
            target,
            &self.anisotropy,
            &self.variogram,
            &self.solver,
        )
        .map_err(|e| KrigingError::MatrixError(format!("{e}")))
    }

    /// Predict a batch of targets. Native: parallel via rayon. WASM:
    /// sequential. See [`super::ordinary_3d::OrdinaryKrigingModel3D::predict_batch`]
    /// for the parallelism rationale.
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

    /// Borrow the known global mean.
    pub fn mean(&self) -> Real {
        self.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn dataset() -> PlanarDataset3D {
        PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(10.0, 0.0, 0.0),
                Coord3D::new(0.0, 10.0, 0.0),
            ],
            vec![1.0, 3.0, 5.0],
        )
        .unwrap()
    }

    fn variogram() -> VariogramModel {
        VariogramModel::new(0.0, 1.0, 10.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn predict_recovers_sample_value_at_sample_location() {
        let model =
            SimpleKrigingModel3D::new(dataset(), Anisotropy3D::identity(), variogram(), 2.0)
                .unwrap();
        let p = model.predict(Coord3D::new(10.0, 0.0, 0.0)).unwrap();
        assert_relative_eq!(p.value as f64, 3.0, epsilon = 1e-3);
        assert!((p.variance as f64).abs() < 1e-3);
    }

    #[test]
    fn predict_approaches_mean_far_from_samples() {
        let model =
            SimpleKrigingModel3D::new(dataset(), Anisotropy3D::identity(), variogram(), 7.0)
                .unwrap();
        let p = model.predict(Coord3D::new(1000.0, 1000.0, 1000.0)).unwrap();
        assert_relative_eq!(p.value as f64, 7.0, epsilon = 1e-3);
    }

    #[test]
    fn rejects_non_finite_mean() {
        let result = SimpleKrigingModel3D::new(
            dataset(),
            Anisotropy3D::identity(),
            variogram(),
            Real::NAN,
        );
        assert!(matches!(result, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn predict_batch_matches_repeated_predict() {
        let model =
            SimpleKrigingModel3D::new(dataset(), Anisotropy3D::identity(), variogram(), 0.0)
                .unwrap();
        let targets = vec![
            Coord3D::new(2.0, 2.0, 0.0),
            Coord3D::new(5.0, 5.0, 1.0),
        ];
        let batch = model.predict_batch(&targets).unwrap();
        for (t, b) in targets.iter().zip(batch.iter()) {
            let single = model.predict(*t).unwrap();
            assert_eq!(single.value, b.value);
            assert_eq!(single.variance, b.variance);
        }
    }
}
