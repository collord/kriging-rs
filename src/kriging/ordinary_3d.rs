//! 3-D ordinary kriging model wrapping the M8 robust solver.
//!
//! Mirrors upstream's [`crate::kriging::ordinary::OrdinaryKrigingModel`]
//! pattern: hold dataset + variogram + anisotropy + solver config, expose
//! `predict(target)` and `predict_batch(&[target])` returning
//! `Prediction3D`. The solver itself
//! ([`crate::kriging::solver::solve_ordinary_kriging_3d`]) handles the
//! Schur-complement Cholesky factor + nugget-inflation retry loop.
//!
//! v1 builds the kriging system from scratch per prediction (i.e., no
//! cached factorization across targets). This is the simpler shape for
//! M9 and works well when the dataset is small (n <~ 500) or when the
//! caller wants to vary anisotropy or neighborhood per prediction. A
//! cached-factorization path is a v2 optimization.

use std::num::NonZeroUsize;

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;
use crate::kriging::diagnostics::Prediction3D;
use crate::kriging::solver::{SolverConfig, solve_ordinary_kriging_3d};
use crate::neighborhood::kdtree_3d::KdTree3D;
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::variogram::models::VariogramModel;

/// Neighborhood restricting which samples enter each prediction's
/// kriging system. Mirrors upstream's 2-D [`crate::Neighborhood`] but
/// works in anisotropic-distance space (samples are filtered after the
/// kd-tree's anisotropy-aware ranking).
///
/// At least one of `max_radius` or `max_neighbors` must be set. When
/// both are set, the intersection is used. The radius is in
/// anisotropic-distance units (so for the isotropic case it's just
/// Euclidean distance in world units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighborhood3D {
    pub max_radius: Option<f64>,
    pub max_neighbors: Option<NonZeroUsize>,
}

impl Neighborhood3D {
    /// Convenience: neighborhood of all samples within an anisotropic
    /// radius, with no count cap.
    pub fn within_radius(radius: f64) -> Self {
        Self {
            max_radius: Some(radius),
            max_neighbors: None,
        }
    }

    /// Convenience: neighborhood of the `k` nearest samples (by
    /// anisotropic distance), with no radius cap.
    pub fn nearest(k: NonZeroUsize) -> Self {
        Self {
            max_radius: None,
            max_neighbors: Some(k),
        }
    }
}

/// 3-D ordinary kriging model.
///
/// Without a neighborhood (the default), every prediction uses all
/// samples (`n × n` kriging system per target). With a
/// [`Neighborhood3D`] set, predictions use only the samples that pass
/// the neighborhood filter; a kd-tree (anisotropy-aware) is built once
/// at construction time and reused.
#[derive(Debug, Clone)]
pub struct OrdinaryKrigingModel3D {
    coords: Vec<Coord3D>,
    values: Vec<Real>,
    variogram: VariogramModel,
    anisotropy: Anisotropy3D,
    solver: SolverConfig,
    neighborhood: Option<Neighborhood3D>,
    // Built lazily on first predict() call if neighborhood is set; cached
    // after. Wrapping in OnceCell would be nicer but Clone of a populated
    // OnceCell isn't free, so we keep the simpler Option<KdTree3D> here.
    // Built at construction in `new_with_neighborhood` so the build cost
    // doesn't surface as a per-predict latency surprise.
    kdtree: Option<std::sync::Arc<KdTree3D>>,
}

impl OrdinaryKrigingModel3D {
    /// Build a model from a dataset, anisotropy, and variogram. Uses the
    /// default [`SolverConfig`] and no neighborhood (every prediction
    /// uses all samples). Configure via [`Self::with_solver_config`] and
    /// [`Self::with_neighborhood`].
    pub fn new(
        dataset: PlanarDataset3D,
        anisotropy: Anisotropy3D,
        variogram: VariogramModel,
    ) -> Result<Self, KrigingError> {
        let (coords, values) = dataset.into_parts();
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        Ok(Self {
            coords,
            values,
            variogram,
            anisotropy,
            solver: SolverConfig::default(),
            neighborhood: None,
            kdtree: None,
        })
    }

    /// Override the solver configuration. Use the default unless you
    /// have a specific reason to tighten or loosen.
    pub fn with_solver_config(mut self, config: SolverConfig) -> Self {
        self.solver = config;
        self
    }

    /// Attach a [`Neighborhood3D`] filter. Builds an anisotropy-aware
    /// kd-tree over the samples for fast per-target neighbor lookup.
    pub fn with_neighborhood(mut self, neighborhood: Neighborhood3D) -> Self {
        self.kdtree = Some(std::sync::Arc::new(KdTree3D::build(
            &self.coords,
            self.anisotropy,
        )));
        self.neighborhood = Some(neighborhood);
        self
    }

    /// Predict a single target. Returns a [`Prediction3D`] with the
    /// kriged value, variance, condition-number proxy, and a flag for
    /// whether nugget inflation kicked in. Surfaces solver failures as
    /// [`KrigingError::MatrixError`] with the underlying
    /// [`crate::kriging::diagnostics::SolverFailure`] message.
    ///
    /// If a [`Neighborhood3D`] is attached, only the samples passing
    /// the filter are used in the kriging system.
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

        solve_ordinary_kriging_3d(
            &sample_subset,
            &value_subset,
            target,
            &self.anisotropy,
            &self.variogram,
            &self.solver,
        )
        .map_err(|e| KrigingError::MatrixError(format!("{e}")))
    }

    fn select_neighbor_indices(
        &self,
        tree: &KdTree3D,
        target: Coord3D,
        nb: Neighborhood3D,
    ) -> Vec<usize> {
        match (nb.max_radius, nb.max_neighbors) {
            (Some(radius), Some(k)) => {
                // Both set: nearest k within radius. nearest_n is
                // distance-sorted ascending; truncate at the first
                // distance exceeding radius.
                tree.nearest_n(target, k)
                    .into_iter()
                    .take_while(|n| n.distance <= radius)
                    .map(|n| n.index)
                    .collect()
            }
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

    /// Predict a batch of targets. Each prediction is independent; the
    /// solver state isn't cached across targets in v1.
    pub fn predict_batch(
        &self,
        targets: &[Coord3D],
    ) -> Result<Vec<Prediction3D>, KrigingError> {
        targets.iter().map(|t| self.predict(*t)).collect()
    }

    /// Borrow the underlying coordinates.
    pub fn coords(&self) -> &[Coord3D] {
        &self.coords
    }

    /// Borrow the underlying values.
    pub fn values(&self) -> &[Real] {
        &self.values
    }

    /// Borrow the anisotropy.
    pub fn anisotropy(&self) -> &Anisotropy3D {
        &self.anisotropy
    }

    /// Borrow the variogram model.
    pub fn variogram(&self) -> &VariogramModel {
        &self.variogram
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn three_point_dataset() -> PlanarDataset3D {
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

    fn exp_variogram() -> VariogramModel {
        VariogramModel::new(0.0, 1.0, 10.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn predict_recovers_sample_value_at_sample_location() {
        let model = OrdinaryKrigingModel3D::new(
            three_point_dataset(),
            Anisotropy3D::identity(),
            exp_variogram(),
        )
        .unwrap();
        // Target is the second sample.
        let p = model
            .predict(Coord3D::new(10.0, 0.0, 0.0))
            .expect("exact-match should solve");
        assert_relative_eq!(p.value as f64, 3.0, epsilon = 1e-3);
        assert!(
            (p.variance as f64).abs() < 1e-3,
            "variance at sample should be near zero, got {}",
            p.variance,
        );
    }

    #[test]
    fn predict_batch_matches_repeated_predict() {
        let model = OrdinaryKrigingModel3D::new(
            three_point_dataset(),
            Anisotropy3D::identity(),
            exp_variogram(),
        )
        .unwrap();
        let targets = vec![
            Coord3D::new(2.0, 2.0, 0.0),
            Coord3D::new(5.0, 5.0, 1.0),
            Coord3D::new(7.0, 3.0, 0.5),
        ];
        let batch = model.predict_batch(&targets).unwrap();
        for (t, b) in targets.iter().zip(batch.iter()) {
            let single = model.predict(*t).unwrap();
            assert_eq!(single.value, b.value);
            assert_eq!(single.variance, b.variance);
        }
    }

    #[test]
    fn solver_config_overrides_default() {
        // Set a ridiculously tight condition threshold; predictions for a
        // near-collinear configuration should fail rather than inflate.
        let dataset = PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(0.001, 0.0, 0.0),
                Coord3D::new(10.0, 0.0, 0.0),
            ],
            vec![1.0, 1.5, 3.0],
        )
        .unwrap();
        let model = OrdinaryKrigingModel3D::new(
            dataset,
            Anisotropy3D::identity(),
            exp_variogram(),
        )
        .unwrap()
        .with_solver_config(SolverConfig {
            condition_threshold: 2.0,
            max_retries: 0,
            ..SolverConfig::default()
        });
        let result = model.predict(Coord3D::new(5.0, 0.0, 0.0));
        assert!(
            result.is_err(),
            "tight solver config should surface a failure, got {:?}",
            result
        );
    }

    #[test]
    fn anisotropy_propagates_into_predictions() {
        let samples = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(0.0, 0.0, 10.0),
            Coord3D::new(10.0, 0.0, 0.0),
        ];
        let values = vec![1.0, 5.0, 3.0];
        let dataset = PlanarDataset3D::new(samples, values).unwrap();
        let target = Coord3D::new(0.0, 0.0, 5.0);

        let iso_model = OrdinaryKrigingModel3D::new(
            dataset.clone(),
            Anisotropy3D::identity(),
            exp_variogram(),
        )
        .unwrap();
        let aniso = Anisotropy3D::from_rotation_matrix(
            nalgebra::Matrix3::identity(),
            nalgebra::Vector3::new(1.0, 1.0, 10.0),
        )
        .unwrap();
        let aniso_model =
            OrdinaryKrigingModel3D::new(dataset, aniso, exp_variogram()).unwrap();

        let p_iso = iso_model.predict(target).unwrap();
        let p_aniso = aniso_model.predict(target).unwrap();
        assert!(
            (p_iso.value as f64 - p_aniso.value as f64).abs() > 0.01,
            "anisotropy should change the prediction: iso={}, aniso={}",
            p_iso.value,
            p_aniso.value,
        );
    }

    #[test]
    fn dataset_borrows_round_trip() {
        let model = OrdinaryKrigingModel3D::new(
            three_point_dataset(),
            Anisotropy3D::identity(),
            exp_variogram(),
        )
        .unwrap();
        assert_eq!(model.coords().len(), 3);
        assert_eq!(model.values().len(), 3);
        assert_eq!(model.variogram().variogram_type(), VariogramType::Exponential);
    }
}
