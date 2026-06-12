//! Coordinate-value datasets for 3-D kriging.
//!
//! [`PlanarDataset3D`] pairs a list of [`Coord3D`] with a list of observed
//! values. Same invariants as [`crate::geo_dataset::GeoDataset`]:
//!
//! 1. `coords.len() == values.len()` (errors with [`KrigingError::DimensionMismatch`]).
//! 2. At least two observations are present (errors with [`KrigingError::InsufficientData`]).
//!
//! The name "planar" follows the v3 fork scope doc's terminology and matches
//! upstream's [`crate::projected::ProjectedDataset`] naming. It does **not**
//! imply 2-D — it means "Cartesian sample geometry," as opposed to geographic
//! lat/lon.

use crate::Real;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;

/// Coord-value pairs with matching length and at least two points, enforced at
/// construction.
#[derive(Debug, Clone)]
pub struct PlanarDataset3D {
    coords: Vec<Coord3D>,
    values: Vec<Real>,
}

impl PlanarDataset3D {
    pub fn new(coords: Vec<Coord3D>, values: Vec<Real>) -> Result<Self, KrigingError> {
        if coords.len() != values.len() {
            return Err(KrigingError::DimensionMismatch(format!(
                "coords ({}) and values ({}) must have equal length",
                coords.len(),
                values.len()
            )));
        }
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        Ok(Self { coords, values })
    }

    #[inline]
    pub fn coords(&self) -> &[Coord3D] {
        &self.coords
    }

    #[inline]
    pub fn values(&self) -> &[Real] {
        &self.values
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    pub fn into_parts(self) -> (Vec<Coord3D>, Vec<Real>) {
        (self.coords, self.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn three_points() -> (Vec<Coord3D>, Vec<Real>) {
        (
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(1.0, 0.0, 0.0),
                Coord3D::new(0.0, 1.0, 1.0),
            ],
            vec![1.0, 2.0, 3.0],
        )
    }

    #[test]
    fn constructs_when_lengths_match_and_n_is_sufficient() {
        let (coords, values) = three_points();
        let ds = PlanarDataset3D::new(coords, values).expect("valid input");
        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());
        assert_eq!(ds.coords().len(), 3);
        assert_eq!(ds.values().len(), 3);
    }

    #[test]
    fn rejects_length_mismatch() {
        let result = PlanarDataset3D::new(
            vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(1.0, 0.0, 0.0)],
            vec![1.0],
        );
        assert!(matches!(result, Err(KrigingError::DimensionMismatch(_))));
    }

    #[test]
    fn rejects_empty_input() {
        let result = PlanarDataset3D::new(vec![], vec![]);
        assert!(matches!(result, Err(KrigingError::InsufficientData(2))));
    }

    #[test]
    fn rejects_single_point() {
        let result = PlanarDataset3D::new(vec![Coord3D::new(0.0, 0.0, 0.0)], vec![1.0]);
        assert!(matches!(result, Err(KrigingError::InsufficientData(2))));
    }

    #[test]
    fn into_parts_round_trips() {
        let (coords, values) = three_points();
        let coords_clone = coords.clone();
        let values_clone = values.clone();
        let ds = PlanarDataset3D::new(coords, values).unwrap();
        let (c, v) = ds.into_parts();
        assert_eq!(c, coords_clone);
        assert_eq!(v, values_clone);
    }
}
