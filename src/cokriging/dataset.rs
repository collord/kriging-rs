//! Multi-variable datasets for cokriging.
//!
//! [`MultiVariableDataset`] pairs a set of locations with **several** value vectors — one per
//! variable — sampled at the *same* locations (the isotopic case). It is the multivariate
//! analogue of [`GeoDataset`](crate::GeoDataset) and the input a full cokriging solver will
//! consume alongside a [`Coregionalization`](super::Coregionalization).
//!
//! Heterotopic data (variables sampled at different locations) is a later extension; the
//! isotopic case keeps the block assembly square and is the common starting point.

use crate::Real;
use crate::distance::GeoCoord;
use crate::error::KrigingError;

/// Locations with one value per variable at each location (isotopic sampling).
///
/// Invariants, enforced at construction:
/// - at least one variable;
/// - every variable's value vector has the same length as `coords`;
/// - at least two locations (a single point defines no covariance structure).
#[derive(Debug, Clone)]
pub struct MultiVariableDataset {
    coords: Vec<GeoCoord>,
    /// `variables[v][i]` is the value of variable `v` at `coords[i]`.
    variables: Vec<Vec<Real>>,
}

impl MultiVariableDataset {
    /// Build a dataset from shared `coords` and one value vector per variable.
    ///
    /// Errors:
    /// - [`KrigingError::InvalidInput`] when there are no variables.
    /// - [`KrigingError::InsufficientData`] when there are fewer than two locations.
    /// - [`KrigingError::DimensionMismatch`] when a variable's length differs from `coords`.
    pub fn new(coords: Vec<GeoCoord>, variables: Vec<Vec<Real>>) -> Result<Self, KrigingError> {
        if variables.is_empty() {
            return Err(KrigingError::InvalidInput(
                "multi-variable dataset requires at least one variable".to_string(),
            ));
        }
        if coords.len() < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        for (v, values) in variables.iter().enumerate() {
            if values.len() != coords.len() {
                return Err(KrigingError::DimensionMismatch(format!(
                    "variable {v} has {} values but there are {} locations",
                    values.len(),
                    coords.len()
                )));
            }
        }
        Ok(Self { coords, variables })
    }

    /// Number of variables `p`.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Number of locations `n`.
    #[inline]
    pub fn n_points(&self) -> usize {
        self.coords.len()
    }

    /// Shared sample locations.
    #[inline]
    pub fn coords(&self) -> &[GeoCoord] {
        &self.coords
    }

    /// Values of variable `v` at every location.
    #[inline]
    pub fn variable(&self, v: usize) -> &[Real] {
        &self.variables[v]
    }

    /// Extract a single variable as a [`GeoDataset`](crate::GeoDataset)-shaped pair, e.g. to run
    /// univariate kriging on one component.
    pub fn variable_pair(&self, v: usize) -> (Vec<GeoCoord>, Vec<Real>) {
        (self.coords.clone(), self.variables[v].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coords() -> Vec<GeoCoord> {
        vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.0, 1.0).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ]
    }

    #[test]
    fn builds_and_exposes_variables() {
        let ds =
            MultiVariableDataset::new(coords(), vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]])
                .unwrap();
        assert_eq!(ds.n_variables(), 2);
        assert_eq!(ds.n_points(), 3);
        assert_eq!(ds.variable(1), &[10.0, 20.0, 30.0]);
        let (c, v) = ds.variable_pair(0);
        assert_eq!(c.len(), 3);
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_bad_shapes() {
        assert!(MultiVariableDataset::new(coords(), vec![]).is_err());
        assert!(MultiVariableDataset::new(coords(), vec![vec![1.0, 2.0]]).is_err());
        let single = vec![GeoCoord::try_new(0.0, 0.0).unwrap()];
        assert!(MultiVariableDataset::new(single, vec![vec![1.0]]).is_err());
    }
}
