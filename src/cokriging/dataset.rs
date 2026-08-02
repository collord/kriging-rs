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

    /// View this isotopic dataset as the general [`MultiVariableSamples`] form (each variable
    /// sharing the same locations).
    pub fn to_samples(&self) -> MultiVariableSamples {
        let per_variable = self
            .variables
            .iter()
            .map(|values| (self.coords.clone(), values.clone()))
            .collect();
        // Invariants already hold, so construction cannot fail.
        MultiVariableSamples::new(per_variable)
            .expect("isotopic dataset always yields valid heterotopic samples")
    }
}

/// **Heterotopic** multi-variable samples: each variable has its **own** sample locations (and
/// possibly a different number of them).
///
/// This is the general input for cokriging when variables are measured at different places — for
/// example a sparsely sampled primary and a densely sampled secondary. The isotopic
/// [`MultiVariableDataset`] is the special case where every variable shares the same locations
/// (convert with [`MultiVariableDataset::to_samples`]).
///
/// Invariants, enforced at construction:
/// - at least one variable;
/// - each variable's `coords` and `values` have equal length;
/// - each variable has at least one sample (drop a variable you have no data for);
/// - at least two samples in total.
#[derive(Debug, Clone)]
pub struct MultiVariableSamples {
    coords: Vec<Vec<GeoCoord>>,
    values: Vec<Vec<Real>>,
}

impl MultiVariableSamples {
    /// Build from one `(coords, values)` pair per variable.
    pub fn new(per_variable: Vec<(Vec<GeoCoord>, Vec<Real>)>) -> Result<Self, KrigingError> {
        if per_variable.is_empty() {
            return Err(KrigingError::InvalidInput(
                "heterotopic samples require at least one variable".to_string(),
            ));
        }
        let mut coords = Vec::with_capacity(per_variable.len());
        let mut values = Vec::with_capacity(per_variable.len());
        let mut total = 0usize;
        for (v, (c, val)) in per_variable.into_iter().enumerate() {
            if c.len() != val.len() {
                return Err(KrigingError::DimensionMismatch(format!(
                    "variable {v} has {} coords but {} values",
                    c.len(),
                    val.len()
                )));
            }
            if c.is_empty() {
                return Err(KrigingError::InsufficientData(1));
            }
            total += c.len();
            coords.push(c);
            values.push(val);
        }
        if total < 2 {
            return Err(KrigingError::InsufficientData(2));
        }
        Ok(Self { coords, values })
    }

    /// Number of variables `p`.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.coords.len()
    }

    /// Number of samples of variable `v`.
    #[inline]
    pub fn n_points(&self, v: usize) -> usize {
        self.coords[v].len()
    }

    /// Total number of samples across all variables.
    pub fn total_points(&self) -> usize {
        self.coords.iter().map(Vec::len).sum()
    }

    /// Sample locations of variable `v`.
    #[inline]
    pub fn coords(&self, v: usize) -> &[GeoCoord] {
        &self.coords[v]
    }

    /// Sample values of variable `v`.
    #[inline]
    pub fn values(&self, v: usize) -> &[Real] {
        &self.values[v]
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

    #[test]
    fn heterotopic_samples_allow_per_variable_locations() {
        let primary_coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(1.0, 1.0).unwrap(),
        ];
        let secondary_coords = vec![
            GeoCoord::try_new(0.0, 0.0).unwrap(),
            GeoCoord::try_new(0.5, 0.5).unwrap(),
            GeoCoord::try_new(1.0, 0.0).unwrap(),
        ];
        let s = MultiVariableSamples::new(vec![
            (primary_coords, vec![10.0, 20.0]),
            (secondary_coords, vec![1.0, 2.0, 3.0]),
        ])
        .unwrap();
        assert_eq!(s.n_variables(), 2);
        assert_eq!(s.n_points(0), 2);
        assert_eq!(s.n_points(1), 3);
        assert_eq!(s.total_points(), 5);
        assert_eq!(s.values(1), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn heterotopic_rejects_empty_variable_and_mismatch() {
        // A variable with no samples is rejected (drop it instead).
        assert!(MultiVariableSamples::new(vec![(vec![], vec![])]).is_err());
        // coords/values length mismatch.
        assert!(
            MultiVariableSamples::new(vec![(
                vec![GeoCoord::try_new(0.0, 0.0).unwrap()],
                vec![1.0, 2.0]
            )])
            .is_err()
        );
    }

    #[test]
    fn isotopic_converts_to_samples() {
        let ds =
            MultiVariableDataset::new(coords(), vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
                .unwrap();
        let s = ds.to_samples();
        assert_eq!(s.n_variables(), 2);
        assert_eq!(s.n_points(0), 3);
        assert_eq!(s.coords(0), ds.coords());
    }
}
