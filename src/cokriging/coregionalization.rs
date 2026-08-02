//! Linear Model of Coregionalization (LMC) — the cross-covariance abstraction.
//!
//! A [`Coregionalization`] models the joint second-order structure of several variables as a
//! sum of *basic structures*, each a normalized spatial correlation `ρ_k(h)` scaled by a
//! positive-semidefinite `p × p` **coregionalization (sill) matrix** `B_k`:
//!
//! ```text
//!   C_ij(h) = Σ_k  B_k[i, j] · ρ_k(h)
//! ```
//!
//! Admissibility (a valid, positive-definite multivariate covariance) requires every `B_k` to
//! be symmetric positive-semidefinite; this is enforced at construction by [`SillMatrix::new`].
//!
//! This is the foundation the full (block) cokriging solver builds on: it turns "which
//! variogram and which cross-correlations" into a single object that can answer
//! [`cross_covariance`](Coregionalization::cross_covariance) for any variable pair and lag. The
//! collocated cokriging in [`super::collocated`] is a shortcut that does *not* need the full
//! LMC (only a single collocated correlation), but it is the same family of models: MM1 is the
//! special case of a single intrinsic structure with a `2 × 2` sill matrix.

use crate::Real;
use crate::error::KrigingError;
use crate::variogram::models::VariogramModel;
use nalgebra::DMatrix;

/// A normalized spatial correlation function `ρ(h)` with `ρ(0) = 1` used as one basic
/// structure of an LMC. The structure's *variance contribution* lives in its
/// [`SillMatrix`], so only the shape/range matters here.
#[derive(Debug, Clone, Copy)]
pub enum CorrelationBasis {
    /// Pure nugget: `ρ(h) = 1` at `h = 0`, else `0`. Use a dedicated nugget structure for the
    /// discontinuity rather than the nugget of a [`VariogramModel`] (which is normalized away).
    Nugget,
    /// Normalized covariance `ρ(h) = C(h) / C(0)` of a [`VariogramModel`]. Only the model's
    /// shape and range enter (its own sill and nugget cancel in the ratio), so build these
    /// from zero-nugget models and add a separate [`Nugget`](Self::Nugget) structure if needed.
    Model(VariogramModel),
}

impl CorrelationBasis {
    /// Correlation at lag `h` (clamped to `[0, 1]`).
    pub fn correlation(&self, h: Real) -> Real {
        match self {
            Self::Nugget => {
                if h <= 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Model(model) => {
                let c0 = model.covariance(0.0);
                if c0 <= 0.0 {
                    // Unbounded models (e.g. Power) have no finite variance; not usable as a
                    // normalized correlation basis.
                    return 0.0;
                }
                (model.covariance(h.max(0.0)) / c0).clamp(0.0, 1.0)
            }
        }
    }
}

/// A symmetric positive-semidefinite `p × p` coregionalization (sill) matrix `B_k`.
///
/// Entry `[i, j]` is the sill contribution of structure `k` to the covariance between
/// variables `i` and `j`. Symmetry and positive-semidefiniteness are validated at
/// construction (the admissibility condition for the LMC).
#[derive(Debug, Clone)]
pub struct SillMatrix {
    n: usize,
    /// Row-major `n × n`.
    entries: Vec<Real>,
}

impl SillMatrix {
    /// Build from a row-major `n × n` buffer. Errors if the length is wrong, any entry is
    /// non-finite, the matrix is not symmetric, or it is not positive-semidefinite.
    pub fn new(n: usize, entries: Vec<Real>) -> Result<Self, KrigingError> {
        if n == 0 {
            return Err(KrigingError::InvalidInput(
                "sill matrix must have at least one variable".to_string(),
            ));
        }
        if entries.len() != n * n {
            return Err(KrigingError::DimensionMismatch(format!(
                "sill matrix expected {} entries for {n}×{n}, got {}",
                n * n,
                entries.len()
            )));
        }
        if entries.iter().any(|v| !v.is_finite()) {
            return Err(KrigingError::InvalidInput(
                "sill matrix entries must be finite".to_string(),
            ));
        }
        let scale = entries.iter().fold(0.0 as Real, |a, b| a.max(b.abs()));
        // Symmetry within tolerance.
        let sym_tol = scale.max(1.0) * 1e-6;
        for i in 0..n {
            for j in (i + 1)..n {
                if (entries[i * n + j] - entries[j * n + i]).abs() > sym_tol {
                    return Err(KrigingError::InvalidInput(format!(
                        "sill matrix must be symmetric (entry [{i},{j}] != [{j},{i}])"
                    )));
                }
            }
        }
        // Positive-semidefinite: smallest eigenvalue must not be meaningfully negative.
        let m = DMatrix::<Real>::from_row_slice(n, n, &entries);
        let eig = m.symmetric_eigen();
        let min_eig = eig
            .eigenvalues
            .iter()
            .copied()
            .fold(Real::INFINITY, |a, b| a.min(b));
        let neg_tol = scale.max(1.0) * 1e-5;
        if min_eig < -neg_tol {
            return Err(KrigingError::InvalidInput(format!(
                "coregionalization sill matrix is not positive-semidefinite \
                 (min eigenvalue {min_eig} < -{neg_tol}); the cross-sills are inadmissible"
            )));
        }
        Ok(Self { n, entries })
    }

    /// Build from rows (each row length `n`).
    pub fn from_rows(rows: Vec<Vec<Real>>) -> Result<Self, KrigingError> {
        let n = rows.len();
        let mut entries = Vec::with_capacity(n * n);
        for row in &rows {
            if row.len() != n {
                return Err(KrigingError::DimensionMismatch(format!(
                    "sill matrix rows must all have length {n}"
                )));
            }
            entries.extend_from_slice(row);
        }
        Self::new(n, entries)
    }

    /// Number of variables `p`.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.n
    }

    /// Entry `[i, j]`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Real {
        self.entries[i * self.n + j]
    }
}

/// One basic structure of the LMC: a spatial correlation shape and its sill matrix.
#[derive(Debug, Clone)]
pub struct CoregionalizationStructure {
    basis: CorrelationBasis,
    sills: SillMatrix,
}

impl CoregionalizationStructure {
    /// Pair a correlation basis with a (PSD) sill matrix.
    pub fn new(basis: CorrelationBasis, sills: SillMatrix) -> Self {
        Self { basis, sills }
    }

    /// Number of variables of this structure's sill matrix.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.sills.n_variables()
    }
}

/// A Linear Model of Coregionalization: `C_ij(h) = Σ_k B_k[i, j]·ρ_k(h)`.
///
/// All structures must share the same number of variables. Build with [`new`](Self::new); query
/// with [`cross_covariance`](Self::cross_covariance) / [`cross_sill`](Self::cross_sill). This is
/// the model a full cokriging solver consumes (see the module docs).
#[derive(Debug, Clone)]
pub struct Coregionalization {
    n_variables: usize,
    structures: Vec<CoregionalizationStructure>,
}

impl Coregionalization {
    /// Assemble an LMC from one or more structures (all with matching variable counts).
    pub fn new(structures: Vec<CoregionalizationStructure>) -> Result<Self, KrigingError> {
        let first = structures.first().ok_or_else(|| {
            KrigingError::InvalidInput(
                "coregionalization requires at least one structure".to_string(),
            )
        })?;
        let n_variables = first.n_variables();
        if structures.iter().any(|s| s.n_variables() != n_variables) {
            return Err(KrigingError::DimensionMismatch(
                "all coregionalization structures must have the same number of variables"
                    .to_string(),
            ));
        }
        Ok(Self {
            n_variables,
            structures,
        })
    }

    /// Number of variables `p`.
    #[inline]
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    /// Number of basic structures.
    #[inline]
    pub fn n_structures(&self) -> usize {
        self.structures.len()
    }

    /// Cross-covariance `C_ij(h) = Σ_k B_k[i, j]·ρ_k(h)` between variables `i` and `j`.
    ///
    /// Panics if `i` or `j` is out of range (`>= n_variables`).
    pub fn cross_covariance(&self, i: usize, j: usize, h: Real) -> Real {
        assert!(
            i < self.n_variables && j < self.n_variables,
            "variable index out of range"
        );
        self.structures
            .iter()
            .map(|s| s.sills.get(i, j) * s.basis.correlation(h))
            .sum()
    }

    /// Auto-covariance `C_ii(h)` of a single variable.
    #[inline]
    pub fn auto_covariance(&self, variable: usize, h: Real) -> Real {
        self.cross_covariance(variable, variable, h)
    }

    /// Cross-sill `C_ij(0) = Σ_k B_k[i, j]` (variance for `i == j`).
    pub fn cross_sill(&self, i: usize, j: usize) -> Real {
        assert!(
            i < self.n_variables && j < self.n_variables,
            "variable index out of range"
        );
        self.structures.iter().map(|s| s.sills.get(i, j)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;
    use approx::assert_relative_eq;

    fn exp_shape(range: Real) -> CorrelationBasis {
        CorrelationBasis::Model(
            VariogramModel::new(0.0, 1.0, range, VariogramType::Exponential).unwrap(),
        )
    }

    #[test]
    fn correlation_basis_starts_at_one_and_decays() {
        let b = exp_shape(100.0);
        assert_relative_eq!(b.correlation(0.0), 1.0, epsilon = 1e-6);
        assert!(b.correlation(50.0) < 1.0 && b.correlation(50.0) > 0.0);
        assert!(b.correlation(1e6) < 1e-3);
        // Nugget basis.
        assert_relative_eq!(
            CorrelationBasis::Nugget.correlation(0.0),
            1.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            CorrelationBasis::Nugget.correlation(0.5),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn sill_matrix_rejects_asymmetric_and_indefinite() {
        // Asymmetric.
        assert!(SillMatrix::from_rows(vec![vec![1.0, 0.2], vec![0.9, 1.0]]).is_err());
        // Indefinite: [[1, 2], [2, 1]] has eigenvalues 3 and -1.
        assert!(SillMatrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 1.0]]).is_err());
        // Valid PSD (correlation-like) matrix.
        assert!(SillMatrix::from_rows(vec![vec![1.0, 0.6], vec![0.6, 1.0]]).is_ok());
    }

    #[test]
    fn cross_covariance_at_zero_equals_cross_sill() {
        let sills = SillMatrix::from_rows(vec![vec![4.0, 1.2], vec![1.2, 9.0]]).unwrap();
        let lmc = Coregionalization::new(vec![CoregionalizationStructure::new(
            exp_shape(80.0),
            sills,
        )])
        .unwrap();
        assert_eq!(lmc.n_variables(), 2);
        assert_relative_eq!(lmc.cross_covariance(0, 0, 0.0), 4.0, epsilon = 1e-5);
        assert_relative_eq!(lmc.cross_covariance(0, 1, 0.0), 1.2, epsilon = 1e-5);
        assert_relative_eq!(lmc.cross_sill(1, 1), 9.0, epsilon = 1e-6);
        // Off-origin: scales the sill by the (decaying) shared correlation.
        let rho = exp_shape(80.0).correlation(40.0);
        assert_relative_eq!(lmc.cross_covariance(0, 1, 40.0), 1.2 * rho, epsilon = 1e-5);
    }

    #[test]
    fn nested_structures_sum() {
        let nugget = CoregionalizationStructure::new(
            CorrelationBasis::Nugget,
            SillMatrix::from_rows(vec![vec![0.5, 0.0], vec![0.0, 0.5]]).unwrap(),
        );
        let structured = CoregionalizationStructure::new(
            exp_shape(50.0),
            SillMatrix::from_rows(vec![vec![2.0, 0.8], vec![0.8, 3.0]]).unwrap(),
        );
        let lmc = Coregionalization::new(vec![nugget, structured]).unwrap();
        assert_eq!(lmc.n_structures(), 2);
        // At h=0 both structures contribute: variance = 0.5 + 2.0 for var 0.
        assert_relative_eq!(lmc.auto_covariance(0, 0.0), 2.5, epsilon = 1e-5);
        // Just off the origin the nugget drops out, only the structured part remains.
        let rho = exp_shape(50.0).correlation(5.0);
        assert_relative_eq!(lmc.auto_covariance(0, 5.0), 2.0 * rho, epsilon = 1e-5);
    }

    #[test]
    fn mismatched_variable_counts_are_rejected() {
        let a = CoregionalizationStructure::new(
            CorrelationBasis::Nugget,
            SillMatrix::from_rows(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap(),
        );
        let b = CoregionalizationStructure::new(
            CorrelationBasis::Nugget,
            SillMatrix::from_rows(vec![vec![1.0]]).unwrap(),
        );
        assert!(Coregionalization::new(vec![a, b]).is_err());
    }
}
