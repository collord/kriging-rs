//! Internal 3-D anisotropy representation.
//!
//! Mathematically, [`Anisotropy3D`] is a pair `(R, s)` where `R` is a rotation
//! from world XYZ into the ellipsoid's principal axes and `s = (s₁, s₂, s₃)`
//! is a diagonal stretch along those principal axes. The transformation
//! applied to a world-frame lag vector `h` is:
//!
//! ```text
//!     h' = diag(s) · R · h
//! ```
//!
//! and the anisotropic distance is `‖h'‖₂`. This is the same metric that
//! GSLib's `setrot.f` produces — see [`crate::interop::gslib_anisotropy`] for
//! the conversion to and from GSLib's `(ang1, ang2, ang3, anis1, anis2)`
//! parameterization at the API boundary.
//!
//! The principal frame is the canonical "math" frame: right-handed XYZ,
//! z-positive-up, radians. GSLib conventions (azimuth clockwise from north,
//! dip positive downward, truncated-π `DEG2RAD`) live exclusively in the
//! interop module; nothing else in the crate sees them.

use nalgebra::{Matrix3, Vector3};

use crate::coord_3d::Coord3D;
use crate::error::KrigingError;

/// Anisotropy as a rotation + diagonal stretch.
///
/// Build with [`Anisotropy3D::identity`],
/// [`Anisotropy3D::from_rotation_matrix`], or
/// [`crate::interop::gslib_anisotropy::from_gslib`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anisotropy3D {
    /// Rotation from world XYZ into the ellipsoid's principal frame. Always
    /// stored as f64 in radians. Orthogonal with determinant +1; constructors
    /// that take user input validate this.
    rotation: Matrix3<f64>,
    /// Diagonal stretch along the principal axes. Components are typically
    /// `>= 1` (compression of the minor / vertical correlation lengths
    /// inflates anisotropic distance), but no sign constraint is imposed
    /// beyond strict positivity.
    stretch: Vector3<f64>,
}

impl Anisotropy3D {
    /// Identity anisotropy: no rotation, no stretch. The anisotropic distance
    /// is identical to the Cartesian Euclidean distance.
    #[inline]
    pub fn identity() -> Self {
        Self {
            rotation: Matrix3::identity(),
            stretch: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// Construct without validating the rotation matrix. Crate-private
    /// escape hatch for the GSLib interop module: `setrot`'s truncated-π
    /// constant introduces ~1e-8 orthogonality residual that the strict
    /// constructor rejects but is intentional GSLib parity. Do not use
    /// outside `interop`.
    #[doc(hidden)]
    pub(crate) fn from_rotation_matrix_unchecked(
        rotation: Matrix3<f64>,
        stretch: Vector3<f64>,
    ) -> Self {
        Self { rotation, stretch }
    }

    /// Construct directly from a rotation matrix and a diagonal stretch.
    /// `rotation` must be orthogonal (within `1e-10`) and have determinant
    /// `+1` (within `1e-10`); `stretch` components must be strictly positive
    /// and finite.
    pub fn from_rotation_matrix(
        rotation: Matrix3<f64>,
        stretch: Vector3<f64>,
    ) -> Result<Self, KrigingError> {
        // Orthogonality: R · Rᵀ should be I.
        let rrt = rotation * rotation.transpose();
        let identity = Matrix3::<f64>::identity();
        let ortho_err = (rrt - identity).abs().max();
        if ortho_err > 1e-10 {
            return Err(KrigingError::InvalidInput(format!(
                "rotation matrix is not orthogonal: max(|R·Rᵀ - I|) = {ortho_err:e}"
            )));
        }
        let det = rotation.determinant();
        if (det - 1.0).abs() > 1e-10 {
            return Err(KrigingError::InvalidInput(format!(
                "rotation matrix has determinant {det}, expected +1"
            )));
        }
        for (i, s) in stretch.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(KrigingError::InvalidInput(format!(
                    "stretch[{i}] = {s} is not strictly positive and finite"
                )));
            }
        }
        Ok(Self { rotation, stretch })
    }

    /// Borrow the rotation matrix.
    #[inline]
    pub fn rotation(&self) -> &Matrix3<f64> {
        &self.rotation
    }

    /// Borrow the stretch vector.
    #[inline]
    pub fn stretch(&self) -> &Vector3<f64> {
        &self.stretch
    }

    /// The combined deformation matrix `D = diag(stretch) · rotation`. This
    /// is the same shape of matrix GSLib's `setrot.f` produces.
    pub fn deformation_matrix(&self) -> Matrix3<f64> {
        let mut d = self.rotation;
        for i in 0..3 {
            for j in 0..3 {
                d[(i, j)] *= self.stretch[i];
            }
        }
        d
    }

    /// Apply the anisotropy transform to a lag vector. Operates on
    /// **differences** (`a - b`), not on absolute coordinates: anisotropy is
    /// a property of vectors in the variogram model's lag space.
    ///
    /// Returns the transformed lag as a `Coord3D` (i.e., `(x', y', z')` in
    /// the principal frame, after stretch).
    pub fn transform_lag(&self, h: Coord3D) -> Coord3D {
        let v = Vector3::new(h.x as f64, h.y as f64, h.z as f64);
        let hp = self.deformation_matrix() * v;
        Coord3D {
            x: hp[0] as crate::Real,
            y: hp[1] as crate::Real,
            z: hp[2] as crate::Real,
        }
    }

    /// Anisotropic distance between two world-frame points: `‖D · (a - b)‖`.
    pub fn anisotropic_distance(&self, a: Coord3D, b: Coord3D) -> f64 {
        let lag = Vector3::new((a.x - b.x) as f64, (a.y - b.y) as f64, (a.z - b.z) as f64);
        (self.deformation_matrix() * lag).norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identity_passes_through_distance() {
        let a = Coord3D::new(1.0, 2.0, 3.0);
        let b = Coord3D::new(4.0, 6.0, 15.0);
        let aniso = Anisotropy3D::identity();
        // 3-4-12 -> 13
        assert_relative_eq!(aniso.anisotropic_distance(a, b), 13.0, epsilon = 1e-9);
    }

    #[test]
    fn axis_aligned_stretch_scales_components() {
        // No rotation; stretch (1, 2, 4). Lag along y should be doubled,
        // along z should be quadrupled.
        let aniso =
            Anisotropy3D::from_rotation_matrix(Matrix3::identity(), Vector3::new(1.0, 2.0, 4.0))
                .unwrap();
        let origin = Coord3D::new(0.0, 0.0, 0.0);
        assert_relative_eq!(
            aniso.anisotropic_distance(origin, Coord3D::new(0.0, 5.0, 0.0)),
            10.0,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            aniso.anisotropic_distance(origin, Coord3D::new(0.0, 0.0, 3.0)),
            12.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn from_rotation_matrix_rejects_non_orthogonal() {
        let bad = Matrix3::new(1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let result = Anisotropy3D::from_rotation_matrix(bad, Vector3::new(1.0, 1.0, 1.0));
        assert!(matches!(result, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn from_rotation_matrix_rejects_reflection() {
        // Determinant -1 (a reflection, not a rotation).
        let reflect = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let result = Anisotropy3D::from_rotation_matrix(reflect, Vector3::new(1.0, 1.0, 1.0));
        assert!(matches!(result, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn from_rotation_matrix_rejects_nonpositive_stretch() {
        let result =
            Anisotropy3D::from_rotation_matrix(Matrix3::identity(), Vector3::new(1.0, 0.0, 1.0));
        assert!(matches!(result, Err(KrigingError::InvalidInput(_))));
        let result =
            Anisotropy3D::from_rotation_matrix(Matrix3::identity(), Vector3::new(1.0, -1.0, 1.0));
        assert!(matches!(result, Err(KrigingError::InvalidInput(_))));
    }

    #[test]
    fn rotation_invariance_under_axis_aligned_isotropy() {
        // 90 deg rotation about z. With isotropic stretch the anisotropic
        // distance should be unaffected by rotation.
        let cos = 0.0;
        let sin = 1.0;
        let rot_z = Matrix3::new(cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0);
        let aniso = Anisotropy3D::from_rotation_matrix(rot_z, Vector3::new(1.0, 1.0, 1.0)).unwrap();
        let origin = Coord3D::new(0.0, 0.0, 0.0);
        let p = Coord3D::new(3.0, 4.0, 0.0);
        assert_relative_eq!(aniso.anisotropic_distance(origin, p), 5.0, epsilon = 1e-9);
    }

    #[test]
    fn deformation_matrix_combines_rotation_and_stretch() {
        // Identity rotation + stretch (1, 2, 4) = diag(1, 2, 4).
        let aniso =
            Anisotropy3D::from_rotation_matrix(Matrix3::identity(), Vector3::new(1.0, 2.0, 4.0))
                .unwrap();
        let d = aniso.deformation_matrix();
        let expected = Matrix3::new(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 4.0);
        assert_relative_eq!(d, expected, epsilon = 1e-12);
    }
}
