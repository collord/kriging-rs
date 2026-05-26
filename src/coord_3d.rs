//! 3-D Cartesian coordinates with Euclidean distance.
//!
//! Mirrors the role of [`crate::projected::ProjectedCoord`] in 2-D. Right-handed
//! Cartesian with **z positive upward** as the canonical internal convention
//! (matches GSLib parity expectations); see
//! [`Orientation`] for the optional sign flip at load time.
//!
//! Distance is the bare Euclidean metric. Anisotropic distance is the
//! responsibility of [`crate::anisotropy_3d::Anisotropy3D`] (added at M3), which
//! operates on lag vectors produced by subtracting two `Coord3D`s.
//!
//! `Coord3D` does not carry a unit; the variogram's range parameter must be
//! expressed in the same linear units as the coordinates.

use crate::Real;

/// A 3-D Cartesian coordinate. Right-handed, z-positive-up by convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coord3D {
    pub x: Real,
    pub y: Real,
    pub z: Real,
}

/// Sign convention for the vertical axis at the user/IO boundary.
///
/// Internal code always assumes [`Orientation::ZPositiveUp`]. Construction via
/// [`Coord3D::with_orientation`] applies a sign flip on `z` when the caller's
/// source data uses depth conventions (e.g. drillhole depths, positive
/// downward).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    ZPositiveUp,
    ZPositiveDown,
}

impl Coord3D {
    #[inline]
    pub fn new(x: Real, y: Real, z: Real) -> Self {
        Self { x, y, z }
    }

    /// Construct a `Coord3D` from raw `(x, y, z)` in the caller's vertical-axis
    /// convention. The internal representation is always z-positive-up; if
    /// `orientation` is [`Orientation::ZPositiveDown`], the sign of `z` is
    /// flipped at construction time.
    #[inline]
    pub fn with_orientation(x: Real, y: Real, z: Real, orientation: Orientation) -> Self {
        match orientation {
            Orientation::ZPositiveUp => Self { x, y, z },
            Orientation::ZPositiveDown => Self { x, y, z: -z },
        }
    }

    /// Componentwise subtraction returning the lag vector `self - other`.
    #[inline]
    pub fn sub(self, other: Coord3D) -> Coord3D {
        Coord3D {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// Squared Euclidean distance between two points. Cheaper than
/// [`euclidean_distance_3d`] when only ordering matters or when the caller will
/// square-root the result anyway.
#[inline]
pub fn euclidean_distance_3d_squared(a: Coord3D, b: Coord3D) -> Real {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    dx * dx + dy * dy + dz * dz
}

/// Straight-line Euclidean distance between two points in 3-D.
#[inline]
pub fn euclidean_distance_3d(a: Coord3D, b: Coord3D) -> Real {
    euclidean_distance_3d_squared(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn distance_to_self_is_zero() {
        let p = Coord3D::new(1.0, 2.0, 3.0);
        assert_eq!(euclidean_distance_3d(p, p), 0.0);
    }

    #[test]
    fn distance_along_each_axis() {
        let origin = Coord3D::new(0.0, 0.0, 0.0);
        assert_relative_eq!(
            euclidean_distance_3d(origin, Coord3D::new(3.0, 0.0, 0.0)),
            3.0
        );
        assert_relative_eq!(
            euclidean_distance_3d(origin, Coord3D::new(0.0, 4.0, 0.0)),
            4.0
        );
        assert_relative_eq!(
            euclidean_distance_3d(origin, Coord3D::new(0.0, 0.0, 5.0)),
            5.0
        );
    }

    #[test]
    fn pythagorean_3d() {
        // 3-4-12 -> 13 (3^2 + 4^2 + 12^2 = 9 + 16 + 144 = 169)
        let d = euclidean_distance_3d(
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(3.0, 4.0, 12.0),
        );
        assert_relative_eq!(d, 13.0);
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Coord3D::new(1.0, 2.0, 3.0);
        let b = Coord3D::new(-4.0, 5.0, 6.0);
        assert_eq!(euclidean_distance_3d(a, b), euclidean_distance_3d(b, a));
    }

    #[test]
    fn squared_distance_avoids_sqrt() {
        let a = Coord3D::new(0.0, 0.0, 0.0);
        let b = Coord3D::new(3.0, 4.0, 12.0);
        assert_relative_eq!(euclidean_distance_3d_squared(a, b), 169.0);
    }

    #[test]
    fn z_positive_down_flips_sign_at_construction() {
        let depth = Coord3D::with_orientation(1.0, 2.0, 50.0, Orientation::ZPositiveDown);
        assert_eq!(depth.z, -50.0);
        let up = Coord3D::with_orientation(1.0, 2.0, 50.0, Orientation::ZPositiveUp);
        assert_eq!(up.z, 50.0);
    }

    #[test]
    fn sub_produces_lag_vector() {
        let a = Coord3D::new(5.0, 7.0, 9.0);
        let b = Coord3D::new(2.0, 3.0, 4.0);
        let lag = a.sub(b);
        assert_eq!(lag, Coord3D::new(3.0, 4.0, 5.0));
    }
}
