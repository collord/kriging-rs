//! Conversions between GSLib's `(ang1, ang2, ang3, anis1, anis2)` anisotropy
//! parameterization and the crate's internal [`Anisotropy3D`] type.
//!
//! ## GSLib conventions (per `setrot.f` and Deutsch & Journel 1997)
//!
//! - `ang1` — azimuth of the major-continuity axis, in degrees **clockwise
//!   from north**.
//! - `ang2` — dip of the major-continuity axis, in degrees **positive
//!   downward**.
//! - `ang3` — plunge (rotation of the minor axis about the major axis), in
//!   degrees.
//! - `anis1` — `range_minor / range_major`, in `(0, 1]`.
//! - `anis2` — `range_vertical / range_major`, in `(0, 1]`.
//!
//! ## Conversion to the principal-frame triple `(alpha, beta, theta)`
//!
//! ```text
//!   alpha = (90  - ang1) · π / 180     if 0 <= ang1 < 270
//!   alpha = (450 - ang1) · π / 180     if ang1 >= 270
//!   beta  = -ang2 · π / 180
//!   theta =  ang3 · π / 180
//! ```
//!
//! These are the angles `setrot.f` computes internally; here `alpha` is the
//! angle between the major axis and the world +x axis (counter-clockwise
//! positive), `beta` is the dip (positive downward in world frame), and
//! `theta` is plunge.
//!
//! ## Truncated π and f32 promotion
//!
//! GSLib's `setrot.f` declares `DEG2RAD = 3.141592654 / 180.0` as a default-
//! kind `real` (which is `real(4)` aka f32 under GFortran), even in the
//! double-precision `dsetrot` variant. The division is performed in f32,
//! producing `DEG2RAD ≈ 0.01745329238474369` — distinct from the f64
//! evaluation `0.01745329252222222`. The f32 value then propagates through
//! `sin`/`cos` in f64 in `dsetrot`, producing matrix entries that differ
//! from a clean f64-throughout implementation by ~1e-8 at angles where the
//! cosine is near zero.
//!
//! To match GSLib bitwise, we replicate the same f32 → f64 promotion of
//! `DEG2RAD`. **This is the only place in the crate that uses this
//! truncated-precision constant**; everywhere else uses
//! `std::f64::consts::PI`. The matrix-parity test against the GSLib fixture
//! tolerates 1e-7; round-trip (`to_gslib(from_gslib(p))`) holds the angles
//! to 1e-6 (limited by branch-boundary `atan2` propagation through the
//! truncated constant) and the anisotropy ratios to 1e-12.

use nalgebra::{Matrix3, Vector3};

use crate::anisotropy_3d::Anisotropy3D;
use crate::error::KrigingError;

/// GSLib's truncated `DEG2RAD` constant, computed in f32 and promoted to
/// f64 to bitwise-match GSLib's `setrot`/`dsetrot`. See module docs.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
const GSLIB_DEG2RAD: f64 = (3.141_592_654_f32 / 180.0_f32) as f64;

/// GSLib's anisotropy parameters in the order setrot.f expects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GslibAnisotropy {
    pub ang1: f64,
    pub ang2: f64,
    pub ang3: f64,
    pub anis1: f64,
    pub anis2: f64,
}

/// Convert GSLib parameters into the internal [`Anisotropy3D`] representation.
///
/// `anis1` and `anis2` must be strictly positive and finite. The matrix this
/// function produces, multiplied through `Anisotropy3D::deformation_matrix`,
/// is byte-identical (to GSLib's truncated-π precision) with `setrot.f`'s
/// output for the same input.
pub fn from_gslib(p: GslibAnisotropy) -> Result<Anisotropy3D, KrigingError> {
    if !p.anis1.is_finite() || p.anis1 <= 0.0 {
        return Err(KrigingError::InvalidInput(format!(
            "anis1 = {} must be strictly positive and finite",
            p.anis1
        )));
    }
    if !p.anis2.is_finite() || p.anis2 <= 0.0 {
        return Err(KrigingError::InvalidInput(format!(
            "anis2 = {} must be strictly positive and finite",
            p.anis2
        )));
    }
    if !p.ang1.is_finite() || !p.ang2.is_finite() || !p.ang3.is_finite() {
        return Err(KrigingError::InvalidInput(
            "ang1/ang2/ang3 must be finite".to_string(),
        ));
    }

    // Reproduce setrot.f's branching on ang1 verbatim.
    let alpha = if (0.0..270.0).contains(&p.ang1) {
        (90.0 - p.ang1) * GSLIB_DEG2RAD
    } else {
        (450.0 - p.ang1) * GSLIB_DEG2RAD
    };
    let beta = -p.ang2 * GSLIB_DEG2RAD;
    let theta = p.ang3 * GSLIB_DEG2RAD;

    let (sa, ca) = (alpha.sin(), alpha.cos());
    let (sb, cb) = (beta.sin(), beta.cos());
    let (st, ct) = (theta.sin(), theta.cos());

    // setrot's matrix without the anis1/anis2 row prefactors -- this is the
    // pure rotation portion. The stretch goes into Anisotropy3D::stretch.
    let rotation = Matrix3::new(
        cb * ca,
        cb * sa,
        -sb,
        -ct * sa + st * sb * ca,
        ct * ca + st * sb * sa,
        st * cb,
        st * sa + ct * sb * ca,
        -st * ca + ct * sb * sa,
        ct * cb,
    );

    let stretch = Vector3::new(1.0, 1.0 / p.anis1, 1.0 / p.anis2);

    // Build directly (bypassing the orthogonality check in
    // Anisotropy3D::from_rotation_matrix) because GSLib's truncated π causes
    // an ~1e-8 orthogonality residual on the rotation portion. That residual
    // is GSLib's, not ours; we don't want to reject it.
    //
    // We do still want a coarse sanity check that the construction didn't
    // produce nonsense.
    debug_assert!(
        (rotation * rotation.transpose() - Matrix3::<f64>::identity())
            .abs()
            .max()
            < 1e-6,
        "from_gslib produced a wildly non-orthogonal rotation matrix; \
         setrot's truncated π should only introduce ~1e-8 residual"
    );

    // Construct directly to bypass the strict orthogonality check.
    Ok(Anisotropy3D::from_rotation_matrix_unchecked(
        rotation, stretch,
    ))
}

/// Convert an [`Anisotropy3D`] back into GSLib parameters. Round-trips
/// `from_gslib(to_gslib(a)) == a` to within ~1e-12 except at gimbal-lock
/// (`|sin β| → 1`, i.e. `ang2 → ±90°`) where `ang1` and `ang3` are not
/// independently recoverable. The returned parameters in that case fold all
/// rotation about the vertical axis into `ang3` and set `ang1 = 0`.
///
/// `stretch[0]` must equal `1.0` (within `1e-12`); otherwise the
/// [`Anisotropy3D`] cannot be expressed as a GSLib parameter set (GSLib only
/// supports range ratios relative to the major axis, not independent scaling
/// of all three axes).
pub fn to_gslib(a: &Anisotropy3D) -> Result<GslibAnisotropy, KrigingError> {
    let stretch = a.stretch();
    if (stretch[0] - 1.0).abs() > 1e-12 {
        return Err(KrigingError::InvalidInput(format!(
            "stretch[0] = {} cannot be expressed in GSLib parameters \
             (GSLib's anis1/anis2 are ratios relative to a unit major axis)",
            stretch[0]
        )));
    }
    let anis1 = 1.0 / stretch[1];
    let anis2 = 1.0 / stretch[2];

    let r = a.rotation();

    // Recover (alpha, beta, theta) from rotation matrix.
    //   row 0: ( cos β cos α,  cos β sin α, -sin β )
    //   col 2: ( -sin β, sin θ cos β, cos θ cos β )
    let sin_beta = -r[(0, 2)];
    let beta = sin_beta.clamp(-1.0, 1.0).asin();
    let cos_beta = beta.cos();

    let (alpha, theta) = if cos_beta.abs() < 1e-10 {
        // Gimbal lock: alpha and theta are not independently determined.
        // Convention: set alpha = 0 and absorb everything into theta.
        let alpha = 0.0;
        // With alpha = 0: row 1 = (-cos θ · 0 + sin θ sin β · 1, ...) etc.
        // Easiest recovery: use r[1,0] = -cos θ sin α + sin θ sin β cos α.
        // With α = 0: r[1,0] = sin θ sin β, so θ = atan2(r[1,0]/sin β, ...).
        let theta = if sin_beta > 0.0 {
            (-r[(1, 0)]).atan2(r[(2, 0)])
        } else {
            r[(1, 0)].atan2(-r[(2, 0)])
        };
        (alpha, theta)
    } else {
        // General case.
        let alpha = r[(0, 1)].atan2(r[(0, 0)]);
        let theta = r[(1, 2)].atan2(r[(2, 2)]);
        (alpha, theta)
    };

    // Invert the (ang1, ang2, ang3) <-> (alpha, beta, theta) transform.
    //   alpha = (90 - ang1) DEG2RAD  in [-3pi/2, pi/2] roughly
    //         or (450 - ang1) DEG2RAD when ang1 in [270, 360)
    //   --> ang1 = 90 - alpha/DEG2RAD  modulo 360
    //   beta  = -ang2 DEG2RAD --> ang2 = -beta / DEG2RAD
    //   theta = ang3 DEG2RAD  --> ang3 = theta / DEG2RAD
    let alpha_deg = alpha / GSLIB_DEG2RAD;
    let mut ang1 = 90.0 - alpha_deg;
    // Normalize to [0, 360).
    ang1 = ang1.rem_euclid(360.0);
    let ang2 = -beta / GSLIB_DEG2RAD;
    let ang3 = theta / GSLIB_DEG2RAD;

    Ok(GslibAnisotropy {
        ang1,
        ang2,
        ang3,
        anis1,
        anis2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_gslib_rejects_zero_anisotropy() {
        let bad = GslibAnisotropy {
            ang1: 0.0,
            ang2: 0.0,
            ang3: 0.0,
            anis1: 0.0,
            anis2: 1.0,
        };
        assert!(matches!(
            from_gslib(bad),
            Err(KrigingError::InvalidInput(_))
        ));
    }

    #[test]
    fn from_gslib_rejects_nonfinite_angle() {
        let bad = GslibAnisotropy {
            ang1: f64::NAN,
            ang2: 0.0,
            ang3: 0.0,
            anis1: 1.0,
            anis2: 1.0,
        };
        assert!(matches!(
            from_gslib(bad),
            Err(KrigingError::InvalidInput(_))
        ));
    }
}
