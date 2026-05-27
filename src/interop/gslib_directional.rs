//! Conversion from GSLib's `gamv` direction parameters to the internal
//! [`crate::variogram::directional_3d::DirectionFilter3D`].
//!
//! ## GSLib convention (per `variograms.f90:295-322` in pygslib)
//!
//! - `azm` — direction azimuth in degrees **clockwise from north**.
//! - `atol` — azimuth tolerance (half-angle of the horizontal cone), degrees.
//! - `bandh` — horizontal bandwidth, maximum deviation perpendicular to
//!   the direction in the xy plane.
//! - `dip` — direction dip in degrees, **measured positive down from
//!   horizontal** (so dip=0 is horizontal, dip=90 is straight down).
//! - `dtol` — dip tolerance (half-angle of the vertical cone), degrees.
//! - `bandv` — vertical bandwidth, maximum deviation perpendicular to
//!   the dip direction.
//!
//! Inside gamv these are turned into two unit vectors:
//!
//! ```text
//!     azmuth     = (90 - azm) * π / 180        // math-style CCW from +x
//!     uvxazm     = cos(azmuth)
//!     uvyazm     = sin(azmuth)                  // horizontal direction
//!
//!     declin     = (90 - dip) * π / 180        // CCW from +z
//!     uvzdec     = cos(declin)                  // vertical component
//!     uvhdec     = sin(declin)                  // horizontal component
//! ```
//!
//! and three checks per candidate pair: azimuth cone, horizontal bandwidth,
//! dip cone, vertical bandwidth.
//!
//! ## Truncated π
//!
//! Unlike `setrot.f` (which has the f32-DEG2RAD trick), `gamv.f90`
//! computes `PI = 4.0*atan(1.0)` and uses `real*8` throughout. This means
//! the gamv direction vectors are at full f64 precision; there is no
//! ~1e-8 residual to match like in M3. Tolerance against gamv is limited
//! by the f32 storage in `Coord3D` (see crate-level `Real` discussion),
//! not by GSLib's constant.

use crate::error::KrigingError;
use crate::variogram::directional_3d::DirectionFilter3D;

/// GSLib gamv direction parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GslibDirection {
    pub azm_deg: f64,
    pub atol_deg: f64,
    pub bandh: f64,
    pub dip_deg: f64,
    pub dtol_deg: f64,
    pub bandv: f64,
}

impl GslibDirection {
    /// Convert to the internal direction filter. Angles are converted to
    /// radians; tolerances are stored as the cone half-angle cosines that
    /// gamv compares against.
    pub fn to_filter(self) -> Result<DirectionFilter3D, KrigingError> {
        if !self.bandh.is_finite() || self.bandh < 0.0 {
            return Err(KrigingError::InvalidInput(format!(
                "bandh = {} must be finite and non-negative",
                self.bandh
            )));
        }
        if !self.bandv.is_finite() || self.bandv < 0.0 {
            return Err(KrigingError::InvalidInput(format!(
                "bandv = {} must be finite and non-negative",
                self.bandv
            )));
        }

        // gamv uses full-precision PI; matches std::f64::consts::PI.
        let deg2rad = std::f64::consts::PI / 180.0;

        let azmuth = (90.0 - self.azm_deg) * deg2rad;
        let uvxazm = azmuth.cos();
        let uvyazm = azmuth.sin();

        let declin = (90.0 - self.dip_deg) * deg2rad;
        let uvzdec = declin.cos();
        let uvhdec = declin.sin();

        // gamv's <=0 fallback to 45 deg (cos(45 deg) cosine threshold).
        let csatol = if self.atol_deg <= 0.0 {
            (45.0 * deg2rad).cos()
        } else {
            (self.atol_deg * deg2rad).cos()
        };
        let csdtol = if self.dtol_deg <= 0.0 {
            (45.0 * deg2rad).cos()
        } else {
            (self.dtol_deg * deg2rad).cos()
        };

        Ok(DirectionFilter3D {
            uv_x_azm: uvxazm,
            uv_y_azm: uvyazm,
            uv_z_dec: uvzdec,
            uv_h_dec: uvhdec,
            cos_atol: csatol,
            cos_dtol: csdtol,
            bandh: self.bandh,
            bandv: self.bandv,
            // gamv treats atol >= 90 as "omnidirectional" for the purpose of
            // double-counting the reverse pair (see variograms.f90:418).
            // We mirror that flag.
            omni: self.atol_deg >= 90.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn azim_north_unit_vector() {
        // azm=0 (north), atol=22.5, no bandwidth limit.
        let g = GslibDirection {
            azm_deg: 0.0,
            atol_deg: 22.5,
            bandh: 1e10,
            dip_deg: 0.0,
            dtol_deg: 22.5,
            bandv: 1e10,
        };
        let f = g.to_filter().unwrap();
        // azm=0 -> azmuth = 90 deg -> (uvx, uvy) = (0, 1) pointing north.
        assert_relative_eq!(f.uv_x_azm, 0.0, epsilon = 1e-15);
        assert_relative_eq!(f.uv_y_azm, 1.0, epsilon = 1e-15);
        // dip=0 -> declin = 90 deg -> (uvz, uvh) = (0, 1).
        assert_relative_eq!(f.uv_z_dec, 0.0, epsilon = 1e-15);
        assert_relative_eq!(f.uv_h_dec, 1.0, epsilon = 1e-15);
        assert_relative_eq!(f.cos_atol, (22.5_f64.to_radians()).cos(), epsilon = 1e-15);
    }

    #[test]
    fn azim_east_points_east() {
        let g = GslibDirection {
            azm_deg: 90.0,
            atol_deg: 22.5,
            bandh: 25.0,
            dip_deg: 0.0,
            dtol_deg: 22.5,
            bandv: 25.0,
        };
        let f = g.to_filter().unwrap();
        // azm=90 -> azmuth = 0 deg -> (uvx, uvy) = (1, 0) pointing east.
        assert_relative_eq!(f.uv_x_azm, 1.0, epsilon = 1e-15);
        assert_relative_eq!(f.uv_y_azm, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn vertical_dip_points_down() {
        let g = GslibDirection {
            azm_deg: 0.0,
            atol_deg: 22.5,
            bandh: 25.0,
            dip_deg: 90.0,
            dtol_deg: 22.5,
            bandv: 25.0,
        };
        let f = g.to_filter().unwrap();
        // dip=90 -> declin = 0 deg -> (uvz, uvh) = (1, 0). Vertical direction.
        assert_relative_eq!(f.uv_z_dec, 1.0, epsilon = 1e-15);
        assert_relative_eq!(f.uv_h_dec, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn omni_flag_set_when_atol_at_least_90() {
        let g = GslibDirection {
            azm_deg: 0.0,
            atol_deg: 90.0,
            bandh: 1e10,
            dip_deg: 0.0,
            dtol_deg: 90.0,
            bandv: 1e10,
        };
        let f = g.to_filter().unwrap();
        assert!(f.omni);
    }

    #[test]
    fn rejects_negative_bandwidth() {
        let g = GslibDirection {
            azm_deg: 0.0,
            atol_deg: 22.5,
            bandh: -1.0,
            dip_deg: 0.0,
            dtol_deg: 22.5,
            bandv: 25.0,
        };
        assert!(matches!(g.to_filter(), Err(KrigingError::InvalidInput(_))));
    }
}
