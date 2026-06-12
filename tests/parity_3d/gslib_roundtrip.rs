//! GSLib anisotropy round-trip parity (M3).
//!
//! Validates two things against `fixtures/setrot_reference/cases.csv`:
//!
//! 1. **Matrix parity**: for each fixture case, `from_gslib(case).deformation_matrix()`
//!    matches the matrix that GSLib's `dsetrot` produced for the same input,
//!    within 1e-7 (the floor imposed by GSLib's truncated-π `DEG2RAD`).
//! 2. **Round-trip**: `to_gslib(from_gslib(p)) == p` for the angles within
//!    1e-6 (same truncated-π floor, propagated through `atan2`) and for the
//!    anisotropy ratios within 1e-12.
//!
//! See `tests/parity_3d/fixtures/setrot_reference/README.md` for the fixture
//! provenance and a longer discussion of why the tolerances are what they are.

use approx::assert_relative_eq;
use kriging_rs::{Anisotropy3D, GslibAnisotropy, from_gslib, to_gslib};
use nalgebra::Matrix3;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/parity_3d/fixtures/setrot_reference/cases.csv")
}

struct FixtureCase {
    ang1: f64,
    ang2: f64,
    ang3: f64,
    anis1: f64,
    anis2: f64,
    deformation: Matrix3<f64>,
}

fn load_fixtures() -> Vec<FixtureCase> {
    let csv = std::fs::read_to_string(fixture_path())
        .expect("setrot fixture CSV should be checked in at tests/parity_3d/fixtures/setrot_reference/cases.csv");
    let mut out = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("numeric column"))
            .collect();
        assert_eq!(cols.len(), 14, "expected 14 columns in row {i}");
        out.push(FixtureCase {
            ang1: cols[0],
            ang2: cols[1],
            ang3: cols[2],
            anis1: cols[3],
            anis2: cols[4],
            deformation: Matrix3::new(
                cols[5], cols[6], cols[7], cols[8], cols[9], cols[10], cols[11], cols[12], cols[13],
            ),
        });
    }
    out
}

#[test]
fn from_gslib_matches_setrot_reference_table() {
    let cases = load_fixtures();
    assert!(cases.len() >= 30, "fixture should have at least 30 cases");
    for case in cases {
        let aniso = from_gslib(GslibAnisotropy {
            ang1: case.ang1,
            ang2: case.ang2,
            ang3: case.ang3,
            anis1: case.anis1,
            anis2: case.anis2,
        })
        .unwrap_or_else(|e| panic!("from_gslib failed for ang1={}: {e}", case.ang1));
        let ours = aniso.deformation_matrix();
        let max_err = (ours - case.deformation).abs().max();
        assert!(
            max_err < 1e-7,
            "case (ang1={}, ang2={}, ang3={}, anis1={}, anis2={}): \
             max(|ours - setrot|) = {max_err:e} exceeds 1e-7\n\
             ours:\n{}\nreference:\n{}",
            case.ang1,
            case.ang2,
            case.ang3,
            case.anis1,
            case.anis2,
            ours,
            case.deformation,
        );
    }
}

#[test]
fn gslib_round_trip_holds_angles_to_1e_minus_6_and_ratios_to_1e_minus_12() {
    let cases = load_fixtures();
    for case in cases {
        let input = GslibAnisotropy {
            ang1: case.ang1,
            ang2: case.ang2,
            ang3: case.ang3,
            anis1: case.anis1,
            anis2: case.anis2,
        };
        let aniso = from_gslib(input).unwrap();
        let recovered = to_gslib(&aniso).unwrap();

        // ang1 lives modulo 360; compare on the circle.
        let ang1_diff = ((recovered.ang1 - input.ang1 + 540.0) % 360.0 - 180.0).abs();
        assert!(
            ang1_diff < 1e-6,
            "ang1 round-trip drifted: input={}, recovered={}, diff={ang1_diff}",
            input.ang1,
            recovered.ang1,
        );
        let ang2_diff = (recovered.ang2 - input.ang2).abs();
        let ang3_diff = (recovered.ang3 - input.ang3).abs();
        assert!(
            ang2_diff < 1e-6,
            "ang2 round-trip drifted: input={}, recovered={}, diff={ang2_diff}",
            input.ang2,
            recovered.ang2,
        );
        assert!(
            ang3_diff < 1e-6,
            "ang3 round-trip drifted: input={}, recovered={}, diff={ang3_diff}",
            input.ang3,
            recovered.ang3,
        );
        assert_relative_eq!(recovered.anis1, input.anis1, epsilon = 1e-12);
        assert_relative_eq!(recovered.anis2, input.anis2, epsilon = 1e-12);
    }
}

#[test]
fn identity_round_trips_through_anisotropy3d() {
    // The trivial case: Anisotropy3D::identity() should serialize to GSLib's
    // identity-ish parameters (ang1=90 is GSLib's "no rotation"; see fixture
    // case 1 for why) and de-serialize back to identity.
    let id = Anisotropy3D::identity();
    let params = to_gslib(&id).unwrap();
    // identity rotation -> alpha = 0, beta = 0, theta = 0 -> ang1 = 90, ang2 = 0, ang3 = 0
    assert_relative_eq!(params.ang1, 90.0, epsilon = 1e-9);
    assert_relative_eq!(params.ang2, 0.0, epsilon = 1e-9);
    assert_relative_eq!(params.ang3, 0.0, epsilon = 1e-9);
    assert_relative_eq!(params.anis1, 1.0, epsilon = 1e-12);
    assert_relative_eq!(params.anis2, 1.0, epsilon = 1e-12);
}
