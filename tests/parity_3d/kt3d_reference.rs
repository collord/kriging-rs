//! GSLib `kt3d` algorithmic-tier parity for 3D ordinary/simple/universal
//! kriging. Implemented at M9 (OK) and M10 (SK/UK).
//!
//! Predicted values and variances at 1000 random locations must match
//! kt3d/geostatspy within 1e-6 (f64) / 1e-4 (f32).

#[test]
#[ignore = "M9: not yet implemented"]
fn ordinary_kriging_3d_matches_kt3d_at_1000_locations() {
    todo!("M9: predict at 1000 random locations, compare to kt3d output within 1e-6");
}

#[test]
#[ignore = "M10: not yet implemented"]
fn simple_kriging_3d_matches_kt3d_at_1000_locations() {
    todo!("M10: same as OK but for simple kriging");
}

#[test]
#[ignore = "M10: not yet implemented"]
fn universal_kriging_3d_linear_trend_matches_kt3d() {
    todo!("M10: linear-trend UK against kt3d");
}
