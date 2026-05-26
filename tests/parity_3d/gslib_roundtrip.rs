//! GSLib anisotropy round-trip parity. Implemented at M3.
//!
//! Validates that `interop::gslib_anisotropy::from_gslib` followed by
//! `to_gslib` round-trips to 1e-12, and that the rotation matrix produced by
//! `from_gslib` matches GSLib's `setrot.f` output on a curated table of
//! ~30 known cases.

#[test]
#[ignore = "M3: not yet implemented"]
fn from_gslib_matches_setrot_reference_table() {
    todo!("M3: load fixture table of (ang1, ang2, ang3, anis1, anis2) -> matrix");
}

#[test]
#[ignore = "M3: not yet implemented"]
fn gslib_round_trip_within_1e_minus_12() {
    todo!("M3: from_gslib(to_gslib(a)) == a within 1e-12");
}
