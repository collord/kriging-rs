//! GSLib `gamv` directional pair-set bitwise parity. Implemented at M6.
//!
//! For a given dataset and gamv parameters, the set of pairs that pass the
//! filter must be exactly identical to gamv's. Pair counts per bin must match
//! exactly. Validated by emitting pair indices from both implementations and
//! computing the symmetric difference.

#[test]
#[ignore = "M6: not yet implemented"]
fn gamv_pair_set_identity_three_directions() {
    todo!("M6: emit pair indices from kriging-rs 3D gamv, compare set to fixture gamv output");
}

#[test]
#[ignore = "M6: not yet implemented"]
fn gamv_pair_counts_per_bin_match_exactly() {
    todo!("M6: bin-by-bin pair count equality");
}
