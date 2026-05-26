//! SGS statistical-correctness gates. Implemented at M11.
//!
//! Five gates (no point-by-point parity expected):
//! 1. Same-target determinism: same seed -> bit-identical realization on the
//!    same compiled binary / platform.
//! 2. Cross-platform statistical equivalence: same seed across linux/wasm/darwin
//!    produces realizations whose summary statistics agree within sampling tolerance.
//! 3. Mean of N realizations -> OK prediction within sigma_OK / sqrt(N).
//! 4. Per-location variance -> kriging variance within sampling tolerance.
//! 5. Experimental variogram of realizations averaged -> input variogram model.
//!
//! Plus a solver-robustness gate (no silent NaN failures).

#[test]
#[ignore = "M11: not yet implemented"]
fn same_target_determinism_bit_identical_realization() {
    todo!("M11: same seed, same target -> bit-identical realization");
}

#[test]
#[ignore = "M11: not yet implemented"]
fn realization_mean_converges_to_ok_prediction() {
    todo!("M11: |mean(realizations) - OK| < sigma_OK / sqrt(N)");
}

#[test]
#[ignore = "M11: not yet implemented"]
fn realization_variance_matches_kriging_variance() {
    todo!("M11: per-location var across realizations vs kriging variance");
}

#[test]
#[ignore = "M11: not yet implemented"]
fn realization_variogram_matches_input_model() {
    todo!("M11: experimental variogram of realizations vs input model");
}

#[test]
#[ignore = "M11: not yet implemented"]
fn sgs_produces_no_nan_on_challenging_dataset() {
    todo!("M11: collinear samples + strong anisotropy -> no silent NaN");
}
