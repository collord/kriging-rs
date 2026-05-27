//! 3D parity test harness. Scaffolded at M1; per-area modules filled in at
//! their respective milestones. Run with `cargo test --test parity_3d -- --ignored`
//! once tests are implemented.

mod parity_3d {
    pub mod gslib_roundtrip;
    pub mod skgstat_reference;
    pub mod gamv_reference;
    pub mod textbook_kriging_reference;
    pub mod sgs_statistical_gates;
}
