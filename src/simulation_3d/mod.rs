//! 3-D sequential Gaussian simulation (M11).
//!
//! See [`sgs_3d::gaussian_simulation_3d_stream`] for the public entry
//! point. The module also exposes the supporting types:
//! [`grid::Grid3D`] for the simulation grid,
//! [`nst::NormalScoreTransform`] for the pre/post-processing transform,
//! and [`sgs_3d::SgsConfig`] for tuning the simulation.

pub mod grid;
pub mod nst;
pub mod sgs_3d;

pub use grid::Grid3D;
pub use nst::NormalScoreTransform;
pub use sgs_3d::{
    SgsConfig, SgsError, SgsModel3D, SgsOutputSpace, gaussian_simulation_3d_stream,
    gaussian_simulation_3d_stream_with,
};
