//! GSLib I/O conversions for 3-D anisotropy and (later) directional variogram
//! parameters. The architectural decision (v3 §D) is that GSLib semantics
//! live exclusively at the API boundary in this module; nothing else in the
//! crate knows about clockwise-from-north azimuth, dip-positive-down, or the
//! ang1→ang2→ang3 composition order.

pub mod gslib_anisotropy;
pub mod gslib_directional;
