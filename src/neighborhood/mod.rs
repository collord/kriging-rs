//! Spatial-index utilities for kriging neighborhood search.
//!
//! Upstream's 2-D path uses naive O(n) neighborhood search; this module is the
//! entry point for spatial indexing in the 3-D fork. v1 adds [`kdtree_3d`] for
//! 3-D point sets. The anisotropy-pre-transformed variant lands at M4; v0
//! exposes only the raw Cartesian tree.

pub mod kdtree_3d;
