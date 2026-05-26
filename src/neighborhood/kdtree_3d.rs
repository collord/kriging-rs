//! 3-D kd-tree wrapper built on [`kiddo::ImmutableKdTree`].
//!
//! v0 of this module (M2) exposes a thin wrapper over raw Cartesian
//! coordinates: nearest-N and within-radius queries. **Anisotropy is added at
//! M4** by pre-transforming input points before tree construction; the query
//! interface does not change.
//!
//! ## Why `f64` tree coordinates
//!
//! The fork scope mandates `f64` for tree coordinates even when the rest of the
//! crate runs on `Real = f32` storage. With strong anisotropy stretching
//! coordinates by 50-100x along one axis, an `f32` tree loses bits of precision
//! near tile boundaries. Casting up at the tree boundary is cheap; casting down
//! to `Real` happens only when distances are returned to callers that need them
//! in storage precision (currently none — distances are reported as `f64`).
//!
//! ## Bucket size
//!
//! Default kiddo bucket size is 32. The fork scope flags pathological
//! anisotropy ratios (e.g. `anis1 = 0.05`) as a case where larger buckets
//! (64-128) outperform smaller ones because the tree's axis-aligned splits
//! degenerate on stretched spaces. Bucket size is exposed as a builder
//! parameter; the default is left at kiddo's 32 for now and revisited with
//! benchmarks at M11.

use std::num::NonZeroUsize;

use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;

use crate::coord_3d::Coord3D;

/// Build-time configuration for [`KdTree3D`].
#[derive(Debug, Clone, Copy)]
pub struct KdTree3DConfig {
    /// kiddo bucket size. Larger values reduce tree depth and let the linear
    /// scan at leaves do more of the work; useful when the input is
    /// near-degenerate (e.g. samples clustered along a borehole or strongly
    /// stretched by anisotropy). Default: 32 (matches kiddo's own default).
    pub bucket_size: usize,
}

impl Default for KdTree3DConfig {
    fn default() -> Self {
        Self { bucket_size: 32 }
    }
}

/// A neighbour returned by [`KdTree3D`] queries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbour3D {
    /// Index into the original `points` slice passed to [`KdTree3D::build`].
    pub index: usize,
    /// Straight-line (Euclidean) distance from the query point to the
    /// neighbour. Always non-negative.
    pub distance: f64,
}

// kiddo's const generics drive type aliases keyed on the bucket size. We use
// the default bucket of 32 until benchmarks at M11 say otherwise.
type Tree = ImmutableKdTree<f64, u64, 3, 32>;

/// Immutable 3-D kd-tree over raw Cartesian coordinates.
///
/// Built once from a slice of [`Coord3D`]; queries return indices back into the
/// original slice along with Euclidean distances.
pub struct KdTree3D {
    tree: Tree,
}

impl KdTree3D {
    /// Build a tree from a non-empty slice of points. Coordinates are cast to
    /// `f64` at construction; the original slice is not retained — callers
    /// keep their own copy and look up neighbour data by the indices returned.
    pub fn build(points: &[Coord3D]) -> Self {
        let entries: Vec<[f64; 3]> = points
            .iter()
            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect();
        let tree: Tree = (&*entries).into();
        Self { tree }
    }

    /// Number of points indexed by the tree.
    #[inline]
    pub fn size(&self) -> usize {
        self.tree.size() as usize
    }

    /// Find the single nearest neighbour to `query`.
    pub fn nearest_one(&self, query: Coord3D) -> Neighbour3D {
        let q = [query.x as f64, query.y as f64, query.z as f64];
        let nn = self.tree.nearest_one::<SquaredEuclidean>(&q);
        Neighbour3D {
            index: nn.item as usize,
            distance: nn.distance.sqrt(),
        }
    }

    /// Find up to `max_count` nearest neighbours to `query`, sorted by
    /// ascending distance. If `max_count` is greater than the tree size, fewer
    /// results are returned.
    pub fn nearest_n(&self, query: Coord3D, max_count: NonZeroUsize) -> Vec<Neighbour3D> {
        let q = [query.x as f64, query.y as f64, query.z as f64];
        self.tree
            .nearest_n::<SquaredEuclidean>(&q, max_count)
            .into_iter()
            .map(|nn| Neighbour3D {
                index: nn.item as usize,
                distance: nn.distance.sqrt(),
            })
            .collect()
    }

    /// Find all neighbours within `radius` (inclusive) of `query`. Results are
    /// unsorted; callers should sort if order matters.
    pub fn within(&self, query: Coord3D, radius: f64) -> Vec<Neighbour3D> {
        let q = [query.x as f64, query.y as f64, query.z as f64];
        let radius_sq = radius * radius;
        self.tree
            .within::<SquaredEuclidean>(&q, radius_sq)
            .into_iter()
            .map(|nn| Neighbour3D {
                index: nn.item as usize,
                distance: nn.distance.sqrt(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn unit_cube_corners() -> Vec<Coord3D> {
        // 8 corners of a unit cube, indexed by binary (z y x):
        //   0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
        //   4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
        vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(1.0, 0.0, 0.0),
            Coord3D::new(0.0, 1.0, 0.0),
            Coord3D::new(1.0, 1.0, 0.0),
            Coord3D::new(0.0, 0.0, 1.0),
            Coord3D::new(1.0, 0.0, 1.0),
            Coord3D::new(0.0, 1.0, 1.0),
            Coord3D::new(1.0, 1.0, 1.0),
        ]
    }

    #[test]
    fn size_matches_input_length() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        assert_eq!(tree.size(), 8);
    }

    #[test]
    fn nearest_one_returns_query_itself_when_query_is_a_point() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        let nn = tree.nearest_one(Coord3D::new(1.0, 1.0, 1.0));
        assert_eq!(nn.index, 7);
        assert_eq!(nn.distance, 0.0);
    }

    #[test]
    fn nearest_one_picks_closest_corner() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        // Query is near corner 5 = (1, 0, 1).
        let nn = tree.nearest_one(Coord3D::new(0.9, 0.1, 0.9));
        assert_eq!(nn.index, 5);
        let expected = ((0.1f64).powi(2) * 3.0).sqrt();
        // Storage is f32 (Real = f32); tree coords are cast to f64 but no
        // precision is recovered. Tolerance is f32-precision-bounded.
        assert_relative_eq!(nn.distance, expected, epsilon = 1e-6);
    }

    #[test]
    fn nearest_n_returns_sorted_neighbours() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        // Query at cube centre: every corner is equidistant (sqrt(0.75)).
        let centre = Coord3D::new(0.5, 0.5, 0.5);
        let nns = tree.nearest_n(centre, NonZeroUsize::new(4).unwrap());
        assert_eq!(nns.len(), 4);
        let expected = (0.75f64).sqrt();
        for nn in &nns {
            assert_relative_eq!(nn.distance, expected, epsilon = 1e-9);
        }
        // Distances should be monotonically non-decreasing (sorted).
        for w in nns.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn nearest_n_capped_by_tree_size() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        let nns = tree.nearest_n(
            Coord3D::new(0.0, 0.0, 0.0),
            NonZeroUsize::new(100).unwrap(),
        );
        assert_eq!(nns.len(), 8);
    }

    #[test]
    fn within_radius_includes_boundary() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        // Query at (0, 0, 0); radius 1.0 should include corners at distance 1
        // (axis-aligned neighbours: idx 1, 2, 4) plus the query point itself.
        let neighbours = tree.within(Coord3D::new(0.0, 0.0, 0.0), 1.0);
        let mut indices: Vec<usize> = neighbours.iter().map(|n| n.index).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 4]);
    }

    #[test]
    fn within_radius_zero_returns_only_exact_matches() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts);
        let neighbours = tree.within(Coord3D::new(0.0, 0.0, 0.0), 0.0);
        assert_eq!(neighbours.len(), 1);
        assert_eq!(neighbours[0].index, 0);
    }

    #[test]
    fn distance_is_euclidean_not_squared() {
        // The tree internally uses SquaredEuclidean; the wrapper must sqrt
        // before exposing the distance. Verify with a 3-4-12 -> 13 triangle:
        // query from origin, the only non-origin point should be at distance 13.
        let pts = vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(3.0, 4.0, 12.0)];
        let tree = KdTree3D::build(&pts);
        let nns = tree.nearest_n(
            Coord3D::new(0.0, 0.0, 0.0),
            NonZeroUsize::new(2).unwrap(),
        );
        assert_eq!(nns.len(), 2);
        assert_eq!(nns[0].index, 0);
        assert_relative_eq!(nns[0].distance, 0.0, epsilon = 1e-9);
        assert_eq!(nns[1].index, 1);
        assert_relative_eq!(nns[1].distance, 13.0, epsilon = 1e-6);
    }
}
