//! 3-D anisotropy-aware kd-tree built on [`kiddo::ImmutableKdTree`].
//!
//! The tree is built over coordinates pre-transformed into the
//! [`Anisotropy3D`] ellipsoid's principal frame, so the Euclidean distance
//! in tree-space is the *anisotropic* distance in world-space. No
//! over-fetching, no re-ranking — the kd-tree's natural geometry is exactly
//! the metric we want to query against.
//!
//! ## API contract
//!
//! - Inputs to [`KdTree3D::build`] are world-frame [`Coord3D`]s and an
//!   [`Anisotropy3D`]. The tree stores transformed coords; it does *not*
//!   retain the original slice.
//! - Queries take world-frame [`Coord3D`]s; the wrapper transforms the
//!   query before passing it to kiddo.
//! - Returned [`Neighbour3D::index`] indexes back into the original
//!   `points` slice. The returned `distance` is the anisotropic distance.
//! - The tree is **bound to one anisotropy for its lifetime**. Changing
//!   anisotropy means rebuilding (this is acceptable because anisotropy is
//!   a property of the variogram model — changing it means a new analysis).
//!
//! ## Why `f64` tree coordinates
//!
//! The fork scope mandates `f64` for tree coordinates even when the rest of
//! the crate runs on `Real = f32` storage. With strong anisotropy stretching
//! coordinates by 50–100× along one axis, an `f32` tree loses bits of
//! precision near tile boundaries. Casting up at the tree boundary is
//! cheap; world-frame inputs are cast `f32 → f64` and then multiplied
//! through the f64 deformation matrix, so no precision is wasted relative
//! to the storage floor.
//!
//! ## Bucket size and pathological anisotropy
//!
//! Default kiddo bucket size is 32. The fork scope flags pathological
//! anisotropy ratios (e.g. `anis1 = 0.05`) as a case where larger buckets
//! (64–128) outperform smaller ones because the tree's axis-aligned splits
//! degenerate on stretched spaces. Bucket size is fixed at 32 for v1 (kiddo
//! exposes it as a const generic; switching requires a new type alias).
//! Revisit at M11 with the SGS benchmark.

use std::num::NonZeroUsize;

use kiddo::SquaredEuclidean;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use nalgebra::Vector3;

use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;

/// A neighbour returned by [`KdTree3D`] queries. The `distance` is the
/// **anisotropic** distance (in tree-space coordinates), not the Cartesian
/// world-space distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbour3D {
    /// Index into the original `points` slice passed to [`KdTree3D::build`].
    pub index: usize,
    /// Anisotropic distance from the query point to this neighbour.
    /// Non-negative.
    pub distance: f64,
}

// kiddo type alias keyed on f64 coords, u64 indices, K=3 dims, bucket=32.
// Bucket size is a const generic so we cannot tune it dynamically without
// introducing a second type alias; do that at M11 if benchmarks justify it.
type Tree = ImmutableKdTree<f64, u64, 3, 32>;

/// Anisotropy-aware 3-D kd-tree.
#[derive(Debug)]
pub struct KdTree3D {
    tree: Tree,
    deformation: nalgebra::Matrix3<f64>,
}

impl KdTree3D {
    /// Build a tree from a non-empty slice of points and an anisotropy.
    /// Coordinates are transformed by `anisotropy.deformation_matrix()` and
    /// the resulting principal-frame coordinates are indexed in f64.
    ///
    /// Pass [`Anisotropy3D::identity`] for an isotropic tree (no rotation,
    /// no stretch — the cheapest case but still goes through the matrix
    /// multiply for code-path uniformity).
    pub fn build(points: &[Coord3D], anisotropy: Anisotropy3D) -> Self {
        let deformation = anisotropy.deformation_matrix();
        let entries: Vec<[f64; 3]> = points
            .iter()
            .map(|p| {
                let v = Vector3::new(p.x as f64, p.y as f64, p.z as f64);
                let tp = deformation * v;
                [tp[0], tp[1], tp[2]]
            })
            .collect();
        let tree: Tree = (&*entries).into();
        Self { tree, deformation }
    }

    /// Number of points indexed by the tree.
    #[inline]
    pub fn size(&self) -> usize {
        self.tree.size() as usize
    }

    fn transform_query(&self, query: Coord3D) -> [f64; 3] {
        let v = Vector3::new(query.x as f64, query.y as f64, query.z as f64);
        let tq = self.deformation * v;
        [tq[0], tq[1], tq[2]]
    }

    /// Find the single nearest neighbour to `query` under anisotropic
    /// distance.
    pub fn nearest_one(&self, query: Coord3D) -> Neighbour3D {
        let q = self.transform_query(query);
        let nn = self.tree.nearest_one::<SquaredEuclidean>(&q);
        Neighbour3D {
            index: nn.item as usize,
            distance: nn.distance.sqrt(),
        }
    }

    /// Find up to `max_count` nearest neighbours to `query`, sorted by
    /// ascending anisotropic distance.
    pub fn nearest_n(&self, query: Coord3D, max_count: NonZeroUsize) -> Vec<Neighbour3D> {
        let q = self.transform_query(query);
        self.tree
            .nearest_n::<SquaredEuclidean>(&q, max_count)
            .into_iter()
            .map(|nn| Neighbour3D {
                index: nn.item as usize,
                distance: nn.distance.sqrt(),
            })
            .collect()
    }

    /// Find all neighbours within `radius` (anisotropic distance, inclusive)
    /// of `query`. Results are unsorted; callers should sort if order matters.
    pub fn within(&self, query: Coord3D, radius: f64) -> Vec<Neighbour3D> {
        let q = self.transform_query(query);
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
    use nalgebra::{Matrix3, Vector3};

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

    // ----- isotropy preserved: M2 tests, now passing identity -----

    #[test]
    fn isotropic_size_matches_input_length() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, Anisotropy3D::identity());
        assert_eq!(tree.size(), 8);
    }

    #[test]
    fn isotropic_nearest_one_returns_query_itself_when_query_is_a_point() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, Anisotropy3D::identity());
        let nn = tree.nearest_one(Coord3D::new(1.0, 1.0, 1.0));
        assert_eq!(nn.index, 7);
        assert_eq!(nn.distance, 0.0);
    }

    #[test]
    fn isotropic_nearest_n_returns_sorted_neighbours() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, Anisotropy3D::identity());
        let centre = Coord3D::new(0.5, 0.5, 0.5);
        let nns = tree.nearest_n(centre, NonZeroUsize::new(4).unwrap());
        assert_eq!(nns.len(), 4);
        let expected = (0.75f64).sqrt();
        for nn in &nns {
            assert_relative_eq!(nn.distance, expected, epsilon = 1e-9);
        }
        for w in nns.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn isotropic_within_radius_includes_boundary() {
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, Anisotropy3D::identity());
        let neighbours = tree.within(Coord3D::new(0.0, 0.0, 0.0), 1.0);
        let mut indices: Vec<usize> = neighbours.iter().map(|n| n.index).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 4]);
    }

    // ----- anisotropy-aware behaviour -----

    #[test]
    fn axis_aligned_stretch_changes_nearest_neighbour() {
        // No rotation, but z is stretched by 10x. With no stretch, the
        // closest point to (0, 0, 0.5) is corner 4 at (0,0,1) (distance 0.5).
        // After stretch z -> 10z, (0,0,0.5) maps to (0,0,5) and corner 4
        // (0,0,1) -> (0,0,10), so the distance becomes 5.0. Meanwhile
        // corner 1 at (1,0,0) -> (1,0,0) stays distance sqrt(1 + 0 + 25) = sqrt(26)
        // from (0,0,5), but corner 0 at (0,0,0) -> (0,0,0) is distance 5.0.
        //
        // The nearest point is therefore corner 0 (tied with 4); kiddo
        // returns a deterministic choice. We assert the *distance* is 5.0
        // not 0.5, which proves the stretch was applied.
        let aniso = Anisotropy3D::from_rotation_matrix(
            Matrix3::identity(),
            Vector3::new(1.0, 1.0, 10.0),
        )
        .unwrap();
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, aniso);
        let nn = tree.nearest_one(Coord3D::new(0.0, 0.0, 0.5));
        assert_relative_eq!(nn.distance, 5.0, epsilon = 1e-9);
        // Without stretch this would be 0.5; with z-stretch it's 5.0.
        assert!(nn.index == 0 || nn.index == 4);
    }

    #[test]
    fn anisotropic_distance_matches_direct_computation() {
        // For every (query, neighbour) pair, the distance reported by the
        // tree must equal Anisotropy3D::anisotropic_distance on the world-
        // frame coordinates. This is the foundational invariant of the
        // pre-transformed kd-tree.
        let aniso = Anisotropy3D::from_rotation_matrix(
            Matrix3::identity(),
            Vector3::new(1.0, 2.0, 5.0),
        )
        .unwrap();
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, aniso);
        let query = Coord3D::new(0.3, 0.4, 0.5);
        let nns = tree.nearest_n(query, NonZeroUsize::new(8).unwrap());
        for nn in nns {
            let expected = aniso.anisotropic_distance(query, pts[nn.index]);
            // Tolerance is 1e-6 because Coord3D storage is Real = f32; the
            // tree pre-transform and the direct anisotropic_distance call
            // round f32 inputs differently through the f64 deformation
            // matrix. Same precision floor as the M2 isotropic case.
            assert_relative_eq!(nn.distance, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn rotation_invariance_distances_unchanged_under_isotropic_rotation() {
        // Pure rotation (no stretch) should not change any distance. Build
        // two trees on the same points: one isotropic, one rotated. Query
        // the same world-space points; distances must agree.
        let cos = 0.5_f64.sqrt();
        let sin = 0.5_f64.sqrt();
        let rot_z_45 = Matrix3::new(
            cos, -sin, 0.0,
            sin,  cos, 0.0,
            0.0,  0.0, 1.0,
        );
        let iso = Anisotropy3D::identity();
        let rotated = Anisotropy3D::from_rotation_matrix(
            rot_z_45,
            Vector3::new(1.0, 1.0, 1.0),
        )
        .unwrap();
        let pts = unit_cube_corners();
        let tree_iso = KdTree3D::build(&pts, iso);
        let tree_rot = KdTree3D::build(&pts, rotated);
        let query = Coord3D::new(0.5, 0.5, 0.5);
        let nns_iso = tree_iso.nearest_n(query, NonZeroUsize::new(8).unwrap());
        let nns_rot = tree_rot.nearest_n(query, NonZeroUsize::new(8).unwrap());
        // Distances should be elementwise equal regardless of which corner
        // got which sort-rank.
        let mut d_iso: Vec<f64> = nns_iso.iter().map(|n| n.distance).collect();
        let mut d_rot: Vec<f64> = nns_rot.iter().map(|n| n.distance).collect();
        d_iso.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d_rot.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in d_iso.iter().zip(d_rot.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn within_radius_uses_anisotropic_metric() {
        // Stretch z by 10x. A radius of 2.0 from (0, 0, 0) in anisotropic
        // distance corresponds to z extent of 0.2 -- so corners with z = 1
        // (anisotropic z = 10) are excluded; corners with z = 0 within
        // anisotropic radius 2 are included.
        let aniso = Anisotropy3D::from_rotation_matrix(
            Matrix3::identity(),
            Vector3::new(1.0, 1.0, 10.0),
        )
        .unwrap();
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, aniso);
        let neighbours = tree.within(Coord3D::new(0.0, 0.0, 0.0), 2.0);
        let mut indices: Vec<usize> = neighbours.iter().map(|n| n.index).collect();
        indices.sort();
        // Should be exactly the four z=0 corners: 0, 1, 2, 3 (each at
        // anisotropic distance 0, 1, 1, sqrt(2) <= 2).
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn pathological_anisotropy_still_returns_correct_neighbours() {
        // anis1 = 0.02 (50x stretch on y) is the v3 pathological case.
        // Functionally this should still find correct neighbours, even if
        // the tree's internal performance degrades (perf is checked in
        // benches, not here).
        let aniso = Anisotropy3D::from_rotation_matrix(
            Matrix3::identity(),
            Vector3::new(1.0, 50.0, 1.0),
        )
        .unwrap();
        let pts = unit_cube_corners();
        let tree = KdTree3D::build(&pts, aniso);
        // Query near corner 1 = (1, 0, 0). With 50x y-stretch, points with
        // different y are far. Corner 1 should still be returned.
        let nn = tree.nearest_one(Coord3D::new(1.0, 0.0, 0.0));
        assert_eq!(nn.index, 1);
        assert_eq!(nn.distance, 0.0);

        // Within radius 1.5 (anisotropic) from (0,0,0): only y=0 corners
        // (indices 0, 1, 4, 5) at distances 0, 1, 1, sqrt(2).
        let neighbours = tree.within(Coord3D::new(0.0, 0.0, 0.0), 1.5);
        let mut indices: Vec<usize> = neighbours.iter().map(|n| n.index).collect();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 4, 5]);
    }

    #[test]
    fn distance_matches_gslib_anisotropy_round_trip() {
        // Cross-validate against the GSLib interop: building a tree with an
        // Anisotropy3D constructed from GSLib parameters should produce
        // distances that match Anisotropy3D::anisotropic_distance.
        let gslib = crate::interop::gslib_anisotropy::GslibAnisotropy {
            ang1: 45.0,
            ang2: 30.0,
            ang3: 15.0,
            anis1: 0.5,
            anis2: 0.3,
        };
        let aniso = crate::interop::gslib_anisotropy::from_gslib(gslib).unwrap();
        let pts = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(1.0, 0.0, 0.0),
            Coord3D::new(0.0, 1.0, 0.0),
            Coord3D::new(0.0, 0.0, 1.0),
            Coord3D::new(2.0, 3.0, 4.0),
        ];
        let tree = KdTree3D::build(&pts, aniso);
        let query = Coord3D::new(0.1, 0.2, 0.3);
        let nns = tree.nearest_n(query, NonZeroUsize::new(5).unwrap());
        for nn in nns {
            let expected = aniso.anisotropic_distance(query, pts[nn.index]);
            // Tolerance is 1e-6 because Coord3D storage is Real = f32; the
            // tree pre-transform and the direct anisotropic_distance call
            // round f32 inputs differently through the f64 deformation
            // matrix. Same precision floor as the M2 isotropic case.
            assert_relative_eq!(nn.distance, expected, epsilon = 1e-6);
        }
    }
}
