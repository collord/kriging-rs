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
use kiddo::float::kdtree::KdTree;
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

// kiddo mutable kd-tree type alias.
type MutableTree = KdTree<f64, u64, 3, 32, u32>;

/// Anisotropy-aware **mutable** 3-D kd-tree.
///
/// Used by SGS to maintain a growing conditioning set during a single
/// realization. The pre-transformation logic is identical to the
/// immutable [`KdTree3D`]; the difference is `add` is supported and
/// `build_with_capacity` allows pre-allocating enough space for the
/// final size to avoid rehashing.
///
/// API mirrors [`KdTree3D`] but each query operates on the current
/// (live) point set rather than a snapshot.

/// Deterministic, per-id tiny perturbation on each axis. Magnitude
/// ~1e-12 (relative; scaled by coordinate magnitude at the callsite),
/// using a splitmix-style hash of `id` so different ids get different
/// offsets and the same id always gets the same offset.
fn jitter_for(id: u64) -> (f64, f64, f64) {
    let mut s = id.wrapping_add(0x9E3779B97F4A7C15);
    let h = |z: &mut u64| -> f64 {
        *z = z.wrapping_add(0x9E3779B97F4A7C15);
        let mut x = *z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B5);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        // Map to (-0.5, 0.5) range, then scale to 1e-12.
        let u = (x as f64) / (u64::MAX as f64) - 0.5;
        u * 1e-12
    };
    let jx = h(&mut s);
    let jy = h(&mut s);
    let jz = h(&mut s);
    (jx, jy, jz)
}

#[derive(Debug)]
pub struct MutableKdTree3D {
    tree: MutableTree,
    deformation: nalgebra::Matrix3<f64>,
    /// Monotonically increasing as `add` is called. kiddo's mutable
    /// tree internally tracks `size()`; this is just a local cache for
    /// callers who want it without a method call.
    next_id: u64,
}

impl MutableKdTree3D {
    /// Construct an empty mutable kd-tree with the given anisotropy.
    pub fn new(anisotropy: Anisotropy3D) -> Self {
        Self {
            tree: MutableTree::with_capacity(0),
            deformation: anisotropy.deformation_matrix(),
            next_id: 0,
        }
    }

    /// Construct an empty tree pre-sized for `capacity` points. Avoids
    /// reallocation as a known number of points is added; useful for
    /// SGS where we know `n_samples + n_grid_cells` upfront.
    pub fn with_capacity(anisotropy: Anisotropy3D, capacity: usize) -> Self {
        Self {
            tree: MutableTree::with_capacity(capacity),
            deformation: anisotropy.deformation_matrix(),
            next_id: 0,
        }
    }

    /// Add an initial batch of points. The returned indices identify
    /// each point for subsequent queries; the first point gets index 0,
    /// the second gets 1, etc.
    pub fn add_batch(&mut self, points: &[Coord3D]) {
        for p in points {
            self.add(*p);
        }
    }

    /// Add a single point and return its assigned index (0-based, in
    /// insertion order).
    ///
    /// Applies a tiny deterministic per-insertion jitter (~1e-12 of
    /// the larger of the coordinate magnitude or 1.0) before inserting
    /// into kiddo's mutable kd-tree. Background: kiddo's mutable tree
    /// panics when more than `bucket_size` (currently 32) items share
    /// the same value on any axis. SGS callers routinely feed
    /// axis-aligned grid cells (where dozens to hundreds of cells
    /// share an x or y coordinate), and there is no realistic dataset
    /// where the bucket size would be enough -- a 100x100 grid alone
    /// has 100 colinear cells per axis. The jitter is far below
    /// kriging-relevant precision (variogram distances start at
    /// metres-to-kilometres, the perturbation is at the ulp scale) so
    /// it doesn't affect nearest-neighbour ordering or kriging
    /// solutions, but it does keep kiddo's internal bucket invariant
    /// satisfied.
    pub fn add(&mut self, point: Coord3D) -> usize {
        let v = Vector3::new(point.x as f64, point.y as f64, point.z as f64);
        let tp = self.deformation * v;
        let id = self.next_id;
        let (jx, jy, jz) = jitter_for(id);
        // Scale jitter by max(|coord|, 1.0) so the perturbation stays
        // ulp-sized even for large-magnitude coordinates -- a pure
        // ~1e-12 absolute jitter would round away at distances on the
        // order of 1e9.
        let sx = tp[0].abs().max(1.0);
        let sy = tp[1].abs().max(1.0);
        let sz = tp[2].abs().max(1.0);
        self.tree
            .add(&[tp[0] + jx * sx, tp[1] + jy * sy, tp[2] + jz * sz], id);
        self.next_id += 1;
        id as usize
    }

    /// Number of points currently in the tree.
    #[inline]
    pub fn size(&self) -> usize {
        self.tree.size() as usize
    }

    fn transform_query(&self, query: Coord3D) -> [f64; 3] {
        let v = Vector3::new(query.x as f64, query.y as f64, query.z as f64);
        let tq = self.deformation * v;
        [tq[0], tq[1], tq[2]]
    }

    /// Find up to `max_count` nearest neighbours under anisotropic
    /// distance. Returns an empty Vec if the tree is empty.
    pub fn nearest_n(&self, query: Coord3D, max_count: usize) -> Vec<Neighbour3D> {
        if self.tree.size() == 0 || max_count == 0 {
            return Vec::new();
        }
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
}

#[cfg(test)]
mod mutable_tests {
    use super::*;
    use crate::Real;
    use approx::assert_relative_eq;

    #[test]
    fn empty_tree_returns_empty_nearest_n() {
        let tree = MutableKdTree3D::new(Anisotropy3D::identity());
        assert_eq!(tree.size(), 0);
        let q = tree.nearest_n(Coord3D::new(0.0, 0.0, 0.0), 5);
        assert!(q.is_empty());
    }

    #[test]
    fn dense_colinear_points_do_not_panic() {
        // Regression: kiddo's mutable kd-tree panics when more than
        // `bucket_size` (32) items share a position on one axis.
        // SGS routinely feeds axis-aligned grid cells where dozens or
        // hundreds of cells share an x or y coordinate, so the tree
        // must add jitter before insertion to keep the bucket
        // invariant satisfied. This test would panic without the
        // jitter path in `MutableKdTree3D::add`.
        let mut tree = MutableKdTree3D::with_capacity(Anisotropy3D::identity(), 1000);
        // 200 points all sharing x=5.0 and y=10.0, varying z. 200 > 32.
        for i in 0..200 {
            tree.add(Coord3D::new(5.0, 10.0, i as Real));
        }
        assert_eq!(tree.size(), 200);
        // The 5 nearest to (5, 10, 100) should be at z=98..102 in some
        // order. Jitter is ulp-scale so ordering by z is preserved.
        let nns = tree.nearest_n(Coord3D::new(5.0, 10.0, 100.0), 5);
        assert_eq!(nns.len(), 5);
        let mut z_values: Vec<f64> = nns
            .iter()
            .map(|n| n.distance) // distance == |z - 100| in this layout
            .collect();
        z_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Distances should be 0, 1, 1, 2, 2 (the centred point and
        // the two pairs flanking it).
        assert_relative_eq!(z_values[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(z_values[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(z_values[2], 1.0, epsilon = 1e-6);
        assert_relative_eq!(z_values[3], 2.0, epsilon = 1e-6);
        assert_relative_eq!(z_values[4], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn add_then_query_returns_added_point() {
        let mut tree = MutableKdTree3D::new(Anisotropy3D::identity());
        let id = tree.add(Coord3D::new(3.0, 4.0, 12.0));
        assert_eq!(id, 0);
        assert_eq!(tree.size(), 1);
        let q = tree.nearest_n(Coord3D::new(0.0, 0.0, 0.0), 1);
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].index, 0);
        assert_relative_eq!(q[0].distance, 13.0, epsilon = 1e-6);
    }

    #[test]
    fn incremental_add_reflects_in_subsequent_queries() {
        let mut tree = MutableKdTree3D::new(Anisotropy3D::identity());
        tree.add(Coord3D::new(0.0, 0.0, 0.0));
        // Before adding the closer point, nearest to (5,0,0) is (0,0,0).
        let q1 = tree.nearest_n(Coord3D::new(5.0, 0.0, 0.0), 1);
        assert_eq!(q1[0].index, 0);
        // After adding (6,0,0), it should be closer.
        tree.add(Coord3D::new(6.0, 0.0, 0.0));
        let q2 = tree.nearest_n(Coord3D::new(5.0, 0.0, 0.0), 1);
        assert_eq!(q2[0].index, 1);
    }

    #[test]
    fn anisotropy_propagates_into_mutable_tree() {
        let aniso = Anisotropy3D::from_rotation_matrix(
            nalgebra::Matrix3::identity(),
            nalgebra::Vector3::new(1.0, 1.0, 10.0),
        )
        .unwrap();
        let mut tree = MutableKdTree3D::new(aniso);
        tree.add(Coord3D::new(0.0, 0.0, 0.0));
        tree.add(Coord3D::new(1.0, 0.0, 0.0));
        tree.add(Coord3D::new(0.0, 0.0, 1.0)); // 10x stretched in z
        // Query at origin: index 0 is closest at d=0. Among the other two,
        // (1,0,0) at d=1 is closer than (0,0,1) at anisotropic d=10.
        let q = tree.nearest_n(Coord3D::new(0.0, 0.0, 0.0), 3);
        assert_eq!(q[0].index, 0);
        assert_eq!(q[1].index, 1);
        assert_eq!(q[2].index, 2);
        assert_relative_eq!(q[1].distance, 1.0, epsilon = 1e-6);
        assert_relative_eq!(q[2].distance, 10.0, epsilon = 1e-6);
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
