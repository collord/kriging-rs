//! Performance characterization of the anisotropy-aware 3-D kd-tree, with
//! emphasis on pathological anisotropy ratios.
//!
//! Per v3 §E, pre-transforming coordinates into the anisotropy ellipsoid's
//! principal frame makes the tree's Euclidean distance equal the anisotropic
//! distance — correct, no over-fetching. But the geometric stretch the
//! pre-transform induces can hurt tree *performance*: with `anis1 = 0.05` or
//! smaller (legitimate for layered sediments, varved deposits, fault-zone
//! permeability fields), the transformed coordinate space becomes a long
//! thin slab and kd-tree axis-aligned splits degenerate.
//!
//! This bench measures build cost and nearest-n query cost across four
//! anisotropy regimes, so the M11 SGS benchmark decisions have a baseline.
//! Run with: `cargo bench --bench bench_3d_kdtree`.
//!
//! Baseline numbers (M4, macOS x86_64, 1000 random points, 100 queries
//! @ n=20, criterion `--quick` mode):
//!
//! | Case                       | Build      | nearest_n |
//! |----------------------------|------------|-----------|
//! | isotropic                  | ~28 µs    | ~199 µs   |
//! | mild_2x                    | ~30 µs    | ~200 µs   |
//! | pathological_flat_50x      | ~29 µs    | ~341 µs   |
//! | pathological_3d_50x        | ~30 µs    | ~307 µs   |
//!
//! Build is essentially unaffected by anisotropy ratio; query cost rises
//! ~1.7x at 50x stretch. This is meaningful (it doesn't justify a
//! fundamental architecture change) but bounded.

use std::num::NonZeroUsize;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kriging_rs::neighborhood::kdtree_3d::KdTree3D;
use kriging_rs::{Anisotropy3D, Coord3D};
use nalgebra::{Matrix3, Vector3};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N_POINTS: usize = 1000;
const SEED: u64 = 0x3DC0FFEE;

fn generate_points(n: usize, seed: u64) -> Vec<Coord3D> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Coord3D::new(
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-50.0..50.0),
            )
        })
        .collect()
}

fn make_anisotropy(stretch: [f64; 3]) -> Anisotropy3D {
    Anisotropy3D::from_rotation_matrix(
        Matrix3::identity(),
        Vector3::new(stretch[0], stretch[1], stretch[2]),
    )
    .expect("valid anisotropy")
}

fn bench_build(c: &mut Criterion) {
    let points = generate_points(N_POINTS, SEED);
    let cases = [
        ("isotropic", [1.0, 1.0, 1.0]),
        ("mild_2x", [1.0, 2.0, 1.0]),
        ("pathological_flat_50x", [1.0, 50.0, 1.0]),
        ("pathological_3d_50x", [1.0, 50.0, 50.0]),
    ];
    let mut group = c.benchmark_group("kdtree_3d_build");
    for (name, stretch) in cases {
        let aniso = make_anisotropy(stretch);
        group.bench_with_input(BenchmarkId::from_parameter(name), &aniso, |b, aniso| {
            b.iter(|| {
                let tree = KdTree3D::build(&points, *aniso);
                std::hint::black_box(tree);
            });
        });
    }
    group.finish();
}

fn bench_nearest_n(c: &mut Criterion) {
    let points = generate_points(N_POINTS, SEED);
    // Pre-generate query points so query cost dominates RNG cost.
    let queries: Vec<Coord3D> = generate_points(100, SEED ^ 0x1234).into_iter().collect();
    let cases = [
        ("isotropic", [1.0, 1.0, 1.0]),
        ("mild_2x", [1.0, 2.0, 1.0]),
        ("pathological_flat_50x", [1.0, 50.0, 1.0]),
        ("pathological_3d_50x", [1.0, 50.0, 50.0]),
    ];
    let n = NonZeroUsize::new(20).unwrap();
    let mut group = c.benchmark_group("kdtree_3d_nearest_n_20");
    for (name, stretch) in cases {
        let aniso = make_anisotropy(stretch);
        let tree = KdTree3D::build(&points, aniso);
        group.bench_with_input(BenchmarkId::from_parameter(name), &tree, |b, tree| {
            b.iter(|| {
                let mut total = 0.0;
                for q in &queries {
                    let nns = tree.nearest_n(*q, n);
                    total += nns[0].distance;
                }
                std::hint::black_box(total);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_build, bench_nearest_n);
criterion_main!(benches);
