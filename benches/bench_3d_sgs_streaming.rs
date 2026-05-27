//! Wall-clock benchmark for 3-D streaming SGS at varying grid sizes.
//! Answers the feasibility-at-N question: how big can a grid get before
//! a single realization blows past interactive latency.
//!
//! Run with: `cargo bench --bench bench_3d_sgs_streaming`.
//!
//! For browser/WASM measurements: native is an upper bound; WASM is
//! typically 2-4x slower than native release for kriging-heavy code
//! (matrix work and trigonometry dominate). To measure WASM directly,
//! `wasm-pack build --target nodejs --release` plus a node harness;
//! deferred to v2 unless the native numbers indicate browser is
//! out of reach.
//!
//! Baseline numbers captured below the benchmark code so future drift
//! is visible.

use std::path::PathBuf;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kriging_rs::{
    Anisotropy3D, Coord3D, Grid3D, PlanarDataset3D, Real, SgsModel3D, VariogramModel,
    VariogramType, gaussian_simulation_3d_stream,
};

fn samples_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/parity_3d/fixtures/skgstat_3d/samples.csv")
}

fn load_dataset() -> PlanarDataset3D {
    let csv = std::fs::read_to_string(samples_path()).expect("samples.csv");
    let mut coords = Vec::new();
    let mut values = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().unwrap())
            .collect();
        coords.push(Coord3D::new(
            cols[0] as Real,
            cols[1] as Real,
            cols[2] as Real,
        ));
        values.push(cols[3] as Real);
    }
    PlanarDataset3D::new(coords, values).unwrap()
}

fn variogram() -> VariogramModel {
    VariogramModel::new(0.05, 1.0, 30.0, VariogramType::Exponential).unwrap()
}

fn build_grid(n: usize) -> Grid3D {
    // n^3 grid covering the same 100x100x50 box as the M5 dataset.
    Grid3D::new(
        n,
        n,
        (n as f64 / 2.0).round() as usize, // half-depth in z to match the dataset shape
        Coord3D::new(0.0, 0.0, 0.0),
        Coord3D::new(
            100.0 / (n as Real - 1.0).max(1.0),
            100.0 / (n as Real - 1.0).max(1.0),
            50.0 / ((n as Real / 2.0).max(1.0)),
        ),
    )
    .unwrap()
}

fn bench_sgs_one_realization(c: &mut Criterion) {
    let dataset = load_dataset();
    let model =
        SgsModel3D::new(dataset, Anisotropy3D::identity(), variogram()).unwrap();

    let sizes = [10, 20, 30, 40, 50];
    let mut group = c.benchmark_group("sgs_3d_one_realization");
    group.sample_size(10); // SGS is slow; criterion's default 100 samples is overkill.

    for n in sizes {
        let grid = build_grid(n);
        let n_cells = grid.n_cells();
        group.throughput(criterion::Throughput::Elements(n_cells as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &grid, |b, grid| {
            b.iter(|| {
                let mut realization_count = 0;
                gaussian_simulation_3d_stream(&model, grid, 1234, 1, |_, gv| {
                    std::hint::black_box(gv);
                    realization_count += 1;
                    Ok(())
                })
                .unwrap();
                std::hint::black_box(realization_count);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sgs_one_realization);
criterion_main!(benches);

// Baseline numbers (M11, macOS x86_64 release, 150-sample dataset,
// max_neighbors = 24, criterion --quick mode):
//
// | Grid (n_x * n_y * n_z)   | n_cells | Wall time / realization | Throughput  |
// |--------------------------|---------|-------------------------|-------------|
// | 10 x 10 x 5              |     500 |    ~5.7 ms              | ~87 K /s    |
// | 20 x 20 x 10             |   4 000 |   ~60 ms                | ~67 K /s    |
// | 30 x 30 x 15             |  13 500 |  ~168 ms                | ~80 K /s    |
// | 40 x 40 x 20             |  32 000 |  ~404 ms                | ~79 K /s    |
// | 50 x 50 x 25             |  62 500 |  ~812 ms                | ~77 K /s    |
//
// Cost is **linear in n_cells** at ~80 K cells/s native release on
// x86_64 macOS. The kd-tree's log-n neighbor lookup keeps per-cell
// cost bounded.
//
// Browser projections (native * 3x for WASM, no measurements yet):
//   30^3 (13.5k) x 50 realizations  ~ 25 s  -- interactive-ish
//   50^3 (62.5k) x 50 realizations  ~ 2 min -- "user is waiting"
//   100^3 (1M cells) x 50 reals     ~ 50 min -- batch only
//
// v3 §"Browser memory budget" claimed an interactive ceiling of
// 100^3 x 50 realizations. The wall-clock cost says that ceiling
// was off by ~2 orders of magnitude: 50^3 x 50 reals is already
// ~2 min native, ~6+ min in browser. v1 SGS-in-browser is
// feasible for grids in the 20^3..40^3 range with O(10-50)
// realizations; bigger workloads need server-side or batch.
