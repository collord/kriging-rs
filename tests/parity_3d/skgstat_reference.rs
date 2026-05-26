//! scikit-gstat numerical-tier parity for the 3-D omnidirectional
//! experimental variogram (M5).
//!
//! Loads:
//! - `fixtures/skgstat_3d/samples.csv` — 150 deterministic 3-D samples.
//! - `fixtures/skgstat_3d/bin_edges.csv` — 10 equal-width bin upper edges.
//! - `fixtures/skgstat_3d/omnidirectional.csv` — semivariance and pair
//!   count per bin from scikit-gstat 1.0.23 with the Matheron estimator.
//!
//! Asserts that `compute_empirical_variogram_3d` produces the same pair
//! counts (exact) and semivariances (within numerical tolerance) when
//! given the same dataset and the same equal-width bin partition.
//!
//! See `fixtures/skgstat_3d/README.md` for fixture provenance and the
//! reason we use scikit-gstat instead of geostatspy as v3 originally
//! planned.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use kriging_rs::variogram::{
    EmpiricalEstimator, PositiveReal, VariogramConfig, compute_empirical_variogram_3d,
};
use kriging_rs::{Anisotropy3D, Coord3D, PlanarDataset3D, Real};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/parity_3d/fixtures/skgstat_3d")
}

fn load_samples() -> PlanarDataset3D {
    let path = fixture_dir().join("samples.csv");
    let csv = std::fs::read_to_string(&path).expect("samples.csv should exist");
    let mut coords = Vec::new();
    let mut values = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue; // header
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("numeric column"))
            .collect();
        assert_eq!(cols.len(), 4, "row {i}: expected x,y,z,v");
        coords.push(Coord3D::new(
            cols[0] as Real,
            cols[1] as Real,
            cols[2] as Real,
        ));
        values.push(cols[3] as Real);
    }
    PlanarDataset3D::new(coords, values).expect("valid dataset")
}

struct BinExpectation {
    semivariance: f64,
    n_pairs: usize,
}

fn load_expectations() -> (Vec<f64>, Vec<BinExpectation>) {
    let edges_csv = std::fs::read_to_string(fixture_dir().join("bin_edges.csv"))
        .expect("bin_edges.csv should exist");
    let mut upper_edges = Vec::new();
    for (i, line) in edges_csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("numeric column"))
            .collect();
        // cols: bin_index, lower_edge, upper_edge
        upper_edges.push(cols[2]);
    }

    let omni_csv = std::fs::read_to_string(fixture_dir().join("omnidirectional.csv"))
        .expect("omnidirectional.csv should exist");
    let mut expectations = Vec::new();
    for (i, line) in omni_csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        // cols: bin_index, semivariance, n_pairs
        expectations.push(BinExpectation {
            semivariance: cols[1].parse().expect("semivariance"),
            n_pairs: cols[2].parse().expect("n_pairs"),
        });
    }
    assert_eq!(upper_edges.len(), expectations.len());
    (upper_edges, expectations)
}

#[test]
fn omnidirectional_variogram_matches_skgstat_reference() {
    let dataset = load_samples();
    let (upper_edges, expectations) = load_expectations();

    // skgstat uses equal-width bins. The last upper edge is the effective
    // max distance. Feed our function the same partition.
    let max_distance = *upper_edges.last().unwrap();
    let n_bins = NonZeroUsize::new(upper_edges.len()).unwrap();

    let config = VariogramConfig {
        max_distance: Some(PositiveReal::try_new(max_distance as Real).unwrap()),
        n_bins,
        estimator: EmpiricalEstimator::Classical, // Matheron
    };

    let ev = compute_empirical_variogram_3d(
        &dataset,
        &Anisotropy3D::identity(),
        &config,
    )
    .expect("omnidirectional variogram should compute");

    // skgstat returns one entry per bin; my code skips empty bins. Filter
    // to the non-empty bins for a 1:1 comparison.
    let non_empty_expectations: Vec<&BinExpectation> =
        expectations.iter().filter(|e| e.n_pairs > 0).collect();

    assert_eq!(
        ev.n_pairs.len(),
        non_empty_expectations.len(),
        "kriging-rs bin count vs skgstat non-empty bin count",
    );

    // Pair-count parity: exact (set identity).
    for (i, (ours, theirs)) in ev.n_pairs.iter().zip(&non_empty_expectations).enumerate() {
        assert_eq!(
            *ours, theirs.n_pairs,
            "bin {i}: pair count mismatch (ours={ours}, skgstat={})",
            theirs.n_pairs
        );
    }

    // Semivariance parity: numerical-equivalence tier per v3.
    #[cfg(not(feature = "f64"))]
    let tolerance: f64 = 1e-4;
    #[cfg(feature = "f64")]
    let tolerance: f64 = 1e-9;

    for (i, (ours, theirs)) in ev
        .semivariances
        .iter()
        .zip(&non_empty_expectations)
        .enumerate()
    {
        let ours = *ours as f64;
        let theirs = theirs.semivariance;
        let rel_err = if theirs.abs() > 1e-12 {
            (ours - theirs).abs() / theirs.abs()
        } else {
            (ours - theirs).abs()
        };
        assert!(
            rel_err < tolerance,
            "bin {i}: semivariance mismatch (ours={ours}, skgstat={theirs}, rel_err={rel_err:e}, tol={tolerance:e})",
        );
    }
}
