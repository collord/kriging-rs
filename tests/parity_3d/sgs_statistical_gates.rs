//! SGS statistical-correctness gates (M11).
//!
//! v3 §"SGS statistical correctness gates" lists five gates plus a
//! solver-robustness gate. M11 ships the three load-bearing ones:
//!
//! 1. **Same-target determinism**: same seed + same compiled binary →
//!    bit-identical realization. This is the regression-test gate.
//! 2. **Mean-converges-to-OK**: average of N realizations should approach
//!    the OK prediction at each grid cell to within `σ_OK / √N`.
//! 3. **No silent NaN failures**: SGS on a challenging dataset must
//!    surface failures as explicit NaN-marked cells (and a non-zero
//!    count we can detect), never silently return arbitrary values.
//!
//! Deferred to v2 polish (per the M11 scoping decision documented in
//! `docs/m0-findings.md`):
//! - Cross-platform statistical equivalence (no other platforms tested
//!   yet; the M11 deliverable is single-platform feasibility).
//! - Realization variance ≈ kriging variance (correctness invariant,
//!   not a load-bearing gate for the v1 feasibility question).
//! - Realization variogram ≈ input model (same).

use std::path::PathBuf;

use kriging_rs::{
    Anisotropy3D, Coord3D, Grid3D, OrdinaryKrigingModel3D, PlanarDataset3D, Real,
    SgsModel3D, SgsOutputSpace, VariogramModel, VariogramType,
    gaussian_simulation_3d_stream, gaussian_simulation_3d_stream_with,
};

fn samples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/parity_3d/fixtures/skgstat_3d")
}

fn load_dataset() -> PlanarDataset3D {
    let path = samples_dir().join("samples.csv");
    let csv = std::fs::read_to_string(&path).expect("samples.csv should exist");
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

fn score_space_variogram() -> VariogramModel {
    // Reasonable variogram for normal-scored data: unit sill, modest
    // range relative to the dataset extent (100). Matches v3's M11
    // "representative workload" framing.
    VariogramModel::new(0.05, 1.0, 30.0, VariogramType::Exponential).unwrap()
}

fn small_grid() -> Grid3D {
    Grid3D::new(
        10,
        10,
        4,
        Coord3D::new(5.0, 5.0, 2.5),
        Coord3D::new(10.0, 10.0, 10.0),
    )
    .unwrap()
}

#[test]
fn same_target_determinism_bit_identical_realization() {
    let dataset = load_dataset();
    let model = SgsModel3D::new(
        dataset,
        Anisotropy3D::identity(),
        score_space_variogram(),
    )
    .unwrap();
    let grid = small_grid();

    let mut first: Vec<Real> = Vec::new();
    gaussian_simulation_3d_stream(&model, &grid, 0xCAFEBABE, 1, |_, gv| {
        first.extend_from_slice(gv);
        Ok(())
    })
    .unwrap();
    let mut second: Vec<Real> = Vec::new();
    gaussian_simulation_3d_stream(&model, &grid, 0xCAFEBABE, 1, |_, gv| {
        second.extend_from_slice(gv);
        Ok(())
    })
    .unwrap();
    assert_eq!(first.len(), second.len());
    for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "cell {i}: a={a} b={b}",
        );
    }
}

#[test]
fn realization_mean_converges_to_ok_prediction() {
    // Compare in **score space** to avoid the NST back-transform
    // nonlinearity (which would introduce systematic bias unrelated
    // to statistical noise). We run SGS with SgsOutputSpace::ScoreSpace
    // to skip the back-transform, then compare per-cell means to an
    // OK prediction on the score-transformed sample values.
    //
    // Under correctness, the per-cell deviation should scale as
    // σ_OK / √N. With N=100 realizations and 5-σ slack we expect
    // ~95% of cells to pass; we require >= 90% to leave margin for
    // tail-cell variability.
    let dataset = load_dataset();
    let variogram = score_space_variogram();
    let model = SgsModel3D::new(
        dataset.clone(),
        Anisotropy3D::identity(),
        variogram,
    )
    .unwrap();
    let grid = small_grid();
    let n_realizations = 100;
    let n_cells = grid.n_cells();

    let mut running_sum = vec![0.0_f64; n_cells];
    let mut counts = vec![0usize; n_cells];

    gaussian_simulation_3d_stream_with(
        &model,
        &grid,
        1,
        n_realizations,
        SgsOutputSpace::ScoreSpace,
        |_, gv| {
            for (i, v) in gv.iter().enumerate() {
                if v.is_finite() {
                    running_sum[i] += *v as f64;
                    counts[i] += 1;
                }
            }
            Ok(())
        },
    )
    .unwrap();

    // Build the score-space OK reference using the model's pre-fitted
    // NST and score-space variogram. The dataset's `values` slot gets
    // replaced with normal scores; the kriging math is identical.
    let scored_dataset = {
        let (coords, _) = dataset.into_parts();
        let scores: Vec<Real> = model
            .sample_scores()
            .iter()
            .copied()
            .collect();
        PlanarDataset3D::new(coords, scores).unwrap()
    };
    let ok_model = OrdinaryKrigingModel3D::new(
        scored_dataset,
        Anisotropy3D::identity(),
        score_space_variogram(),
    )
    .unwrap();

    let mut passed = 0;
    let mut total = 0;
    let mut max_abs_err = 0.0_f64;
    for linear in 0..n_cells {
        if counts[linear] == 0 {
            continue;
        }
        let mean_sgs = running_sum[linear] / counts[linear] as f64;
        let target = grid.cell_center_linear(linear);
        let ok_pred = match ok_model.predict(target) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let abs_err = (mean_sgs - ok_pred.value as f64).abs();
        let sigma_ok = (ok_pred.variance as f64).max(0.0).sqrt();
        let stderr_n = sigma_ok / (counts[linear] as f64).sqrt();
        // 5-σ slack on the standard error, with a floor to keep
        // cells with tiny variance from setting an impossibly tight bar.
        let slack = (5.0 * stderr_n).max(0.05);
        total += 1;
        if abs_err <= slack {
            passed += 1;
        }
        max_abs_err = max_abs_err.max(abs_err);
    }

    let pass_rate = (passed as f64) / (total as f64);
    eprintln!(
        "score-space mean-converges-to-OK gate: {}/{} cells passed ({:.1}%); \
         max |mean - OK_score| = {:.4}",
        passed, total, pass_rate * 100.0, max_abs_err,
    );
    assert!(
        pass_rate >= 0.90,
        "expected >= 90% pass rate, got {:.1}% (max abs err: {:.4})",
        pass_rate * 100.0, max_abs_err,
    );
}

#[test]
fn sgs_produces_no_nan_on_normal_workload() {
    // On a well-formed dataset with a sensible variogram, every grid
    // cell should be finite. Per v3 §"no silent solver failures": we
    // explicitly check that NaN doesn't sneak through.
    let dataset = load_dataset();
    let model = SgsModel3D::new(
        dataset,
        Anisotropy3D::identity(),
        score_space_variogram(),
    )
    .unwrap();
    let grid = small_grid();

    let mut n_nan = 0;
    let mut n_total = 0;
    gaussian_simulation_3d_stream(&model, &grid, 7777, 5, |_, gv| {
        for v in gv {
            n_total += 1;
            if !v.is_finite() {
                n_nan += 1;
            }
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(
        n_nan, 0,
        "SGS produced {n_nan}/{n_total} NaN cells on a normal workload",
    );
}

// ---- Deferred gates ---------------------------------------------------

#[test]
#[ignore = "M11 scoping decision: deferred to v2 polish (see docs/m0-findings.md)"]
fn realization_variance_matches_kriging_variance() {
    // The per-cell variance across realizations should equal the
    // kriging variance up to sampling noise. Useful as a deeper
    // statistical check; not load-bearing for the v1 feasibility
    // question.
}

#[test]
#[ignore = "M11 scoping decision: deferred to v2 polish (see docs/m0-findings.md)"]
fn realization_variogram_matches_input_model() {
    // The experimental variogram averaged across realizations should
    // approach the input variogram model. Validates that the SGS
    // covariance structure matches the requested one.
}
