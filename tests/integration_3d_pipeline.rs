//! End-to-end smoke tests for the 3-D pipeline (M7 — v0.9 tag-eligible).
//!
//! Validates that data → 3-D empirical variogram → fit produces a sensible
//! parametric `VariogramModel`, for both omnidirectional and directional
//! flows. These aren't bitwise-parity tests (those live in
//! `tests/parity_3d/`); they're "the whole pipeline composes and produces
//! finite, reasonable output" tests.
//!
//! Uses the same 150-point M5 dataset for continuity with the parity
//! fixtures.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use kriging_rs::variogram::{
    DirectionalConfig3D, EmpiricalEstimator, FitResult, PositiveReal, VariogramConfig,
    VariogramType, compute_directional_variogram_3d, compute_empirical_variogram_3d, fit_variogram,
};
use kriging_rs::{Anisotropy3D, Coord3D, GslibDirection, PlanarDataset3D, Real};

fn load_sample_dataset() -> PlanarDataset3D {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/parity_3d/fixtures/skgstat_3d/samples.csv");
    let csv = std::fs::read_to_string(&path).expect("samples.csv should exist");
    let mut coords = Vec::new();
    let mut values = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("numeric"))
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

fn assert_model_sensible(fit: &FitResult) {
    assert!(
        fit.residuals.is_finite() && fit.residuals >= 0.0,
        "residuals should be a finite non-negative number, got {}",
        fit.residuals
    );
    let (nugget, sill, range) = fit.model.params();
    assert!(nugget >= 0.0, "nugget must be non-negative");
    assert!(sill > nugget, "sill must exceed nugget");
    assert!(range > 0.0, "range must be positive");
    // gamma at lag 0 is the nugget; at large lag approaches sill.
    let g_zero = fit.model.semivariance(0.0);
    let g_large = fit.model.semivariance(range * 10.0);
    assert!(
        g_zero.is_finite() && g_large.is_finite(),
        "gamma evaluations should be finite",
    );
    // Loose monotonicity sanity: γ at the range should be greater than γ at 0
    // for the model families this test uses.
    let g_at_range = fit.model.semivariance(range);
    assert!(
        g_at_range > g_zero,
        "gamma should increase from lag 0 to the range",
    );
}

#[test]
fn omnidirectional_pipeline_produces_fittable_model() {
    let dataset = load_sample_dataset();
    let config = VariogramConfig {
        max_distance: Some(PositiveReal::try_new(120.0).unwrap()),
        n_bins: NonZeroUsize::new(12).unwrap(),
        estimator: EmpiricalEstimator::Classical,
    };
    let empirical = compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &config)
        .expect("empirical variogram should compute");
    assert!(
        !empirical.semivariances.is_empty(),
        "should have non-empty bins"
    );

    let fit =
        fit_variogram(&empirical, VariogramType::Exponential).expect("fitting should succeed");
    assert_model_sensible(&fit);
}

#[test]
fn omnidirectional_with_anisotropy_pipeline_runs() {
    // Confirms the full pipeline (3-D anisotropy -> 3-D empirical -> fit)
    // composes without surprises. Stretch z by 2x.
    let dataset = load_sample_dataset();
    let aniso = Anisotropy3D::from_rotation_matrix(
        nalgebra::Matrix3::identity(),
        nalgebra::Vector3::new(1.0, 1.0, 2.0),
    )
    .unwrap();
    let config = VariogramConfig {
        max_distance: Some(PositiveReal::try_new(150.0).unwrap()),
        n_bins: NonZeroUsize::new(12).unwrap(),
        estimator: EmpiricalEstimator::Classical,
    };
    let empirical = compute_empirical_variogram_3d(&dataset, &aniso, &config)
        .expect("anisotropic empirical variogram should compute");

    let fit = fit_variogram(&empirical, VariogramType::Spherical).expect("fitting should succeed");
    assert_model_sensible(&fit);
}

#[test]
fn directional_pipeline_produces_fittable_model() {
    // Directional (gamv-style) variogram fed into the same fitter. Use the
    // azim_north direction from the M6 fixture parameters.
    let dataset = load_sample_dataset();
    let filter = GslibDirection {
        azm_deg: 0.0,
        atol_deg: 22.5,
        bandh: 25.0,
        dip_deg: 0.0,
        dtol_deg: 22.5,
        bandv: 25.0,
    }
    .to_filter()
    .unwrap();
    let config = DirectionalConfig3D {
        xlag: PositiveReal::try_new(12.0).unwrap(),
        xltol: PositiveReal::try_new(6.0).unwrap(),
        n_lags: 10,
        estimator: EmpiricalEstimator::Classical,
    };
    let empirical = compute_directional_variogram_3d(&dataset, &filter, &config)
        .expect("directional variogram should compute");

    // Directional variograms have fewer bins; fitting should still work
    // because the EmpiricalVariogram struct is identical.
    let fit = fit_variogram(&empirical, VariogramType::Exponential)
        .expect("fitting a directional variogram should succeed");
    assert_model_sensible(&fit);
}

#[test]
fn multiple_model_types_fit_on_same_3d_empirical() {
    // Smoke-test that the existing 2-D model types all accept a 3-D
    // empirical variogram. If any model panics on 3-D-produced data this
    // catches it.
    let dataset = load_sample_dataset();
    let config = VariogramConfig {
        max_distance: Some(PositiveReal::try_new(120.0).unwrap()),
        n_bins: NonZeroUsize::new(12).unwrap(),
        estimator: EmpiricalEstimator::Classical,
    };
    let empirical =
        compute_empirical_variogram_3d(&dataset, &Anisotropy3D::identity(), &config).unwrap();

    for model_type in [
        VariogramType::Spherical,
        VariogramType::Exponential,
        VariogramType::Gaussian,
        VariogramType::Cubic,
        VariogramType::Stable,
        VariogramType::Matern,
    ] {
        let fit = fit_variogram(&empirical, model_type)
            .unwrap_or_else(|e| panic!("fit failed for {model_type:?}: {e}"));
        assert_model_sensible(&fit);
    }
}
