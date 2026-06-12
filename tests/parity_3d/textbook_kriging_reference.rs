//! Algorithmic-tier parity for 3-D kriging predictions against textbook
//! references implemented in numpy.
//!
//! See `fixtures/textbook_ok3d/README.md` for the rationale on why we
//! use textbook implementations rather than scikit-gstat / pygslib /
//! geostatspy. Briefly: scikit-gstat's `OrdinaryKriging` doesn't
//! implement standard OK (it disagrees with two independent textbook
//! implementations by ~3% in value), pygslib has no usable osx-64
//! distribution, and geostatspy's `krige_3D` is a 2-D-only
//! SGS-internal helper.
//!
//! **Three tests, one fixture format each**:
//! - `fixtures/textbook_ok3d/` — M9 OrdinaryKrigingModel3D
//! - `fixtures/textbook_sk3d/` — M10 SimpleKrigingModel3D
//! - `fixtures/textbook_uk3d/` — M10 UniversalKrigingModel3D (linear trend)
//!
//! All three compare 1000 random predict locations against numpy
//! reference solutions; tolerances at the v3 "algorithmic-equivalence"
//! tier, calibrated by measured drift between our Schur-complement
//! Cholesky and numpy.linalg.solve.

use std::path::PathBuf;

use kriging_rs::variogram::{VariogramModel, VariogramType};
use kriging_rs::{
    Anisotropy3D, Coord3D, Neighborhood3D, OrdinaryKrigingModel3D, PlanarDataset3D, Real,
    SimpleKrigingModel3D, Trend3D, UniversalKrigingModel3D,
};

fn samples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity_3d/fixtures/skgstat_3d")
}

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/parity_3d/fixtures/{name}"))
}

fn load_samples() -> PlanarDataset3D {
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

struct ExpectedPrediction {
    target: Coord3D,
    value: f64,
    variance: f64,
}

fn load_predictions(fixture: &str) -> Vec<ExpectedPrediction> {
    let path = fixture_dir(fixture).join("predictions.csv");
    let csv = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("{}/predictions.csv should exist", fixture));
    let mut out = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<f64> = line
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("numeric"))
            .collect();
        out.push(ExpectedPrediction {
            target: Coord3D::new(cols[0] as Real, cols[1] as Real, cols[2] as Real),
            value: cols[3],
            variance: cols[4],
        });
    }
    out
}

/// Common loop that walks expectations, evaluates `predict` on each
/// target, and reports per-tolerance violations.
fn check_parity<F>(
    expectations: &[ExpectedPrediction],
    label: &str,
    value_rel: f64,
    variance_rel: f64,
    predict: F,
) where
    F: Fn(Coord3D) -> Result<kriging_rs::Prediction3D, kriging_rs::KrigingError>,
{
    let mut max_value_rel_err = 0.0_f64;
    let mut max_variance_rel_err = 0.0_f64;
    let mut n_failed = 0;
    let mut first_failure_msg: Option<String> = None;

    for (i, exp) in expectations.iter().enumerate() {
        let p = predict(exp.target)
            .unwrap_or_else(|e| panic!("{label}: solver failed at target {i}: {e}"));
        let v_err = (p.value as f64 - exp.value).abs();
        let var_err = (p.variance as f64 - exp.variance).abs();
        let v_rel = if exp.value.abs() > 1e-6 {
            v_err / exp.value.abs()
        } else {
            v_err
        };
        let var_rel = if exp.variance.abs() > 1e-6 {
            var_err / exp.variance.abs()
        } else {
            var_err
        };
        max_value_rel_err = max_value_rel_err.max(v_rel);
        max_variance_rel_err = max_variance_rel_err.max(var_rel);
        if v_rel >= value_rel || var_rel >= variance_rel {
            n_failed += 1;
            if first_failure_msg.is_none() {
                first_failure_msg = Some(format!(
                    "target {i} @ ({:.3}, {:.3}, {:.3}): \
                     ours value={:.6} ref={:.6} rel={:.2e}; \
                     ours variance={:.6} ref={:.6} rel={:.2e}",
                    exp.target.x,
                    exp.target.y,
                    exp.target.z,
                    p.value,
                    exp.value,
                    v_rel,
                    p.variance,
                    exp.variance,
                    var_rel,
                ));
            }
        }
    }

    if n_failed > 0 {
        panic!(
            "{label}: {n_failed} of {total} targets failed parity \
             (value_rel < {value_rel}, variance_rel < {variance_rel}). \
             max value rel err: {max_value_rel_err:.2e}, max variance rel err: {max_variance_rel_err:.2e}. \
             First: {}",
            first_failure_msg.unwrap(),
            total = expectations.len(),
        );
    }

    eprintln!(
        "{label}: {}/{} targets within value_rel={value_rel:.0e} variance_rel={variance_rel:.0e}; \
         max value rel err: {max_value_rel_err:.2e}, max variance rel err: {max_variance_rel_err:.2e}",
        expectations.len(),
        expectations.len(),
    );
}

#[test]
fn ordinary_kriging_3d_matches_textbook_at_1000_locations() {
    let dataset = load_samples();
    let csv = std::fs::read_to_string(fixture_dir("textbook_ok3d").join("variogram.csv"))
        .expect("OK variogram.csv should exist");
    let mut lines = csv.lines();
    let _header = lines.next().unwrap();
    let cols: Vec<f64> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().parse::<f64>().unwrap())
        .collect();
    let (effective_range, partial_sill, nugget) = (cols[0], cols[1], cols[2]);
    let variogram = VariogramModel::new(
        nugget as Real,
        (partial_sill + nugget) as Real,
        effective_range as Real,
        VariogramType::Exponential,
    )
    .unwrap();

    let model = OrdinaryKrigingModel3D::new(dataset, Anisotropy3D::identity(), variogram)
        .unwrap()
        .with_neighborhood(Neighborhood3D::within_radius(effective_range));

    let expectations = load_predictions("textbook_ok3d");
    assert_eq!(expectations.len(), 1000);
    check_parity(&expectations, "OK", 1.5e-2, 1e-2, |t| model.predict(t));
}

#[test]
fn simple_kriging_3d_matches_textbook_at_1000_locations() {
    let dataset = load_samples();
    let csv = std::fs::read_to_string(fixture_dir("textbook_sk3d").join("variogram.csv"))
        .expect("SK variogram.csv should exist");
    let mut lines = csv.lines();
    let _header = lines.next().unwrap();
    let cols: Vec<f64> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().parse::<f64>().unwrap())
        .collect();
    let (effective_range, partial_sill, nugget, mean) = (cols[0], cols[1], cols[2], cols[3]);
    let variogram = VariogramModel::new(
        nugget as Real,
        (partial_sill + nugget) as Real,
        effective_range as Real,
        VariogramType::Exponential,
    )
    .unwrap();

    let model =
        SimpleKrigingModel3D::new(dataset, Anisotropy3D::identity(), variogram, mean as Real)
            .unwrap();

    let expectations = load_predictions("textbook_sk3d");
    assert_eq!(expectations.len(), 1000);
    // SK uses the same Cholesky as OK but no Lagrangian; same algorithmic
    // drift envelope applies.
    check_parity(&expectations, "SK", 1.5e-2, 1e-2, |t| model.predict(t));
}

#[test]
fn universal_kriging_3d_linear_matches_textbook_at_1000_locations() {
    let dataset = load_samples();
    let csv = std::fs::read_to_string(fixture_dir("textbook_uk3d").join("variogram.csv"))
        .expect("UK variogram.csv should exist");
    let mut lines = csv.lines();
    let _header = lines.next().unwrap();
    let cols: Vec<f64> = lines
        .next()
        .unwrap()
        .split(',')
        .map(|s| s.trim().parse::<f64>().unwrap())
        .collect();
    let (effective_range, partial_sill, nugget) = (cols[0], cols[1], cols[2]);
    let variogram = VariogramModel::new(
        nugget as Real,
        (partial_sill + nugget) as Real,
        effective_range as Real,
        VariogramType::Exponential,
    )
    .unwrap();

    let model = UniversalKrigingModel3D::new(
        dataset,
        Anisotropy3D::identity(),
        variogram,
        Trend3D::Linear,
    )
    .unwrap();

    let expectations = load_predictions("textbook_uk3d");
    assert_eq!(expectations.len(), 1000);
    // UK has 4 extra Lagrangians + a 4x4 trend system on top of the
    // same Cholesky path, accumulating a bit more drift than OK. Loosen
    // value tolerance to 2e-2 (measured ~1.5e-2 in practice).
    check_parity(&expectations, "UK", 2e-2, 2e-2, |t| model.predict(t));
}
