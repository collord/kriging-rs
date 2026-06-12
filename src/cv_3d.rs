//! Leave-one-out cross-validation for 3-D kriging models.
//!
//! For each sample `i`, fit the chosen 3-D kriging model on the
//! remaining `n − 1` samples and predict the held-out sample's value.
//! Returns one [`CvResidual`] per sample in input order.
//!
//! Mirrors upstream's 2-D [`crate::cv`] module; the [`CvResidual`] and
//! [`CvSummary`] types are reused unchanged because they're
//! metric-agnostic. The functions here differ only in which 3-D model
//! they instantiate per fold.
//!
//! K-fold CV is **not implemented in v1** — v3's M12 deliverable is
//! "Cross-validation for 3D OK/SK/UK"; LOOCV satisfies the textbook
//! validation invariants (RMSE, MSDR) and K-fold's structural shape
//! is mechanical to add later. The smaller intervention here.
//!
//! Parallelism: native builds parallelize the fold loop with rayon.
//! WASM falls back to a sequential loop.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::cv::CvResidual;
use crate::error::KrigingError;
use crate::kriging::ordinary_3d::OrdinaryKrigingModel3D;
use crate::kriging::simple_3d::SimpleKrigingModel3D;
use crate::kriging::universal_3d::{Trend3D, UniversalKrigingModel3D};
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::variogram::models::VariogramModel;

/// Build the training-set coordinates and values by leaving out index `i`.
/// Pre-allocates exact capacity to avoid reallocation in the fold loop.
fn leave_one_out_training(
    coords: &[Coord3D],
    values: &[Real],
    i: usize,
) -> (Vec<Coord3D>, Vec<Real>) {
    let n = coords.len();
    let mut fold_coords = Vec::with_capacity(n - 1);
    let mut fold_values = Vec::with_capacity(n - 1);
    for j in 0..n {
        if j == i {
            continue;
        }
        fold_coords.push(coords[j]);
        fold_values.push(values[j]);
    }
    (fold_coords, fold_values)
}

/// Leave-one-out cross-validation for [`OrdinaryKrigingModel3D`].
///
/// For each sample `i`, builds an `OrdinaryKrigingModel3D` on the
/// remaining `n − 1` samples and predicts at `coords[i]`. The residual
/// captures `(observed, predicted, kriging_variance)`. Requires at
/// least 3 samples (each fold needs ≥ 2 training points for OK).
///
/// Native: parallelized over folds with rayon. WASM: sequential.
pub fn leave_one_out_ordinary_3d(
    coords: &[Coord3D],
    values: &[Real],
    anisotropy: Anisotropy3D,
    variogram: VariogramModel,
) -> Result<Vec<CvResidual>, KrigingError> {
    let n = coords.len();
    if coords.len() != values.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({}) and values ({}) must have equal length",
            coords.len(),
            values.len()
        )));
    }
    if n < 3 {
        return Err(KrigingError::InsufficientData(3));
    }

    let fold_indices: Vec<usize> = (0..n).collect();

    let per_fold = |i: usize| -> Result<CvResidual, KrigingError> {
        let (fold_coords, fold_values) = leave_one_out_training(coords, values, i);
        let dataset = PlanarDataset3D::new(fold_coords, fold_values)?;
        let model = OrdinaryKrigingModel3D::new(dataset, anisotropy, variogram)?;
        let pred = model.predict(coords[i])?;
        Ok(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        })
    };

    #[cfg(not(target_arch = "wasm32"))]
    let residuals: Result<Vec<CvResidual>, KrigingError> =
        fold_indices.par_iter().map(|&i| per_fold(i)).collect();

    #[cfg(target_arch = "wasm32")]
    let residuals: Result<Vec<CvResidual>, KrigingError> =
        fold_indices.iter().map(|&i| per_fold(i)).collect();

    residuals
}

/// Leave-one-out cross-validation for [`SimpleKrigingModel3D`].
///
/// `mean` is the known global mean used by SK (does not change across
/// folds — SK's mean is a model assumption, not a sample statistic).
/// Native: parallelized over folds with rayon. WASM: sequential.
pub fn leave_one_out_simple_3d(
    coords: &[Coord3D],
    values: &[Real],
    anisotropy: Anisotropy3D,
    variogram: VariogramModel,
    mean: Real,
) -> Result<Vec<CvResidual>, KrigingError> {
    let n = coords.len();
    if coords.len() != values.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({}) and values ({}) must have equal length",
            coords.len(),
            values.len()
        )));
    }
    if n < 3 {
        return Err(KrigingError::InsufficientData(3));
    }

    let fold_indices: Vec<usize> = (0..n).collect();

    let per_fold = |i: usize| -> Result<CvResidual, KrigingError> {
        let (fold_coords, fold_values) = leave_one_out_training(coords, values, i);
        let dataset = PlanarDataset3D::new(fold_coords, fold_values)?;
        let model = SimpleKrigingModel3D::new(dataset, anisotropy, variogram, mean)?;
        let pred = model.predict(coords[i])?;
        Ok(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        })
    };

    #[cfg(not(target_arch = "wasm32"))]
    let residuals: Result<Vec<CvResidual>, KrigingError> =
        fold_indices.par_iter().map(|&i| per_fold(i)).collect();

    #[cfg(target_arch = "wasm32")]
    let residuals: Result<Vec<CvResidual>, KrigingError> =
        fold_indices.iter().map(|&i| per_fold(i)).collect();

    residuals
}

/// Leave-one-out cross-validation for [`UniversalKrigingModel3D`] with
/// the v1 [`Trend3D::Linear`] basis.
///
/// Requires at least 5 samples — the trend basis is 4-dimensional, so
/// each fold's training set needs ≥ 4. Native: parallelized over
/// folds. WASM: sequential.
pub fn leave_one_out_universal_3d_linear(
    coords: &[Coord3D],
    values: &[Real],
    anisotropy: Anisotropy3D,
    variogram: VariogramModel,
) -> Result<Vec<CvResidual>, KrigingError> {
    let n = coords.len();
    if coords.len() != values.len() {
        return Err(KrigingError::DimensionMismatch(format!(
            "coords ({}) and values ({}) must have equal length",
            coords.len(),
            values.len()
        )));
    }
    // Linear trend has 4 basis components, so each fold's training set
    // needs at least 4 points -- which means n >= 5.
    if n < 5 {
        return Err(KrigingError::InsufficientData(5));
    }

    let fold_indices: Vec<usize> = (0..n).collect();

    let per_fold = |i: usize| -> Result<CvResidual, KrigingError> {
        let (fold_coords, fold_values) = leave_one_out_training(coords, values, i);
        let dataset = PlanarDataset3D::new(fold_coords, fold_values)?;
        let model = UniversalKrigingModel3D::new(dataset, anisotropy, variogram, Trend3D::Linear)?;
        let pred = model.predict(coords[i])?;
        Ok(CvResidual {
            index: i,
            observed: values[i],
            predicted: pred.value,
            variance: pred.variance,
        })
    };

    #[cfg(not(target_arch = "wasm32"))]
    let residuals: Result<Vec<CvResidual>, KrigingError> =
        fold_indices.par_iter().map(|&i| per_fold(i)).collect();

    #[cfg(target_arch = "wasm32")]
    let residuals: Result<Vec<CvResidual>, KrigingError> =
        fold_indices.iter().map(|&i| per_fold(i)).collect();

    residuals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cv::CvSummary;
    use crate::variogram::models::VariogramType;
    use std::path::PathBuf;

    fn variogram() -> VariogramModel {
        VariogramModel::new(0.05, 1.0, 30.0, VariogramType::Exponential).unwrap()
    }

    fn smooth_field(c: Coord3D) -> Real {
        // Smooth deterministic field that LOOCV should track well.
        let x = c.x as f64;
        let y = c.y as f64;
        let z = c.z as f64;
        (10.0 + 5.0 * (x / 30.0).sin() + 3.0 * (y / 25.0).cos() + 0.5 * z) as Real
    }

    fn synthetic_3d_dataset() -> (Vec<Coord3D>, Vec<Real>) {
        // 30 well-spread points; smooth values plus tiny noise -> LOOCV
        // should produce small residuals.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        let mut k = 0u64;
        for i in 0..3 {
            for j in 0..5 {
                for l in 0..2 {
                    let c =
                        Coord3D::new((i as Real) * 30.0, (j as Real) * 20.0, (l as Real) * 25.0);
                    coords.push(c);
                    // Cheap deterministic "noise" derived from a counter.
                    k = k
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let n = ((k >> 32) as f64 / (1u64 << 32) as f64 - 0.5) as Real * 0.1;
                    values.push(smooth_field(c) + n);
                }
            }
        }
        (coords, values)
    }

    /// Load the M5 sample dataset for tests that want realistic data.
    fn load_m5_dataset() -> (Vec<Coord3D>, Vec<Real>) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/parity_3d/fixtures/skgstat_3d/samples.csv");
        let csv = std::fs::read_to_string(&path).expect("samples.csv");
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
        (coords, values)
    }

    #[test]
    fn loocv_ordinary_3d_produces_one_residual_per_sample_in_order() {
        let (coords, values) = synthetic_3d_dataset();
        let residuals =
            leave_one_out_ordinary_3d(&coords, &values, Anisotropy3D::identity(), variogram())
                .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i, "residual {i} should reference sample {i}");
            assert_eq!(
                r.observed, values[i],
                "observed should match the input value at index {i}",
            );
            assert!(r.predicted.is_finite(), "prediction at {i} not finite");
            assert!(r.variance.is_finite() && r.variance >= 0.0);
        }
    }

    #[test]
    fn loocv_ordinary_3d_rejects_dataset_under_3() {
        let coords = vec![Coord3D::new(0.0, 0.0, 0.0), Coord3D::new(10.0, 0.0, 0.0)];
        let values = vec![1.0 as Real, 2.0 as Real];
        let r = leave_one_out_ordinary_3d(&coords, &values, Anisotropy3D::identity(), variogram());
        assert!(matches!(r, Err(KrigingError::InsufficientData(3))));
    }

    #[test]
    fn loocv_ordinary_3d_predictions_differ_from_in_sample() {
        // Holding sample i out should produce a prediction different
        // from the in-sample prediction at the same location (which
        // would just be values[i] for low-nugget OK).
        let (coords, values) = synthetic_3d_dataset();
        let residuals =
            leave_one_out_ordinary_3d(&coords, &values, Anisotropy3D::identity(), variogram())
                .unwrap();
        let n_distinct = residuals
            .iter()
            .filter(|r| (r.predicted as f64 - r.observed as f64).abs() > 1e-6)
            .count();
        // For a smooth-ish field with non-trivial nugget, *most* held-out
        // residuals should be non-zero. (A few co-located cases could
        // legitimately produce zero residuals; >= 50% is a safe bar.)
        assert!(
            n_distinct >= coords.len() / 2,
            "expected most LOOCV predictions to differ from observed values, got {}/{}",
            n_distinct,
            coords.len(),
        );
    }

    #[test]
    fn loocv_ordinary_3d_msdr_within_reasonable_bounds() {
        // MSDR is (mean of squared residual / kriging variance). Under a
        // well-calibrated variogram, MSDR should be roughly ~1. Real
        // datasets can easily land in [0.3, 5.0]; we use [0.05, 20]
        // as a sanity bar -- this is checking "kriging variance is in
        // the right ballpark for the actual residuals," not strict
        // calibration.
        let (coords, values) = load_m5_dataset();
        let residuals = leave_one_out_ordinary_3d(
            &coords,
            &values,
            Anisotropy3D::identity(),
            // A variogram approximately fitted to the M5 dataset's empirical
            // shape. Total sill = partial sill + nugget = 55, range 100.
            VariogramModel::new(5.0, 55.0, 100.0, VariogramType::Exponential).unwrap(),
        )
        .unwrap();
        let summary = CvSummary::from_residuals(&residuals);
        assert!(summary.n > 100, "expected most folds to produce residuals");
        assert!(
            (0.05..=20.0).contains(&summary.msdr),
            "MSDR {} out of sanity range [0.05, 20]",
            summary.msdr,
        );
        assert!(summary.rmse.is_finite() && summary.rmse > 0.0);
    }

    #[test]
    fn loocv_simple_3d_produces_residuals_for_each_sample() {
        let (coords, values) = synthetic_3d_dataset();
        let mean: Real = values.iter().copied().sum::<Real>() / (values.len() as Real);
        let residuals = leave_one_out_simple_3d(
            &coords,
            &values,
            Anisotropy3D::identity(),
            variogram(),
            mean,
        )
        .unwrap();
        assert_eq!(residuals.len(), coords.len());
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.index, i);
            assert!(r.predicted.is_finite());
        }
    }

    #[test]
    fn loocv_universal_3d_linear_requires_5_samples() {
        let four_samples_coords = vec![
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(10.0, 0.0, 0.0),
            Coord3D::new(0.0, 10.0, 0.0),
            Coord3D::new(0.0, 0.0, 10.0),
        ];
        let four_values = vec![1.0 as Real, 2.0 as Real, 3.0 as Real, 4.0 as Real];
        let r = leave_one_out_universal_3d_linear(
            &four_samples_coords,
            &four_values,
            Anisotropy3D::identity(),
            variogram(),
        );
        assert!(matches!(r, Err(KrigingError::InsufficientData(5))));
    }

    #[test]
    fn loocv_universal_3d_linear_recovers_smooth_field() {
        // For a smooth deterministic field with low nugget, LOOCV
        // residuals from UK with a linear trend should be small
        // compared to the value range.
        let (coords, values) = synthetic_3d_dataset();
        let residuals = leave_one_out_universal_3d_linear(
            &coords,
            &values,
            Anisotropy3D::identity(),
            variogram(),
        )
        .unwrap();
        let summary = CvSummary::from_residuals(&residuals);
        assert!(summary.n > 0);
        // values are in roughly [10, 25] range with O(0.1) noise; RMSE
        // should be much smaller than the value range under any sane
        // kriger.
        let value_range = values
            .iter()
            .copied()
            .fold(0.0 as Real, |acc, v| acc.max(v))
            - values.iter().copied().fold(Real::MAX, |acc, v| acc.min(v));
        assert!(
            (summary.rmse as f64) < (value_range as f64) * 0.5,
            "LOOCV RMSE {} too large relative to value range {}",
            summary.rmse,
            value_range,
        );
    }
}
