use kriging_rs::Real;
use kriging_rs::variogram::empirical::{
    PositiveReal, VariogramConfig, compute_empirical_variogram,
};
use kriging_rs::variogram::fitting::fit_variogram;
use kriging_rs::variogram::models::{VariogramModel, VariogramType};
use kriging_rs::{
    BinomialKrigingModel, BinomialObservation, BinomialPrior, GeoCoord, GeoDataset,
    OrdinaryKrigingModel,
};
use std::num::NonZeroUsize;

fn coord(lat: Real, lon: Real) -> GeoCoord {
    GeoCoord::try_new(lat, lon).unwrap()
}

fn fit_variogram_for_type(
    dataset: &GeoDataset,
    config: &VariogramConfig,
    variogram_type: VariogramType,
) -> VariogramModel {
    let empirical = compute_empirical_variogram(dataset, config).expect("empirical variogram");
    fit_variogram(&empirical, variogram_type)
        .expect("fit variogram")
        .model
}

#[test]
fn ordinary_kriging_predicts_midpoint_between_linear_samples() {
    let coords = vec![coord(0.0, 0.0), coord(0.0, 1.0), coord(0.0, 2.0)];
    let values = vec![10.0, 20.0, 30.0];
    let variogram = VariogramModel::new(0.01, 100.0, 500.0, VariogramType::Exponential).unwrap();
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    let model = OrdinaryKrigingModel::new(dataset, variogram).expect("model should build");
    let pred = model
        .predict(coord(0.0, 1.0))
        .expect("prediction should succeed");
    assert!((pred.value - 20.0).abs() < 1e-4);
}

#[test]
fn ordinary_pipeline_with_variogram_fit_runs_end_to_end() {
    let coords = vec![coord(0.0, 0.0), coord(0.5, 0.5), coord(1.0, 1.0)];
    let values = vec![2.0, 3.0, 4.0];
    let predictions = vec![coord(0.75, 0.75)];
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    let model = fit_variogram_for_type(
        &dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let ordinary = OrdinaryKrigingModel::new(dataset, model).expect("ordinary model should build");
    let out = ordinary
        .predict_batch(&predictions)
        .expect("ordinary batch predict");
    assert_eq!(out.len(), 1);
    assert!(out[0].value.is_finite());
    assert!(out[0].variance.is_finite());
}

#[test]
fn binomial_pipeline_with_variogram_fit_runs_end_to_end() {
    let obs = vec![
        BinomialObservation::new(coord(0.0, 0.0), 3, 10).unwrap(),
        BinomialObservation::new(coord(0.0, 1.0), 6, 10).unwrap(),
        BinomialObservation::new(coord(1.0, 0.5), 8, 10).unwrap(),
    ];
    let coords = obs.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let logits = obs
        .iter()
        .map(BinomialObservation::smoothed_logit)
        .collect::<Vec<_>>();
    let dataset = GeoDataset::new(coords, logits).expect("dataset");
    let model = fit_variogram_for_type(
        &dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let binomial = BinomialKrigingModel::new(obs, model).expect("binomial model should build");
    let out = binomial
        .predict_batch(&[coord(0.5, 0.5)])
        .expect("binomial batch predict");
    assert_eq!(out.len(), 1);
    assert!((0.0..=1.0).contains(&out[0].prevalence));
}

#[test]
fn explicit_variogram_config_accepts_custom_settings() {
    let coords = vec![
        coord(0.0, 0.0),
        coord(0.5, 0.5),
        coord(1.0, 1.0),
        coord(1.0, 0.0),
    ];
    let values = vec![2.0, 3.0, 4.0, 3.5];
    let predictions = vec![coord(0.75, 0.75)];
    let dataset = GeoDataset::new(coords, values).expect("dataset");
    let model = fit_variogram_for_type(
        &dataset,
        &VariogramConfig {
            max_distance: Some(PositiveReal::try_new(200.0).unwrap()),
            n_bins: NonZeroUsize::new(4).unwrap(),
            estimator: Default::default(),
        },
        VariogramType::Gaussian,
    );
    let ordinary = OrdinaryKrigingModel::new(dataset, model).expect("ordinary model should build");
    let out = ordinary
        .predict_batch(&predictions)
        .expect("ordinary batch predict");
    assert_eq!(out.len(), 1);
    assert!(out[0].value.is_finite());
    assert!(out[0].variance.is_finite());
}

#[test]
fn binomial_prior_changes_prevalence_with_fitted_model() {
    let obs = vec![
        BinomialObservation::new(coord(0.0, 0.0), 0, 5).unwrap(),
        BinomialObservation::new(coord(0.0, 1.0), 2, 5).unwrap(),
        BinomialObservation::new(coord(1.0, 0.0), 5, 5).unwrap(),
    ];
    let pred_coord = [coord(0.25, 0.25)];
    let coords = obs.iter().map(|o| o.coord()).collect::<Vec<_>>();
    let default_prior = BinomialPrior::default();
    let default_logits = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(default_prior))
        .collect::<Vec<_>>();
    let default_dataset = GeoDataset::new(coords.clone(), default_logits).expect("dataset");
    let default_model = fit_variogram_for_type(
        &default_dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let out_default =
        BinomialKrigingModel::new_with_prior(obs.clone(), default_model, default_prior)
            .expect("default prior model")
            .predict_batch(&pred_coord)
            .expect("default prior prediction");

    let heavy_prior = BinomialPrior::new(5.0, 5.0).unwrap();
    let heavy_logits = obs
        .iter()
        .map(|o| o.smoothed_logit_with_prior(heavy_prior))
        .collect::<Vec<_>>();
    let heavy_dataset = GeoDataset::new(coords, heavy_logits).expect("dataset");
    let heavy_model = fit_variogram_for_type(
        &heavy_dataset,
        &VariogramConfig::default(),
        VariogramType::Exponential,
    );
    let out_heavy = BinomialKrigingModel::new_with_prior(obs, heavy_model, heavy_prior)
        .expect("heavy prior model")
        .predict_batch(&pred_coord)
        .expect("heavy prior prediction");
    assert_eq!(out_default.len(), 1);
    assert_eq!(out_heavy.len(), 1);
    assert!((out_default[0].prevalence - out_heavy[0].prevalence).abs() > 1e-6);
}

#[test]
fn binomial_predict_batch_matches_repeated_predict() {
    let obs = vec![
        BinomialObservation::new(coord(0.0, 0.0), 1, 5).unwrap(),
        BinomialObservation::new(coord(0.0, 1.0), 2, 5).unwrap(),
        BinomialObservation::new(coord(1.0, 0.0), 4, 5).unwrap(),
        BinomialObservation::new(coord(1.0, 1.0), 5, 5).unwrap(),
    ];
    let model = BinomialKrigingModel::new(
        obs,
        VariogramModel::new(0.01, 2.0, 300.0, VariogramType::Exponential).unwrap(),
    )
    .expect("binomial model");
    let coords = vec![coord(0.2, 0.2), coord(0.4, 0.7), coord(0.8, 0.3)];
    let batch = model.predict_batch(&coords).expect("batch predict");
    let singles = coords
        .iter()
        .map(|coord| model.predict(*coord).expect("single predict"))
        .collect::<Vec<_>>();
    assert_eq!(batch.len(), singles.len());
    for (batch_pred, single_pred) in batch.iter().zip(singles.iter()) {
        assert!((batch_pred.prevalence - single_pred.prevalence).abs() < 1e-12);
        assert!((batch_pred.logit_value - single_pred.logit_value).abs() < 1e-12);
        assert!((batch_pred.variance - single_pred.variance).abs() < 1e-12);
    }
}

/// Regression: the nugget must reach the kriging-matrix diagonal. With the total sill held
/// fixed, a larger nugget means less spatial structure, so the variance near (but not at)
/// stations must rise; far beyond the range it is `sill + 1/(1ᵀK⁻¹1)`, so at least the sill;
/// and kriging stays (nearly) exact at a station. Before the fix `γ(0)` returned the nugget,
/// `C(0)` was the partial sill, and the variance *fell* as the nugget rose — in every
/// kriging path below.
#[test]
fn larger_nugget_raises_kriging_variance_in_every_path() {
    use kriging_rs::{
        Anisotropy2D, Anisotropy3D, Coord3D, OrdinaryKrigingModel3D, PlanarDataset3D,
        ProjectedCoord, ProjectedDataset, ProjectedKrigingModel,
    };

    let (sill, range) = (1.0, 5.0);
    let pts: [(Real, Real); 4] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.2)];
    let values: Vec<Real> = vec![0.0, 2.0, 1.0, 3.0];
    let (near, far) = ((0.3, 0.4), (500.0, 500.0));

    // (variance near the stations, variance far away, estimate at station 1)
    let run = |nugget: Real| -> [(Real, Real, Real); 3] {
        let vg = VariogramModel::new(nugget, sill, range, VariogramType::Exponential).unwrap();

        // 2-D geographic: 1 degree ≈ 111 km, so scale the planar layout down to stay in range.
        let geo = |(x, y): (Real, Real)| coord(y * 0.009, x * 0.009);
        let ds = GeoDataset::new(pts.iter().map(|&p| geo(p)).collect(), values.clone()).unwrap();
        let ok = OrdinaryKrigingModel::new(ds, vg).unwrap();
        let ok2d = (
            ok.predict(geo(near)).unwrap().variance,
            ok.predict(coord(60.0, 60.0)).unwrap().variance,
            ok.predict(geo(pts[1])).unwrap().value,
        );

        let pc = |(x, y): (Real, Real)| ProjectedCoord::new(x, y);
        let ds =
            ProjectedDataset::new(pts.iter().map(|&p| pc(p)).collect(), values.clone()).unwrap();
        let pk = ProjectedKrigingModel::new(ds, vg, Anisotropy2D::isotropic()).unwrap();
        let proj = (
            pk.predict(pc(near)).unwrap().variance,
            pk.predict(pc(far)).unwrap().variance,
            pk.predict(pc(pts[1])).unwrap().value,
        );

        let c3 = |(x, y): (Real, Real)| Coord3D::new(x, y, 0.0);
        let ds =
            PlanarDataset3D::new(pts.iter().map(|&p| c3(p)).collect(), values.clone()).unwrap();
        let k3 = OrdinaryKrigingModel3D::new(ds, Anisotropy3D::identity(), vg).unwrap();
        let ok3d = (
            k3.predict(c3(near)).unwrap().variance,
            k3.predict(c3(far)).unwrap().variance,
            k3.predict(c3(pts[1])).unwrap().value,
        );
        [ok2d, proj, ok3d]
    };

    let labels = ["2-D ordinary", "projected", "3-D ordinary"];
    let by_nugget: Vec<_> = [0.0, 0.3, 0.6].into_iter().map(run).collect();
    for (k, label) in labels.iter().enumerate() {
        for w in by_nugget.windows(2) {
            let (lo, hi) = (w[0][k], w[1][k]);
            assert!(
                hi.0 > lo.0 + 0.05,
                "{label}: near variance must rise with nugget: {lo:?} -> {hi:?}"
            );
        }
        for row in &by_nugget {
            let (_, far_var, at_station) = row[k];
            assert!(
                far_var >= sill - 1e-4,
                "{label}: far variance {far_var} below sill {sill}"
            );
            // Not bit-exact: the diagonal jitter (≤ 1 % of the nugget) sits on the matrix
            // diagonal but not on the right-hand side.
            assert!(
                (at_station - values[1]).abs() < 1e-2,
                "{label}: not exact at station: {at_station}"
            );
        }
    }
}
