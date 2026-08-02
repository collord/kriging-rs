//! Collocated cokriging & cosimulation (Markov Model 1), plus a peek at the coregionalization
//! foundation for full cokriging.
//!
//! Run with: `cargo run --example collocated_cokriging`

use kriging_rs::cokriging::{
    CokrigingKind, CokrigingModel, Coregionalization, CoregionalizationStructure, CorrelationBasis,
    LmcFitOptions, MultiVariableDataset, MultiVariableSamples, SillMatrix,
    compute_empirical_cross_variogram, fit_lmc,
};
use kriging_rs::simulation::{SimulationOptions, collocated_cosimulate, cosimulate};
use kriging_rs::{
    CollocatedCokrigingModel, EmpiricalEstimator, GeoCoord, GeoDataset, SecondaryVariable,
    VariogramConfig, VariogramModel, VariogramType,
};
use std::num::NonZeroUsize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sparse primary observations (e.g. ground measurements).
    let coords = vec![
        GeoCoord::try_new(37.77, -122.42)?,
        GeoCoord::try_new(37.78, -122.41)?,
        GeoCoord::try_new(37.76, -122.40)?,
        GeoCoord::try_new(37.75, -122.43)?,
    ];
    let primary = vec![15.0, 18.0, 14.0, 13.0];

    // A secondary covariate (e.g. a remotely sensed field) known everywhere. Here we estimate
    // its moments and its collocated correlation with the primary from paired samples.
    let secondary_at_data = vec![2.2, 2.7, 2.1, 1.6];
    let secondary = SecondaryVariable::from_paired(&primary, &secondary_at_data)?;
    println!(
        "secondary: mean={:.3}, std={:.3}, corr={:.3}",
        secondary.mean(),
        secondary.std_dev(),
        secondary.correlation()
    );

    let variogram = VariogramModel::new(0.1, 6.0, 5.0, VariogramType::Exponential)?;
    let primary_mean = 15.0;

    // --- Collocated cokriging: predict the primary using the collocated secondary datum. ---
    let model = CollocatedCokrigingModel::new(
        GeoDataset::new(coords.clone(), primary.clone())?,
        variogram,
        primary_mean,
        secondary,
    )?;
    let target = GeoCoord::try_new(37.765, -122.415)?;
    let secondary_at_target = 2.4;
    let pred = model.predict(target, secondary_at_target)?;
    println!(
        "cokriging prediction: value={:.3}, variance={:.6}",
        pred.value, pred.variance
    );

    // --- Collocated cosimulation: one conditional realization on a small target set. ---
    let targets = vec![
        GeoCoord::try_new(37.765, -122.415)?,
        GeoCoord::try_new(37.775, -122.405)?,
    ];
    let target_secondary = vec![2.4, 2.7];
    let realization = collocated_cosimulate(
        &coords,
        &primary,
        &targets,
        &target_secondary,
        variogram,
        primary_mean,
        secondary,
        SimulationOptions::new(42),
    )?;
    println!("cosimulated primary at targets: {realization:?}");

    // --- Foundation for full cokriging: a 2-variable Linear Model of Coregionalization. ---
    // Nugget + exponential structure, each with a positive-semidefinite 2x2 sill matrix.
    let lmc = Coregionalization::new(vec![
        CoregionalizationStructure::new(
            CorrelationBasis::Nugget,
            SillMatrix::from_rows(vec![vec![0.5, 0.0], vec![0.0, 0.5]])?,
        ),
        CoregionalizationStructure::new(
            CorrelationBasis::Model(VariogramModel::new(
                0.0,
                1.0,
                5.0,
                VariogramType::Exponential,
            )?),
            SillMatrix::from_rows(vec![vec![4.0, 1.5], vec![1.5, 3.0]])?,
        ),
    ])?;
    println!(
        "LMC: {} variables, {} structures; C_01(0)={:.3}",
        lmc.n_variables(),
        lmc.n_structures(),
        lmc.cross_sill(0, 1),
    );

    // --- Full block cokriging: predict the primary from a 2-variable dataset via the LMC. ---
    // Here the secondary is sampled at the *same* locations as the primary (isotopic).
    let secondary_values = vec![1.9, 2.6, 1.7, 1.4];
    let multi = MultiVariableDataset::new(coords, vec![primary, secondary_values])?;
    let cokriging = CokrigingModel::new(multi, lmc, CokrigingKind::Ordinary)?;
    let ck = cokriging.predict(0, target)?; // target variable 0 = primary
    println!(
        "block ordinary cokriging (var 0): value={:.3}, variance={:.6}",
        ck.value, ck.variance
    );

    // --- Heterotopic cokriging: a sparse primary + a denser secondary, sampled elsewhere. ---
    let primary_sites = vec![
        GeoCoord::try_new(37.77, -122.42)?,
        GeoCoord::try_new(37.75, -122.43)?,
    ];
    let secondary_sites = vec![
        GeoCoord::try_new(37.78, -122.41)?,
        GeoCoord::try_new(37.76, -122.40)?,
        GeoCoord::try_new(37.765, -122.415)?,
    ];
    let hetero = MultiVariableSamples::new(vec![
        (primary_sites, vec![15.0, 13.0]),
        (secondary_sites, vec![2.9, 1.8, 2.4]),
    ])?;
    let coreg = Coregionalization::new(vec![CoregionalizationStructure::new(
        CorrelationBasis::Model(VariogramModel::new(
            0.0,
            1.0,
            5.0,
            VariogramType::Exponential,
        )?),
        SillMatrix::from_rows(vec![vec![6.0, 1.4], vec![1.4, 0.5]])?,
    )])?;
    let het_model =
        CokrigingModel::new_heterotopic(hetero.clone(), coreg.clone(), CokrigingKind::Ordinary)?;
    let het_pred = het_model.predict(0, target)?;
    println!(
        "heterotopic cokriging (sparse primary): value={:.3}, variance={:.6}",
        het_pred.value, het_pred.variance
    );

    // --- Multivariate cosimulation: joint realization of both variables at new locations. ---
    let cosim_targets = vec![
        GeoCoord::try_new(37.765, -122.42)?,
        GeoCoord::try_new(37.77, -122.41)?,
    ];
    let cosim = cosimulate(
        hetero,
        coreg,
        vec![14.0, 2.3], // per-variable means
        &cosim_targets,
        SimulationOptions::new(7),
    )?;
    println!(
        "cosimulated primary:   {:?}\ncosimulated secondary: {:?}",
        cosim.samples[0], cosim.samples[1]
    );

    // --- Fit an LMC from data (Goulard–Voltz) instead of hand-building the sills. ---
    let fit_coords = vec![
        GeoCoord::try_new(37.70, -122.45)?,
        GeoCoord::try_new(37.71, -122.44)?,
        GeoCoord::try_new(37.72, -122.46)?,
        GeoCoord::try_new(37.73, -122.43)?,
        GeoCoord::try_new(37.74, -122.45)?,
        GeoCoord::try_new(37.75, -122.42)?,
        GeoCoord::try_new(37.76, -122.44)?,
        GeoCoord::try_new(37.77, -122.46)?,
    ];
    let var_a = vec![12.0, 13.5, 11.0, 14.0, 12.5, 10.5, 13.0, 11.5];
    let var_b = vec![3.0, 3.6, 2.7, 3.9, 3.2, 2.5, 3.4, 2.9]; // correlated with var_a
    let fit_ds = MultiVariableDataset::new(fit_coords, vec![var_a, var_b])?;
    let config = VariogramConfig {
        max_distance: None,
        n_bins: NonZeroUsize::new(5).unwrap(),
        estimator: EmpiricalEstimator::Classical,
    };
    let cross = compute_empirical_cross_variogram(&fit_ds, &config)?;
    let fitted = fit_lmc(
        &cross,
        vec![
            CorrelationBasis::Nugget,
            CorrelationBasis::Model(VariogramModel::new(
                0.0,
                1.0,
                3.0,
                VariogramType::Exponential,
            )?),
        ],
        LmcFitOptions::default(),
    )?;
    println!(
        "fitted LMC: residual={:.4}, iterations={}, C_00(0)={:.3}, C_01(0)={:.3}",
        fitted.residual,
        fitted.iterations,
        fitted.coregionalization.cross_sill(0, 0),
        fitted.coregionalization.cross_sill(0, 1),
    );

    Ok(())
}
