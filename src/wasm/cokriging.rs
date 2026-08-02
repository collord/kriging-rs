//! WebAssembly bindings for the [`crate::cokriging`] subsystem.
//!
//! Every entry point is a free function taking one options object (deserialized once via
//! `serde_wasm_bindgen`) so a new parameter is additive — mirroring the [`VariogramSpec`]
//! boundary convention used elsewhere. Coverage:
//!
//! - `cokrigeCollocated` / `collocatedSecondaryFromPaired` / `collocatedCosimulate` — Markov
//!   Model 1 collocated cokriging & cosimulation of a primary from a dense secondary.
//! - `cokrige` / `cosimulate` — full block cokriging and multivariate cosimulation over a
//!   Linear Model of Coregionalization.
//! - `computeCrossVariogram` / `fitLmc` — empirical cross-variograms and Goulard–Voltz LMC
//!   fitting (returns an admissible coregionalization ready to feed back into `cokrige`).

#![allow(clippy::unnecessary_cast)]

use std::num::NonZeroUsize;

use js_sys::Float64Array;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::Real;
use crate::cokriging::fitting::LmcFitOptions;
use crate::cokriging::{
    CokrigingKind, CokrigingModel, CollocatedCokrigingModel, Coregionalization,
    CoregionalizationStructure, CorrelationBasis, MultiVariableDataset, MultiVariableSamples,
    SecondaryVariable, SillMatrix, compute_empirical_cross_variogram, fit_lmc,
};
use crate::geo_dataset::GeoDataset;
use crate::simulation::{collocated_cosimulate, cosimulate};
use crate::variogram::VariogramSpec;
use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal, VariogramConfig};
use crate::variogram::models::VariogramModel;

use super::{
    coded_err, err_to_js, kriging_err_to_js, map_predictions, parse_simulation_options,
    spec_to_model, to_coords, variogram_type_name,
};

// ---------------------------------------------------------------------------
// Shared spec fragments
// ---------------------------------------------------------------------------

/// A collocated secondary's global characterization: its mean, standard deviation, and the
/// primary–secondary correlation at lag zero.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SecondarySpec {
    mean: f64,
    std_dev: f64,
    correlation: f64,
}

impl SecondarySpec {
    fn build(&self) -> Result<SecondaryVariable, JsValue> {
        SecondaryVariable::new(
            self.mean as Real,
            self.std_dev as Real,
            self.correlation as Real,
        )
        .map_err(kriging_err_to_js)
    }
}

/// One LMC basic structure's correlation basis: either a pure nugget, or the normalized
/// covariance of a [`VariogramSpec`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CorrelationBasisSpec {
    #[serde(default)]
    nugget: bool,
    #[serde(default)]
    variogram: Option<VariogramSpec>,
}

impl CorrelationBasisSpec {
    fn build(&self) -> Result<CorrelationBasis, JsValue> {
        if self.nugget {
            Ok(CorrelationBasis::Nugget)
        } else {
            let vg = self.variogram.as_ref().ok_or_else(|| {
                coded_err(
                    "correlation basis needs `nugget: true` or a `variogram`",
                    "invalid_basis",
                )
            })?;
            Ok(CorrelationBasis::Model(spec_to_model(vg)?))
        }
    }
}

/// One LMC structure: a correlation basis plus its `p × p` (PSD) sill matrix, given as rows.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CoregionalizationStructureSpec {
    basis: CorrelationBasisSpec,
    sills: Vec<Vec<f64>>,
}

/// A Linear Model of Coregionalization as a list of structures.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CoregionalizationSpec {
    structures: Vec<CoregionalizationStructureSpec>,
}

impl CoregionalizationSpec {
    fn build(&self) -> Result<Coregionalization, JsValue> {
        let mut structures = Vec::with_capacity(self.structures.len());
        for s in &self.structures {
            let basis = s.basis.build()?;
            let rows: Vec<Vec<Real>> = s
                .sills
                .iter()
                .map(|row| row.iter().map(|v| *v as Real).collect())
                .collect();
            let sills = SillMatrix::from_rows(rows).map_err(kriging_err_to_js)?;
            structures.push(CoregionalizationStructure::new(basis, sills));
        }
        Coregionalization::new(structures).map_err(kriging_err_to_js)
    }
}

/// One variable's samples for the heterotopic layout.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VariableSamplesSpec {
    lats: Vec<f64>,
    lons: Vec<f64>,
    values: Vec<f64>,
}

/// Multi-variable data, either **isotopic** (`lats`/`lons`/`variables` share locations) or
/// **heterotopic** (`perVariable`, each variable with its own locations).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MultiVariableDataSpec {
    #[serde(default)]
    lats: Option<Vec<f64>>,
    #[serde(default)]
    lons: Option<Vec<f64>>,
    #[serde(default)]
    variables: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    per_variable: Option<Vec<VariableSamplesSpec>>,
}

impl MultiVariableDataSpec {
    fn build_dataset(&self) -> Result<MultiVariableDataset, JsValue> {
        let (lats, lons, variables) = match (&self.lats, &self.lons, &self.variables) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                return Err(coded_err(
                    "isotopic data needs `lats`, `lons`, and `variables`",
                    "invalid_input",
                ));
            }
        };
        let coords = to_coords(lats, lons)?;
        let vars: Vec<Vec<Real>> = variables
            .iter()
            .map(|col| col.iter().map(|v| *v as Real).collect())
            .collect();
        MultiVariableDataset::new(coords, vars).map_err(kriging_err_to_js)
    }

    fn build_samples(&self) -> Result<MultiVariableSamples, JsValue> {
        if let Some(per) = &self.per_variable {
            let per_variable = per
                .iter()
                .map(|v| {
                    let coords = to_coords(&v.lats, &v.lons)?;
                    let values: Vec<Real> = v.values.iter().map(|x| *x as Real).collect();
                    Ok((coords, values))
                })
                .collect::<Result<Vec<_>, JsValue>>()?;
            MultiVariableSamples::new(per_variable).map_err(kriging_err_to_js)
        } else {
            Ok(self.build_dataset()?.to_samples())
        }
    }
}

/// Cokriging estimator: `type: "ordinary"`, or `type: "simple"` with one `means` entry per
/// variable.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CokrigingKindSpec {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    means: Option<Vec<f64>>,
}

impl CokrigingKindSpec {
    fn build(&self) -> Result<CokrigingKind, JsValue> {
        match self.kind.to_ascii_lowercase().as_str() {
            "ordinary" => Ok(CokrigingKind::Ordinary),
            "simple" => {
                let means = self.means.as_ref().ok_or_else(|| {
                    coded_err("simple cokriging needs a `means` array", "invalid_input")
                })?;
                Ok(CokrigingKind::Simple {
                    means: means.iter().map(|m| *m as Real).collect(),
                })
            }
            _ => Err(coded_err(
                "kind `type` must be 'simple' or 'ordinary'",
                "unknown_kind",
            )),
        }
    }
}

fn variogram_config(n_bins: usize, max_distance: Option<f64>) -> Result<VariogramConfig, JsValue> {
    let n_bins = NonZeroUsize::new(n_bins)
        .ok_or_else(|| coded_err("nBins must be at least 1", "invalid_bins"))?;
    let max_distance = match max_distance {
        Some(v) if v > 0.0 && v.is_finite() => {
            Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?)
        }
        Some(_) => {
            return Err(coded_err(
                "maxDistance must be finite and positive",
                "invalid_input",
            ));
        }
        None => None,
    };
    Ok(VariogramConfig {
        max_distance,
        n_bins,
        estimator: EmpiricalEstimator::Classical,
    })
}

// ---------------------------------------------------------------------------
// Serialized results
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsSecondary {
    mean: f64,
    std_dev: f64,
    correlation: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsVariogramSpec {
    variogram_type: String,
    nugget: f64,
    sill: f64,
    range: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shape2: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsBasis {
    nugget: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    variogram: Option<JsVariogramSpec>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsStructure {
    basis: JsBasis,
    sills: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsCoregionalization {
    structures: Vec<JsStructure>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsLmcFit {
    coregionalization: JsCoregionalization,
    residual: f64,
    iterations: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsCrossVariogram {
    n_variables: usize,
    distances: Vec<f64>,
    n_pairs: Vec<usize>,
    /// `gamma[bin]` is the flattened row-major `p × p` cross-semivariance matrix at that lag.
    gamma: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsCosimulationResult {
    n_variables: usize,
    n_targets: usize,
    /// `samples[v][i]` is the simulated value of variable `v` at target `i`.
    samples: Vec<Vec<f64>>,
}

fn model_to_spec(model: &VariogramModel) -> JsVariogramSpec {
    let (nugget, sill, range) = model.params();
    JsVariogramSpec {
        variogram_type: variogram_type_name(model.variogram_type()).to_string(),
        nugget: nugget as f64,
        sill: sill as f64,
        range: range as f64,
        shape: model.shape().map(|s| s as f64),
        shape2: model.shape2().map(|s| s as f64),
    }
}

fn coregionalization_to_js(coreg: &Coregionalization) -> JsCoregionalization {
    let structures = coreg
        .structures()
        .iter()
        .map(|s| {
            let basis = match s.basis() {
                CorrelationBasis::Nugget => JsBasis {
                    nugget: true,
                    variogram: None,
                },
                CorrelationBasis::Model(m) => JsBasis {
                    nugget: false,
                    variogram: Some(model_to_spec(m)),
                },
            };
            let sills = s.sills();
            let p = sills.n_variables();
            let rows: Vec<Vec<f64>> = (0..p)
                .map(|i| (0..p).map(|j| sills.get(i, j) as f64).collect())
                .collect();
            JsStructure { basis, sills: rows }
        })
        .collect();
    JsCoregionalization { structures }
}

// ---------------------------------------------------------------------------
// Collocated cokriging (Markov Model 1)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollocatedCokrigingOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    values: Vec<f64>,
    variogram: VariogramSpec,
    primary_mean: f64,
    secondary: SecondarySpec,
    target_lats: Vec<f64>,
    target_lons: Vec<f64>,
    target_secondary_values: Vec<f64>,
}

/// Collocated simple cokriging of a primary variable given a dense secondary known at every
/// target. Returns `[{ value, variance }]`, one per target.
#[wasm_bindgen(js_name = cokrigeCollocated)]
pub fn wasm_cokrige_collocated(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: CollocatedCokrigingOptions =
        serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    let coords = to_coords(&opts.lats, &opts.lons)?;
    let values: Vec<Real> = opts.values.iter().map(|v| *v as Real).collect();
    let dataset = GeoDataset::new(coords, values).map_err(kriging_err_to_js)?;
    let model = spec_to_model(&opts.variogram)?;
    let secondary = opts.secondary.build()?;
    let ck = CollocatedCokrigingModel::new(dataset, model, opts.primary_mean as Real, secondary)
        .map_err(kriging_err_to_js)?;
    let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
    let secondary_values: Vec<Real> = opts
        .target_secondary_values
        .iter()
        .map(|v| *v as Real)
        .collect();
    let preds = ck
        .predict_batch(&targets, &secondary_values)
        .map_err(kriging_err_to_js)?;
    serde_wasm_bindgen::to_value(&map_predictions(preds)).map_err(err_to_js)
}

/// Estimate a [`SecondarySpec`] (`{ mean, stdDev, correlation }`) from paired primary/secondary
/// samples measured at the same locations.
#[wasm_bindgen(js_name = collocatedSecondaryFromPaired)]
pub fn wasm_collocated_secondary_from_paired(
    primary: &[f64],
    secondary: &[f64],
) -> Result<JsValue, JsValue> {
    let p: Vec<Real> = primary.iter().map(|v| *v as Real).collect();
    let s: Vec<Real> = secondary.iter().map(|v| *v as Real).collect();
    let sec = SecondaryVariable::from_paired(&p, &s).map_err(kriging_err_to_js)?;
    let out = JsSecondary {
        mean: sec.mean() as f64,
        std_dev: sec.std_dev() as f64,
        correlation: sec.correlation() as f64,
    };
    serde_wasm_bindgen::to_value(&out).map_err(err_to_js)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CollocatedCosimulateOptions {
    conditioning_lats: Vec<f64>,
    conditioning_lons: Vec<f64>,
    conditioning_values: Vec<f64>,
    target_lats: Vec<f64>,
    target_lons: Vec<f64>,
    target_secondary_values: Vec<f64>,
    variogram: VariogramSpec,
    primary_mean: f64,
    secondary: SecondarySpec,
    seed: u64,
    #[serde(default)]
    target_order: Option<Vec<u32>>,
}

/// Sequential Gaussian cosimulation of a primary honoring a collocated secondary. Returns a
/// `Float64Array` of primary realizations in input target order.
#[wasm_bindgen(js_name = collocatedCosimulate)]
pub fn wasm_collocated_cosimulate(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: CollocatedCosimulateOptions =
        serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    let cond_coords = to_coords(&opts.conditioning_lats, &opts.conditioning_lons)?;
    let cond_values: Vec<Real> = opts
        .conditioning_values
        .iter()
        .map(|v| *v as Real)
        .collect();
    let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
    let secondary_values: Vec<Real> = opts
        .target_secondary_values
        .iter()
        .map(|v| *v as Real)
        .collect();
    let model = spec_to_model(&opts.variogram)?;
    let secondary = opts.secondary.build()?;
    let sim_opts = parse_simulation_options(opts.seed, opts.target_order);
    let out = collocated_cosimulate(
        &cond_coords,
        &cond_values,
        &targets,
        &secondary_values,
        model,
        opts.primary_mean as Real,
        secondary,
        sim_opts,
    )
    .map_err(kriging_err_to_js)?;
    let out_f64: Vec<f64> = out.into_iter().map(|v| v as f64).collect();
    Ok(Float64Array::from(out_f64.as_slice()).into())
}

// ---------------------------------------------------------------------------
// Full block cokriging over an LMC
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CokrigeOptions {
    data: MultiVariableDataSpec,
    coregionalization: CoregionalizationSpec,
    kind: CokrigingKindSpec,
    target_variable: usize,
    target_lats: Vec<f64>,
    target_lons: Vec<f64>,
}

/// Full block cokriging (simple or ordinary) of one `targetVariable` over an LMC, using every
/// variable's data. Returns `[{ value, variance }]`, one per target.
#[wasm_bindgen(js_name = cokrige)]
pub fn wasm_cokrige(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: CokrigeOptions = serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    let samples = opts.data.build_samples()?;
    let coreg = opts.coregionalization.build()?;
    let kind = opts.kind.build()?;
    let model = CokrigingModel::new_heterotopic(samples, coreg, kind).map_err(kriging_err_to_js)?;
    let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
    let preds = model
        .predict_batch(opts.target_variable, &targets)
        .map_err(kriging_err_to_js)?;
    serde_wasm_bindgen::to_value(&map_predictions(preds)).map_err(err_to_js)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CosimulateOptions {
    data: MultiVariableDataSpec,
    coregionalization: CoregionalizationSpec,
    means: Vec<f64>,
    target_lats: Vec<f64>,
    target_lons: Vec<f64>,
    seed: u64,
    #[serde(default)]
    target_order: Option<Vec<u32>>,
}

/// Multivariate sequential Gaussian cosimulation of all variables jointly over an LMC (simple
/// cokriging with known `means`). Returns `{ nVariables, nTargets, samples }`.
#[wasm_bindgen(js_name = cosimulate)]
pub fn wasm_cosimulate(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: CosimulateOptions = serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    let samples = opts.data.build_samples()?;
    let coreg = opts.coregionalization.build()?;
    let means: Vec<Real> = opts.means.iter().map(|m| *m as Real).collect();
    let targets = to_coords(&opts.target_lats, &opts.target_lons)?;
    let sim_opts = parse_simulation_options(opts.seed, opts.target_order);
    let result =
        cosimulate(samples, coreg, means, &targets, sim_opts).map_err(kriging_err_to_js)?;
    let out = JsCosimulationResult {
        n_variables: result.n_variables,
        n_targets: result.n_targets,
        samples: result
            .samples
            .iter()
            .map(|row| row.iter().map(|v| *v as f64).collect())
            .collect(),
    };
    serde_wasm_bindgen::to_value(&out).map_err(err_to_js)
}

// ---------------------------------------------------------------------------
// Empirical cross-variogram + LMC fitting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CrossVariogramOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    variables: Vec<Vec<f64>>,
    n_bins: usize,
    #[serde(default)]
    max_distance: Option<f64>,
}

/// Empirical (isotopic) cross-variogram matrix per lag. Returns
/// `{ nVariables, distances, nPairs, gamma }` where `gamma[bin]` is the flattened `p × p` matrix.
#[wasm_bindgen(js_name = computeCrossVariogram)]
pub fn wasm_compute_cross_variogram(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: CrossVariogramOptions = serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    let data = MultiVariableDataSpec {
        lats: Some(opts.lats),
        lons: Some(opts.lons),
        variables: Some(opts.variables),
        per_variable: None,
    };
    let dataset = data.build_dataset()?;
    let config = variogram_config(opts.n_bins, opts.max_distance)?;
    let emp = compute_empirical_cross_variogram(&dataset, &config).map_err(kriging_err_to_js)?;
    let p = emp.n_variables();
    let gamma: Vec<Vec<f64>> = (0..emp.n_bins())
        .map(|bin| {
            (0..p)
                .flat_map(|i| (0..p).map(move |j| (bin, i, j)))
                .map(|(bin, i, j)| emp.gamma(bin, i, j) as f64)
                .collect()
        })
        .collect();
    let out = JsCrossVariogram {
        n_variables: p,
        distances: emp.distances().iter().map(|d| *d as f64).collect(),
        n_pairs: emp.n_pairs().to_vec(),
        gamma,
    };
    serde_wasm_bindgen::to_value(&out).map_err(err_to_js)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FitLmcOptions {
    lats: Vec<f64>,
    lons: Vec<f64>,
    variables: Vec<Vec<f64>>,
    n_bins: usize,
    #[serde(default)]
    max_distance: Option<f64>,
    bases: Vec<CorrelationBasisSpec>,
    #[serde(default)]
    max_iterations: Option<usize>,
    #[serde(default)]
    tolerance: Option<f64>,
}

/// Fit an LMC to isotopic data by Goulard–Voltz: computes the empirical cross-variogram, then
/// fits each basic structure's PSD sill matrix. Returns `{ coregionalization, residual,
/// iterations }`; the `coregionalization` is directly consumable by `cokrige`/`cosimulate`.
#[wasm_bindgen(js_name = fitLmc)]
pub fn wasm_fit_lmc(options: JsValue) -> Result<JsValue, JsValue> {
    let opts: FitLmcOptions = serde_wasm_bindgen::from_value(options).map_err(err_to_js)?;
    let data = MultiVariableDataSpec {
        lats: Some(opts.lats),
        lons: Some(opts.lons),
        variables: Some(opts.variables),
        per_variable: None,
    };
    let dataset = data.build_dataset()?;
    let config = variogram_config(opts.n_bins, opts.max_distance)?;
    let empirical =
        compute_empirical_cross_variogram(&dataset, &config).map_err(kriging_err_to_js)?;
    let bases = opts
        .bases
        .iter()
        .map(|b| b.build())
        .collect::<Result<Vec<_>, JsValue>>()?;
    let mut lmc_opts = LmcFitOptions::default();
    if let Some(m) = opts.max_iterations {
        lmc_opts.max_iterations = m;
    }
    if let Some(t) = opts.tolerance {
        lmc_opts.tolerance = t as Real;
    }
    let fit = fit_lmc(&empirical, bases, lmc_opts).map_err(kriging_err_to_js)?;
    let out = JsLmcFit {
        coregionalization: coregionalization_to_js(&fit.coregionalization),
        residual: fit.residual as f64,
        iterations: fit.iterations,
    };
    serde_wasm_bindgen::to_value(&out).map_err(err_to_js)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pin the camelCase field contracts that `serde_wasm_bindgen` relies on at runtime (which
    // cannot be exercised here without `wasm-pack`). We deserialize via `serde_json`, which uses
    // the same serde field names / `rename_all` / `serde(default)` rules.

    #[test]
    fn coregionalization_spec_round_trips() {
        let json = r#"{
            "structures": [
                { "basis": { "nugget": true }, "sills": [[1.0, 0.5], [0.5, 2.0]] },
                { "basis": { "variogram": { "variogramType": "exponential", "nugget": 0.0, "sill": 1.0, "range": 10.0 } },
                  "sills": [[3.0, 1.0], [1.0, 4.0]] }
            ]
        }"#;
        let spec: CoregionalizationSpec = serde_json::from_str(json).unwrap();
        let coreg = spec.build().unwrap();
        assert_eq!(coreg.n_variables(), 2);
        assert_eq!(coreg.n_structures(), 2);
        // The nugget structure contributes its full sill at lag 0 and nothing beyond.
        assert!((coreg.cross_sill(0, 0) - (1.0 + 3.0)).abs() < 1e-6);
    }

    #[test]
    fn data_spec_selects_isotopic_or_heterotopic() {
        let iso = r#"{"lats":[0.0,0.1],"lons":[0.0,0.1],"variables":[[1.0,2.0],[3.0,4.0]]}"#;
        let spec: MultiVariableDataSpec = serde_json::from_str(iso).unwrap();
        let samples = spec.build_samples().unwrap();
        assert_eq!(samples.n_variables(), 2);

        let het = r#"{"perVariable":[
            {"lats":[0.0],"lons":[0.0],"values":[1.0]},
            {"lats":[0.1,0.2],"lons":[0.1,0.2],"values":[2.0,3.0]}
        ]}"#;
        let spec: MultiVariableDataSpec = serde_json::from_str(het).unwrap();
        let samples = spec.build_samples().unwrap();
        assert_eq!(samples.n_variables(), 2);
        assert_eq!(samples.n_points(1), 2);
    }

    #[test]
    fn kind_spec_parses_type_and_means() {
        let ordinary: CokrigingKindSpec = serde_json::from_str(r#"{"type":"ordinary"}"#).unwrap();
        assert!(matches!(ordinary.build().unwrap(), CokrigingKind::Ordinary));

        let simple: CokrigingKindSpec =
            serde_json::from_str(r#"{"type":"simple","means":[0.0,1.5]}"#).unwrap();
        match simple.build().unwrap() {
            CokrigingKind::Simple { means } => assert_eq!(means.len(), 2),
            _ => panic!("expected simple"),
        }
    }
}
