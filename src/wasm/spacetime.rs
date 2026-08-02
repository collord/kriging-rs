//! WebAssembly bindings for the [`crate::spacetime`] module.
//!
//! Layout mirrors the 2-D bindings in [`super`]: one `WasmSpaceTime*Kriging` struct per
//! kriging family, each with `fromArrays` and zero-copy `predictBatchArrays` constructors,
//! plus stand-alone functions for empirical-variogram computation and parametric fitting.

#![allow(clippy::too_many_arguments)]
// JS interop only accepts `f64`. See note in `super` (src/wasm/mod.rs).
#![allow(clippy::unnecessary_cast)]

use js_sys::{Float64Array, Object};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

use crate::BinomialBuildNotes;
use crate::Real;
use crate::distance::GeoCoord;
use crate::projected::{Anisotropy2D, ProjectedCoord};
use crate::spacetime::{
    EmpiricalSpaceTimeVariogram, GeoMetric, ProjectedMetric, SpaceTimeBinomialKrigingModel,
    SpaceTimeBinomialObservation, SpaceTimeCoord, SpaceTimeDataset, SpaceTimeFitConfig,
    SpaceTimeFitResult, SpaceTimeOrdinaryKrigingModel, SpaceTimeSimpleKrigingModel,
    SpaceTimeUniversalKrigingModel, SpaceTimeUniversalTrend, SpaceTimeVariogram,
    SpaceTimeVariogramConfig, SpaceTimeVariogramType, compute_empirical_spacetime_variogram,
    fit_spacetime_variogram,
};
use crate::variogram::VariogramSpec;
use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal};
use crate::variogram::models::{VariogramModel, VariogramType};

use super::{
    JsBinomialPrediction, JsPrediction, binomial_cv_result_to_js, binomial_many_simulation_to_js,
    binomial_simulation_to_js, coded_err, cv_result_to_js, err_to_js, kriging_err_to_js,
    map_binomial_predictions, map_predictions, parse_binomial_prior, parse_simulation_options,
    set_object_field, spec_to_model, split_binomial_predictions, split_predictions,
};

const FAMILY_HELP: &str = "family must be 'separable' or 'productSum'";

/// Space-time variogram description carried as one (de)serializable object.
///
/// `spatial` and `temporal` are each a [`VariogramSpec`]; `family` selects the space-time
/// construction (`separable` or `productSum`), and `k1`/`k2`/`k3` are the product-sum weights.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SpaceTimeVariogramSpec {
    family: String,
    spatial: VariogramSpec,
    temporal: VariogramSpec,
    #[serde(default)]
    k1: Option<f64>,
    #[serde(default)]
    k2: Option<f64>,
    #[serde(default)]
    k3: Option<f64>,
}

impl SpaceTimeVariogramSpec {
    fn build(&self) -> Result<SpaceTimeVariogram, JsValue> {
        let spatial = spec_to_model(&self.spatial)?;
        let temporal = spec_to_model(&self.temporal)?;
        match self.family.to_ascii_lowercase().as_str() {
            "separable" => {
                SpaceTimeVariogram::new_separable(spatial, temporal).map_err(kriging_err_to_js)
            }
            "product_sum" | "productsum" | "product-sum" => {
                let k1 = self.k1.unwrap_or(1.0) as Real;
                let k2 = self.k2.unwrap_or(0.0) as Real;
                let k3 = self.k3.unwrap_or(0.0) as Real;
                SpaceTimeVariogram::new_product_sum(spatial, temporal, k1, k2, k3)
                    .map_err(kriging_err_to_js)
            }
            _ => Err(coded_err(FAMILY_HELP, "unknown_family")),
        }
    }
}

/// Deserialize a JS space-time variogram spec object and build the model.
fn spacetime_variogram_from_js(variogram: JsValue) -> Result<SpaceTimeVariogram, JsValue> {
    let spec: SpaceTimeVariogramSpec =
        serde_wasm_bindgen::from_value(variogram).map_err(err_to_js)?;
    spec.build()
}

fn parse_universal_trend(name: &str) -> Result<SpaceTimeUniversalTrend, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "constant" => SpaceTimeUniversalTrend::Constant,
        "linearintime" | "linear_in_time" => SpaceTimeUniversalTrend::LinearInTime,
        "quadraticintime" | "quadratic_in_time" => SpaceTimeUniversalTrend::QuadraticInTime,
        "linearinspace" | "linear_in_space" => SpaceTimeUniversalTrend::LinearInSpace,
        "linearinspaceandtime" | "linear_in_space_and_time" => {
            SpaceTimeUniversalTrend::LinearInSpaceAndTime
        }
        "quadraticinspaceandtime" | "quadratic_in_space_and_time" => {
            SpaceTimeUniversalTrend::QuadraticInSpaceAndTime
        }
        _ => {
            return Err(coded_err(
                "trend must be 'constant', 'linearInTime', 'quadraticInTime', \
                 'linearInSpace', 'linearInSpaceAndTime', or 'quadraticInSpaceAndTime'",
                "unknown_trend",
            ));
        }
    })
}

fn parse_estimator(name: &str) -> Result<EmpiricalEstimator, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "classical" | "matheron" => EmpiricalEstimator::Classical,
        "cressiehawkins" | "cressie_hawkins" | "cressie-hawkins" => {
            EmpiricalEstimator::CressieHawkins
        }
        _ => {
            return Err(coded_err(
                "estimator must be 'classical' (Matheron) or 'cressieHawkins'",
                "unknown_estimator",
            ));
        }
    })
}

fn parse_family(name: &str) -> Result<SpaceTimeVariogramType, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "separable" => SpaceTimeVariogramType::Separable,
        "product_sum" | "productsum" | "product-sum" => SpaceTimeVariogramType::ProductSum,
        _ => return Err(coded_err(FAMILY_HELP, "unknown_family")),
    })
}

fn build_geo_dataset(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
) -> Result<SpaceTimeDataset<GeoCoord>, JsValue> {
    if lats.len() != lons.len() || lats.len() != times.len() || lats.len() != values.len() {
        return Err(coded_err(
            "lats, lons, times, and values must all have the same length",
            "mismatched_arrays",
        ));
    }
    let mut coords = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let geo = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        coords.push(SpaceTimeCoord::try_new(geo, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    SpaceTimeDataset::new(coords, values_real).map_err(kriging_err_to_js)
}

fn build_geo_targets(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
) -> Result<Vec<SpaceTimeCoord<GeoCoord>>, JsValue> {
    if lats.len() != lons.len() || lats.len() != times.len() {
        return Err(coded_err(
            "lats, lons, and times must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let geo = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        out.push(SpaceTimeCoord::try_new(geo, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

fn build_projected_dataset(
    xs: &[f64],
    ys: &[f64],
    times: &[f64],
    values: &[f64],
) -> Result<SpaceTimeDataset<ProjectedCoord>, JsValue> {
    if xs.len() != ys.len() || xs.len() != times.len() || xs.len() != values.len() {
        return Err(coded_err(
            "xs, ys, times, and values must all have the same length",
            "mismatched_arrays",
        ));
    }
    let mut coords = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let proj = ProjectedCoord::new(xs[i] as Real, ys[i] as Real);
        coords.push(SpaceTimeCoord::try_new(proj, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    SpaceTimeDataset::new(coords, values_real).map_err(kriging_err_to_js)
}

fn build_projected_targets(
    xs: &[f64],
    ys: &[f64],
    times: &[f64],
) -> Result<Vec<SpaceTimeCoord<ProjectedCoord>>, JsValue> {
    if xs.len() != ys.len() || xs.len() != times.len() {
        return Err(coded_err(
            "xs, ys, and times must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let proj = ProjectedCoord::new(xs[i] as Real, ys[i] as Real);
        out.push(SpaceTimeCoord::try_new(proj, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

fn projected_metric(major_angle_deg: f64, range_ratio: f64) -> Result<ProjectedMetric, JsValue> {
    let aniso = Anisotropy2D::new(major_angle_deg as Real, range_ratio as Real)
        .map_err(kriging_err_to_js)?;
    Ok(ProjectedMetric::with_anisotropy(aniso))
}

fn variogram_type_name(vt: VariogramType) -> &'static str {
    match vt {
        VariogramType::Spherical => "spherical",
        VariogramType::Exponential => "exponential",
        VariogramType::Gaussian => "gaussian",
        VariogramType::Cubic => "cubic",
        VariogramType::Stable => "stable",
        VariogramType::Matern => "matern",
        VariogramType::Power => "power",
        VariogramType::HoleEffect => "holeeffect",
        VariogramType::ConfluentHypergeometric => "confluenthypergeometric",
    }
}

// ---------- Geo: Ordinary ----------

#[wasm_bindgen]
pub struct WasmSpaceTimeOrdinaryKriging {
    inner: SpaceTimeOrdinaryKrigingModel<GeoMetric>,
}

#[wasm_bindgen]
impl WasmSpaceTimeOrdinaryKriging {
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        values: &[f64],
        variogram: JsValue,
    ) -> Result<WasmSpaceTimeOrdinaryKriging, JsValue> {
        let dataset = build_geo_dataset(lats, lons, times, values)?;
        let variogram = spacetime_variogram_from_js(variogram)?;
        let inner = SpaceTimeOrdinaryKrigingModel::new(GeoMetric, dataset, variogram)
            .map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_predictions(out)).map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }
}

// ---------- Geo: Simple ----------

#[wasm_bindgen]
pub struct WasmSpaceTimeSimpleKriging {
    inner: SpaceTimeSimpleKrigingModel<GeoMetric>,
}

#[wasm_bindgen]
impl WasmSpaceTimeSimpleKriging {
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        values: &[f64],
        mean: f64,
        variogram: JsValue,
    ) -> Result<WasmSpaceTimeSimpleKriging, JsValue> {
        let dataset = build_geo_dataset(lats, lons, times, values)?;
        let variogram = spacetime_variogram_from_js(variogram)?;
        let inner = SpaceTimeSimpleKrigingModel::new(GeoMetric, dataset, variogram, mean as Real)
            .map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }
}

// ---------- Geo: Universal ----------

#[wasm_bindgen]
pub struct WasmSpaceTimeUniversalKriging {
    inner: SpaceTimeUniversalKrigingModel<GeoMetric>,
}

#[wasm_bindgen]
impl WasmSpaceTimeUniversalKriging {
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        values: &[f64],
        trend: &str,
        variogram: JsValue,
    ) -> Result<WasmSpaceTimeUniversalKriging, JsValue> {
        let dataset = build_geo_dataset(lats, lons, times, values)?;
        let variogram = spacetime_variogram_from_js(variogram)?;
        let trend = parse_universal_trend(trend)?;
        let inner = SpaceTimeUniversalKrigingModel::new(GeoMetric, dataset, variogram, trend)
            .map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }
}

// ---------- Geo: Binomial ----------

#[wasm_bindgen]
pub struct WasmSpaceTimeBinomialKriging {
    inner: SpaceTimeBinomialKrigingModel<GeoMetric>,
    build_notes: BinomialBuildNotes,
}

#[wasm_bindgen]
impl WasmSpaceTimeBinomialKriging {
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
        successes: &[u32],
        trials: &[u32],
        variogram: JsValue,
    ) -> Result<WasmSpaceTimeBinomialKriging, JsValue> {
        if lats.len() != lons.len()
            || lats.len() != times.len()
            || lats.len() != successes.len()
            || lats.len() != trials.len()
        {
            return Err(coded_err(
                "lats, lons, times, successes, trials must all have the same length",
                "mismatched_arrays",
            ));
        }
        let mut observations = Vec::new();
        let mut zero_trial_drops: Vec<usize> = Vec::new();
        for i in 0..lats.len() {
            if trials[i] == 0 {
                zero_trial_drops.push(i);
                continue;
            }
            let coord =
                GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
            let st = SpaceTimeCoord::try_new(coord, times[i] as Real).map_err(kriging_err_to_js)?;
            observations.push(
                SpaceTimeBinomialObservation::new(st, successes[i], trials[i])
                    .map_err(kriging_err_to_js)?,
            );
        }
        if observations.len() < 2 {
            return Err(coded_err(
                "need at least two non-zero-trial space-time sites after dropping trials==0",
                "insufficient_data",
            ));
        }
        let variogram = spacetime_variogram_from_js(variogram)?;
        let fit = SpaceTimeBinomialKrigingModel::new(GeoMetric, observations, variogram)
            .map_err(kriging_err_to_js)?;
        let mut build_notes = fit.notes;
        build_notes.zero_trial_dropped_indices = zero_trial_drops;
        build_notes.zero_trial_dropped_indices.sort_unstable();
        Ok(Self {
            inner: fit.model,
            build_notes,
        })
    }

    #[wasm_bindgen(js_name = getBuildNotes)]
    pub fn get_build_notes(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.build_notes).map_err(err_to_js)
    }

    pub fn predict(&self, lat: f64, lon: f64, time: f64) -> Result<JsValue, JsValue> {
        let coord = GeoCoord::try_new(lat as Real, lon as Real).map_err(kriging_err_to_js)?;
        let target = SpaceTimeCoord::try_new(coord, time as Real).map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsBinomialPrediction {
            prevalence: pred.prevalence as f64,
            logit_value: pred.logit_value as f64,
            variance: pred.variance as f64,
            prevalence_variance: pred.prevalence_variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&map_binomial_predictions(out)).map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(
        &self,
        lats: &[f64],
        lons: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_geo_targets(lats, lons, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        let (prev, logits, vars, prev_vars) = split_binomial_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "prevalences",
            &Float64Array::from(prev.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "logitValues",
            &Float64Array::from(logits.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(vars.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "prevalenceVariances",
            &Float64Array::from(prev_vars.as_slice()).into(),
        )?;
        Ok(result.into())
    }
}

// ---------- Projected: Ordinary ----------

#[wasm_bindgen]
pub struct WasmSpaceTimeOrdinaryProjectedKriging {
    inner: SpaceTimeOrdinaryKrigingModel<ProjectedMetric>,
}

#[wasm_bindgen]
impl WasmSpaceTimeOrdinaryProjectedKriging {
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        xs: &[f64],
        ys: &[f64],
        times: &[f64],
        values: &[f64],
        major_angle_deg: f64,
        range_ratio: f64,
        variogram: JsValue,
    ) -> Result<WasmSpaceTimeOrdinaryProjectedKriging, JsValue> {
        let dataset = build_projected_dataset(xs, ys, times, values)?;
        let variogram = spacetime_variogram_from_js(variogram)?;
        let metric = projected_metric(major_angle_deg, range_ratio)?;
        let inner = SpaceTimeOrdinaryKrigingModel::new(metric, dataset, variogram)
            .map_err(kriging_err_to_js)?;
        Ok(Self { inner })
    }

    pub fn predict(&self, x: f64, y: f64, time: f64) -> Result<JsValue, JsValue> {
        let target =
            SpaceTimeCoord::try_new(ProjectedCoord::new(x as Real, y as Real), time as Real)
                .map_err(kriging_err_to_js)?;
        let pred = self.inner.predict(target).map_err(kriging_err_to_js)?;
        serde_wasm_bindgen::to_value(&JsPrediction {
            value: pred.value as f64,
            variance: pred.variance as f64,
        })
        .map_err(err_to_js)
    }

    #[wasm_bindgen(js_name = predictBatchArrays)]
    pub fn predict_batch_arrays(
        &self,
        xs: &[f64],
        ys: &[f64],
        times: &[f64],
    ) -> Result<JsValue, JsValue> {
        let targets = build_projected_targets(xs, ys, times)?;
        let out = self
            .inner
            .predict_batch(&targets)
            .map_err(kriging_err_to_js)?;
        let (values, variances) = split_predictions(out);
        let result = Object::new();
        set_object_field(
            &result,
            "values",
            &Float64Array::from(values.as_slice()).into(),
        )?;
        set_object_field(
            &result,
            "variances",
            &Float64Array::from(variances.as_slice()).into(),
        )?;
        Ok(result.into())
    }
}

// ---------- Empirical + fitting helpers ----------

fn empirical_to_js(emp: &EmpiricalSpaceTimeVariogram) -> Result<JsValue, JsValue> {
    let result = Object::new();
    set_object_field(
        &result,
        "nSpatialBins",
        &JsValue::from_f64(emp.n_spatial_bins as f64),
    )?;
    set_object_field(
        &result,
        "nTemporalBins",
        &JsValue::from_f64(emp.n_temporal_bins as f64),
    )?;
    let spatial: Vec<f64> = emp.spatial_lags.iter().map(|v| *v as f64).collect();
    let temporal: Vec<f64> = emp.temporal_lags.iter().map(|v| *v as f64).collect();
    let semis: Vec<f64> = emp.semivariances.iter().map(|v| *v as f64).collect();
    let counts: Vec<f64> = emp.n_pairs.iter().map(|v| *v as f64).collect();
    set_object_field(
        &result,
        "spatialLags",
        &Float64Array::from(spatial.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "temporalLags",
        &Float64Array::from(temporal.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "semivariances",
        &Float64Array::from(semis.as_slice()).into(),
    )?;
    set_object_field(
        &result,
        "nPairs",
        &Float64Array::from(counts.as_slice()).into(),
    )?;
    Ok(result.into())
}

fn variogram_to_js(model: VariogramModel) -> Result<JsValue, JsValue> {
    let (nugget, sill, range) = model.params();
    let obj = Object::new();
    set_object_field(
        &obj,
        "variogramType",
        &JsValue::from_str(variogram_type_name(model.variogram_type())),
    )?;
    set_object_field(&obj, "nugget", &JsValue::from_f64(nugget as f64))?;
    set_object_field(&obj, "sill", &JsValue::from_f64(sill as f64))?;
    set_object_field(&obj, "range", &JsValue::from_f64(range as f64))?;
    if let Some(s) = model.shape() {
        set_object_field(&obj, "shape", &JsValue::from_f64(s as f64))?;
    }
    Ok(obj.into())
}

fn fit_to_js(fit: SpaceTimeFitResult) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    let (family_name, k1, k2, k3) = match fit.model {
        SpaceTimeVariogram::Separable { .. } => ("separable", 1.0 as Real, 0.0, 0.0),
        SpaceTimeVariogram::ProductSum { k1, k2, k3, .. } => ("productSum", k1, k2, k3),
    };
    set_object_field(&obj, "family", &JsValue::from_str(family_name))?;
    set_object_field(&obj, "spatial", &variogram_to_js(fit.model.spatial())?)?;
    set_object_field(&obj, "temporal", &variogram_to_js(fit.model.temporal())?)?;
    set_object_field(&obj, "k1", &JsValue::from_f64(k1 as f64))?;
    set_object_field(&obj, "k2", &JsValue::from_f64(k2 as f64))?;
    set_object_field(&obj, "k3", &JsValue::from_f64(k3 as f64))?;
    set_object_field(&obj, "residuals", &JsValue::from_f64(fit.residuals as f64))?;
    Ok(obj.into())
}

#[wasm_bindgen(js_name = wasmComputeEmpiricalSpaceTimeVariogram)]
pub fn wasm_compute_empirical_spacetime_variogram(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    max_spatial_distance: Option<f64>,
    max_temporal_lag: Option<f64>,
    n_spatial_bins: usize,
    n_temporal_bins: usize,
    estimator: &str,
) -> Result<JsValue, JsValue> {
    let dataset = build_geo_dataset(lats, lons, times, values)?;
    let n_spatial = std::num::NonZeroUsize::new(n_spatial_bins)
        .ok_or_else(|| coded_err("nSpatialBins must be > 0", "invalid_input"))?;
    let n_temporal = std::num::NonZeroUsize::new(n_temporal_bins)
        .ok_or_else(|| coded_err("nTemporalBins must be > 0", "invalid_input"))?;
    let estimator = parse_estimator(estimator)?;
    let max_s = match max_spatial_distance {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let max_t = match max_temporal_lag {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let config = SpaceTimeVariogramConfig {
        max_spatial_distance: max_s,
        max_temporal_lag: max_t,
        n_spatial_bins: n_spatial,
        n_temporal_bins: n_temporal,
        estimator,
    };
    let emp = compute_empirical_spacetime_variogram(&GeoMetric, &dataset, &config)
        .map_err(kriging_err_to_js)?;
    empirical_to_js(&emp)
}

#[wasm_bindgen(js_name = wasmFitSpaceTimeVariogram)]
pub fn wasm_fit_spacetime_variogram(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    max_spatial_distance: Option<f64>,
    max_temporal_lag: Option<f64>,
    n_spatial_bins: usize,
    n_temporal_bins: usize,
    estimator: &str,
    family: &str,
    spatial_model: &str,
    temporal_model: &str,
) -> Result<JsValue, JsValue> {
    let emp_js = wasm_compute_empirical_spacetime_variogram(
        lats,
        lons,
        times,
        values,
        max_spatial_distance,
        max_temporal_lag,
        n_spatial_bins,
        n_temporal_bins,
        estimator,
    )?;
    // Recompute the empirical variogram in Rust form for the actual fit (avoiding a JS
    // round-trip). This duplicates the loop above but keeps the JS API simple.
    let dataset = build_geo_dataset(lats, lons, times, values)?;
    let n_spatial = std::num::NonZeroUsize::new(n_spatial_bins)
        .ok_or_else(|| coded_err("nSpatialBins must be > 0", "invalid_input"))?;
    let n_temporal = std::num::NonZeroUsize::new(n_temporal_bins)
        .ok_or_else(|| coded_err("nTemporalBins must be > 0", "invalid_input"))?;
    let estimator_v = parse_estimator(estimator)?;
    let max_s = match max_spatial_distance {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let max_t = match max_temporal_lag {
        Some(v) => Some(PositiveReal::try_new(v as Real).map_err(kriging_err_to_js)?),
        None => None,
    };
    let emp = compute_empirical_spacetime_variogram(
        &GeoMetric,
        &dataset,
        &SpaceTimeVariogramConfig {
            max_spatial_distance: max_s,
            max_temporal_lag: max_t,
            n_spatial_bins: n_spatial,
            n_temporal_bins: n_temporal,
            estimator: estimator_v,
        },
    )
    .map_err(kriging_err_to_js)?;
    let family = parse_family(family)?;
    let spatial_model = parse_variogram_type(spatial_model)?;
    let temporal_model = parse_variogram_type(temporal_model)?;
    let config = SpaceTimeFitConfig {
        family,
        spatial_model,
        temporal_model,
    };
    let fit = fit_spacetime_variogram(&emp, config).map_err(kriging_err_to_js)?;
    let result = Object::new();
    set_object_field(&result, "empirical", &emp_js)?;
    set_object_field(&result, "fit", &fit_to_js(fit)?)?;
    Ok(result.into())
}

fn parse_variogram_type(name: &str) -> Result<VariogramType, JsValue> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "spherical" => VariogramType::Spherical,
        "exponential" => VariogramType::Exponential,
        "gaussian" => VariogramType::Gaussian,
        "cubic" => VariogramType::Cubic,
        "stable" => VariogramType::Stable,
        "matern" => VariogramType::Matern,
        "power" => VariogramType::Power,
        "holeeffect" | "hole_effect" | "hole-effect" => VariogramType::HoleEffect,
        _ => return Err(coded_err("unknown variogram type", "unknown_variogram")),
    })
}

// ---------------------------------------------------------------------------
// Space–time cross-validation
// ---------------------------------------------------------------------------

fn st_build_geo_coords(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
) -> Result<Vec<SpaceTimeCoord<GeoCoord>>, JsValue> {
    if lats.len() != lons.len() || lats.len() != times.len() {
        return Err(coded_err(
            "lats, lons, and times must have the same length",
            "mismatched_arrays",
        ));
    }
    let mut out = Vec::with_capacity(lats.len());
    for i in 0..lats.len() {
        let geo = GeoCoord::try_new(lats[i] as Real, lons[i] as Real).map_err(kriging_err_to_js)?;
        out.push(SpaceTimeCoord::try_new(geo, times[i] as Real).map_err(kriging_err_to_js)?);
    }
    Ok(out)
}

/// Leave-one-out CV for space-time ordinary kriging on geographic coordinates.
#[wasm_bindgen(js_name = leaveOneOutSpaceTime)]
pub fn wasm_leave_one_out_spacetime(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    variogram: JsValue,
) -> Result<JsValue, JsValue> {
    if values.len() != lats.len() {
        return Err(coded_err(
            "values must have the same length as lats/lons/times",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let vg = spacetime_variogram_from_js(variogram)?;
    let residuals = crate::cv::leave_one_out_spacetime(GeoMetric, &coords, &values_real, vg)
        .map_err(kriging_err_to_js)?;
    cv_result_to_js(residuals)
}

/// K-fold CV for space-time ordinary kriging on geographic coordinates.
#[wasm_bindgen(js_name = kFoldSpaceTime)]
pub fn wasm_k_fold_spacetime(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    k: usize,
    variogram: JsValue,
) -> Result<JsValue, JsValue> {
    if values.len() != lats.len() {
        return Err(coded_err(
            "values must have the same length as lats/lons/times",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let vg = spacetime_variogram_from_js(variogram)?;
    let residuals = crate::cv::k_fold_spacetime(GeoMetric, &coords, &values_real, vg, k)
        .map_err(kriging_err_to_js)?;
    cv_result_to_js(residuals)
}

/// Leave-one-out CV for space-time simple kriging (known `mean`).
#[wasm_bindgen(js_name = leaveOneOutSpaceTimeSimple)]
pub fn wasm_leave_one_out_spacetime_simple(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    mean: f64,
    variogram: JsValue,
) -> Result<JsValue, JsValue> {
    if values.len() != lats.len() {
        return Err(coded_err(
            "values must have the same length as lats/lons/times",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let vg = spacetime_variogram_from_js(variogram)?;
    let residuals = crate::cv::leave_one_out_spacetime_simple(
        GeoMetric,
        &coords,
        &values_real,
        vg,
        mean as Real,
    )
    .map_err(kriging_err_to_js)?;
    cv_result_to_js(residuals)
}

/// K-fold CV for space-time simple kriging (known `mean`).
#[wasm_bindgen(js_name = kFoldSpaceTimeSimple)]
pub fn wasm_k_fold_spacetime_simple(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    mean: f64,
    k: usize,
    variogram: JsValue,
) -> Result<JsValue, JsValue> {
    if values.len() != lats.len() {
        return Err(coded_err(
            "values must have the same length as lats/lons/times",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let vg = spacetime_variogram_from_js(variogram)?;
    let residuals =
        crate::cv::k_fold_spacetime_simple(GeoMetric, &coords, &values_real, vg, mean as Real, k)
            .map_err(kriging_err_to_js)?;
    cv_result_to_js(residuals)
}

/// Leave-one-out CV for space-time universal kriging with a named polynomial trend.
#[wasm_bindgen(js_name = leaveOneOutSpaceTimeUniversal)]
pub fn wasm_leave_one_out_spacetime_universal(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    trend: &str,
    variogram: JsValue,
) -> Result<JsValue, JsValue> {
    if values.len() != lats.len() {
        return Err(coded_err(
            "values must have the same length as lats/lons/times",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let vg = spacetime_variogram_from_js(variogram)?;
    let trend = parse_universal_trend(trend)?;
    let residuals =
        crate::cv::leave_one_out_spacetime_universal(GeoMetric, &coords, &values_real, vg, trend)
            .map_err(kriging_err_to_js)?;
    cv_result_to_js(residuals)
}

/// K-fold CV for space-time universal kriging with a named polynomial trend.
#[wasm_bindgen(js_name = kFoldSpaceTimeUniversal)]
pub fn wasm_k_fold_spacetime_universal(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    values: &[f64],
    trend: &str,
    k: usize,
    variogram: JsValue,
) -> Result<JsValue, JsValue> {
    if values.len() != lats.len() {
        return Err(coded_err(
            "values must have the same length as lats/lons/times",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let values_real: Vec<Real> = values.iter().map(|v| *v as Real).collect();
    let vg = spacetime_variogram_from_js(variogram)?;
    let trend = parse_universal_trend(trend)?;
    let residuals =
        crate::cv::k_fold_spacetime_universal(GeoMetric, &coords, &values_real, vg, trend, k)
            .map_err(kriging_err_to_js)?;
    cv_result_to_js(residuals)
}

/// Leave-one-out CV for space-time binomial kriging. Returns dual-scale residuals.
#[wasm_bindgen(js_name = leaveOneOutSpaceTimeBinomial)]
pub fn wasm_leave_one_out_spacetime_binomial(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    successes: &[u32],
    trials: &[u32],
    variogram: JsValue,
    prior_alpha: Option<f64>,
    prior_beta: Option<f64>,
) -> Result<JsValue, JsValue> {
    if lats.len() != lons.len()
        || lats.len() != times.len()
        || lats.len() != successes.len()
        || lats.len() != trials.len()
    {
        return Err(coded_err(
            "lats, lons, times, successes, and trials must all have the same length",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
    let residuals = crate::cv::leave_one_out_spacetime_binomial(
        GeoMetric, &coords, successes, trials, vg, prior,
    )
    .map_err(kriging_err_to_js)?;
    binomial_cv_result_to_js(residuals)
}

/// K-fold CV for space-time binomial kriging. Returns dual-scale residuals.
#[wasm_bindgen(js_name = kFoldSpaceTimeBinomial)]
pub fn wasm_k_fold_spacetime_binomial(
    lats: &[f64],
    lons: &[f64],
    times: &[f64],
    successes: &[u32],
    trials: &[u32],
    k: usize,
    variogram: JsValue,
    prior_alpha: Option<f64>,
    prior_beta: Option<f64>,
) -> Result<JsValue, JsValue> {
    if lats.len() != lons.len()
        || lats.len() != times.len()
        || lats.len() != successes.len()
        || lats.len() != trials.len()
    {
        return Err(coded_err(
            "lats, lons, times, successes, and trials must all have the same length",
            "mismatched_arrays",
        ));
    }
    let coords = st_build_geo_coords(lats, lons, times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
    let residuals =
        crate::cv::k_fold_spacetime_binomial(GeoMetric, &coords, successes, trials, vg, prior, k)
            .map_err(kriging_err_to_js)?;
    binomial_cv_result_to_js(residuals)
}

// ---------------------------------------------------------------------------
// Space–time conditional simulation
// ---------------------------------------------------------------------------

/// Sequential Gaussian simulation over space-time ordinary kriging on geographic
/// coordinates. Returns a `Float64Array` of sampled values in input target order.
#[wasm_bindgen(js_name = conditionalSimulateSpaceTime)]
pub fn wasm_conditional_simulate_spacetime(
    conditioning_lats: &[f64],
    conditioning_lons: &[f64],
    conditioning_times: &[f64],
    conditioning_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    target_times: &[f64],
    variogram: JsValue,
    seed: u64,
    target_order: Option<Vec<u32>>,
) -> Result<JsValue, JsValue> {
    if conditioning_values.len() != conditioning_lats.len() {
        return Err(coded_err(
            "conditioningValues must match conditioning lats/lons/times length",
            "mismatched_arrays",
        ));
    }
    let cond_coords =
        st_build_geo_coords(conditioning_lats, conditioning_lons, conditioning_times)?;
    let cond_values: Vec<Real> = conditioning_values.iter().map(|v| *v as Real).collect();
    let targets = st_build_geo_coords(target_lats, target_lons, target_times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let options = parse_simulation_options(seed, target_order);
    let samples = crate::simulation::conditional_simulate_spacetime(
        GeoMetric,
        &cond_coords,
        &cond_values,
        &targets,
        vg,
        options,
    )
    .map_err(kriging_err_to_js)?;
    let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
    Ok(Float64Array::from(samples_f64.as_slice()).into())
}

/// SGS using space-time simple kriging with a known `mean`.
#[wasm_bindgen(js_name = conditionalSimulateSpaceTimeSimple)]
pub fn wasm_conditional_simulate_spacetime_simple(
    conditioning_lats: &[f64],
    conditioning_lons: &[f64],
    conditioning_times: &[f64],
    conditioning_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    target_times: &[f64],
    mean: f64,
    variogram: JsValue,
    seed: u64,
    target_order: Option<Vec<u32>>,
) -> Result<JsValue, JsValue> {
    if conditioning_values.len() != conditioning_lats.len() {
        return Err(coded_err(
            "conditioningValues must match conditioning lats/lons/times length",
            "mismatched_arrays",
        ));
    }
    let cond_coords =
        st_build_geo_coords(conditioning_lats, conditioning_lons, conditioning_times)?;
    let cond_values: Vec<Real> = conditioning_values.iter().map(|v| *v as Real).collect();
    let targets = st_build_geo_coords(target_lats, target_lons, target_times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let options = parse_simulation_options(seed, target_order);
    let samples = crate::simulation::conditional_simulate_spacetime_simple(
        GeoMetric,
        &cond_coords,
        &cond_values,
        &targets,
        vg,
        mean as Real,
        options,
    )
    .map_err(kriging_err_to_js)?;
    let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
    Ok(Float64Array::from(samples_f64.as_slice()).into())
}

/// SGS using space-time universal kriging with a named polynomial `trend`.
#[wasm_bindgen(js_name = conditionalSimulateSpaceTimeUniversal)]
pub fn wasm_conditional_simulate_spacetime_universal(
    conditioning_lats: &[f64],
    conditioning_lons: &[f64],
    conditioning_times: &[f64],
    conditioning_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    target_times: &[f64],
    trend: &str,
    variogram: JsValue,
    seed: u64,
    target_order: Option<Vec<u32>>,
) -> Result<JsValue, JsValue> {
    if conditioning_values.len() != conditioning_lats.len() {
        return Err(coded_err(
            "conditioningValues must match conditioning lats/lons/times length",
            "mismatched_arrays",
        ));
    }
    let cond_coords =
        st_build_geo_coords(conditioning_lats, conditioning_lons, conditioning_times)?;
    let cond_values: Vec<Real> = conditioning_values.iter().map(|v| *v as Real).collect();
    let targets = st_build_geo_coords(target_lats, target_lons, target_times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let trend = parse_universal_trend(trend)?;
    let options = parse_simulation_options(seed, target_order);
    let samples = crate::simulation::conditional_simulate_spacetime_universal(
        GeoMetric,
        &cond_coords,
        &cond_values,
        &targets,
        vg,
        trend,
        options,
    )
    .map_err(kriging_err_to_js)?;
    let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
    Ok(Float64Array::from(samples_f64.as_slice()).into())
}

/// SGS for space-time binomial kriging. Returns an object with `logitSamples` and
/// `prevalenceSamples` typed arrays in input target order.
#[wasm_bindgen(js_name = conditionalSimulateSpaceTimeBinomial)]
pub fn wasm_conditional_simulate_spacetime_binomial(
    conditioning_lats: &[f64],
    conditioning_lons: &[f64],
    conditioning_times: &[f64],
    successes: &[u32],
    trials: &[u32],
    target_lats: &[f64],
    target_lons: &[f64],
    target_times: &[f64],
    variogram: JsValue,
    prior_alpha: Option<f64>,
    prior_beta: Option<f64>,
    seed: u64,
    target_order: Option<Vec<u32>>,
) -> Result<JsValue, JsValue> {
    if conditioning_lats.len() != conditioning_lons.len()
        || conditioning_lats.len() != conditioning_times.len()
        || conditioning_lats.len() != successes.len()
        || conditioning_lats.len() != trials.len()
    {
        return Err(coded_err(
            "conditioning arrays (lats, lons, times, successes, trials) must have the same length",
            "mismatched_arrays",
        ));
    }
    let cond_coords =
        st_build_geo_coords(conditioning_lats, conditioning_lons, conditioning_times)?;
    let targets = st_build_geo_coords(target_lats, target_lons, target_times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
    let options = parse_simulation_options(seed, target_order);
    let result = crate::simulation::conditional_simulate_spacetime_binomial(
        GeoMetric,
        &cond_coords,
        successes,
        trials,
        &targets,
        vg,
        prior,
        options,
    )
    .map_err(kriging_err_to_js)?;
    binomial_simulation_to_js(result)
}

/// Multi-realization SGS over space-time ordinary kriging on geographic coordinates.
/// Returns a flat row-major `Float64Array` of length `nRealizations * nTargets`. Row `k`
/// matches a single-call `conditionalSimulateSpaceTime(seed = baseSeed + k, …)`.
#[wasm_bindgen(js_name = conditionalSimulateSpaceTimeMany)]
#[allow(clippy::too_many_arguments)]
pub fn wasm_conditional_simulate_spacetime_many(
    conditioning_lats: &[f64],
    conditioning_lons: &[f64],
    conditioning_times: &[f64],
    conditioning_values: &[f64],
    target_lats: &[f64],
    target_lons: &[f64],
    target_times: &[f64],
    variogram: JsValue,
    n_realizations: u32,
    base_seed: u64,
    target_order: Option<Vec<u32>>,
) -> Result<JsValue, JsValue> {
    if conditioning_values.len() != conditioning_lats.len() {
        return Err(coded_err(
            "conditioningValues must match conditioning lats/lons/times length",
            "mismatched_arrays",
        ));
    }
    let cond_coords =
        st_build_geo_coords(conditioning_lats, conditioning_lons, conditioning_times)?;
    let cond_values: Vec<Real> = conditioning_values.iter().map(|v| *v as Real).collect();
    let targets = st_build_geo_coords(target_lats, target_lons, target_times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let order = target_order.map(|v| v.into_iter().map(|x| x as usize).collect());
    let samples = crate::simulation::conditional_simulate_many_spacetime(
        GeoMetric,
        &cond_coords,
        &cond_values,
        &targets,
        vg,
        n_realizations as usize,
        base_seed,
        order,
    )
    .map_err(kriging_err_to_js)?;
    let samples_f64: Vec<f64> = samples.iter().map(|v| *v as f64).collect();
    Ok(Float64Array::from(samples_f64.as_slice()).into())
}

/// Multi-realization SGS for space-time binomial kriging. Returns an object with
/// `nRealizations`, `nTargets`, and flat row-major `logitSamples` / `prevalenceSamples`
/// `Float64Array`s. Row `k` matches a single-call
/// `conditionalSimulateSpaceTimeBinomial(seed = baseSeed + k, …)`.
#[wasm_bindgen(js_name = conditionalSimulateSpaceTimeManyBinomial)]
#[allow(clippy::too_many_arguments)]
pub fn wasm_conditional_simulate_spacetime_many_binomial(
    conditioning_lats: &[f64],
    conditioning_lons: &[f64],
    conditioning_times: &[f64],
    successes: &[u32],
    trials: &[u32],
    target_lats: &[f64],
    target_lons: &[f64],
    target_times: &[f64],
    variogram: JsValue,
    prior_alpha: Option<f64>,
    prior_beta: Option<f64>,
    n_realizations: u32,
    base_seed: u64,
    target_order: Option<Vec<u32>>,
) -> Result<JsValue, JsValue> {
    if conditioning_lats.len() != conditioning_lons.len()
        || conditioning_lats.len() != conditioning_times.len()
        || conditioning_lats.len() != successes.len()
        || conditioning_lats.len() != trials.len()
    {
        return Err(coded_err(
            "conditioning arrays (lats, lons, times, successes, trials) must have the same length",
            "mismatched_arrays",
        ));
    }
    let cond_coords =
        st_build_geo_coords(conditioning_lats, conditioning_lons, conditioning_times)?;
    let targets = st_build_geo_coords(target_lats, target_lons, target_times)?;
    let vg = spacetime_variogram_from_js(variogram)?;
    let prior = parse_binomial_prior(prior_alpha, prior_beta)?;
    let order = target_order.map(|v| v.into_iter().map(|x| x as usize).collect());
    let result = crate::simulation::conditional_simulate_many_spacetime_binomial(
        GeoMetric,
        &cond_coords,
        successes,
        trials,
        &targets,
        vg,
        prior,
        n_realizations as usize,
        base_seed,
        order,
    )
    .map_err(kriging_err_to_js)?;
    binomial_many_simulation_to_js(result)
}
