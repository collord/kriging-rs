//! WASM bindings for the 3-D kriging + SGS surface (M13).
//!
//! Mirrors upstream's 2-D `wasm` module idiom: each Rust model is
//! wrapped in a `Wasm*3D` struct exposed to JS as a typed class with
//! a `fromArrays` static constructor (zero-copy, takes flat
//! Float64Arrays) and typed predict/simulate methods that return
//! Float64Arrays. Error returns are `Object { message, code }` per
//! the upstream pattern; helpers live in `super::*`.
//!
//! ## Anisotropy
//!
//! The 3-D path always takes an anisotropy spec. Pass zero
//! placeholders to disable: `(ang1, ang2, ang3, anis1, anis2) =
//! (0, 0, 0, 1, 1)` is GSLib's "no rotation, no stretch" identity.
//! Internally the conversion goes through
//! [`crate::interop::gslib_anisotropy::from_gslib`] so all GSLib
//! semantics live at the boundary, not inside the kriging math.

use js_sys::{Float64Array, Function, Object, Reflect};
use std::num::NonZeroUsize;
use wasm_bindgen::prelude::*;

use super::{coded_err, kriging_err_to_js, set_object_field, variogram_model_from_js};
use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::interop::gslib_anisotropy::{GslibAnisotropy, from_gslib};
use crate::interop::gslib_directional::GslibDirection;
use crate::kriging::diagnostics::Prediction3D;
use crate::kriging::ordinary_3d::{Neighborhood3D, OrdinaryKrigingModel3D};
use crate::kriging::simple_3d::SimpleKrigingModel3D;
use crate::kriging::universal_3d::{Trend3D, UniversalKrigingModel3D};
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::simulation_3d::{
    Grid3D, SgsError, SgsModel3D, SgsOutputSpace, gaussian_simulation_3d_stream_with,
};
use crate::variogram::directional_3d::{DirectionalConfig3D, compute_directional_variogram_3d};
use crate::variogram::empirical::{EmpiricalEstimator, PositiveReal};

// ---------- shared helpers ----------

/// Build a `Vec<Coord3D>` from parallel `xs/ys/zs` typed-array slices.
fn to_coords_3d(xs: &[f64], ys: &[f64], zs: &[f64]) -> Result<Vec<Coord3D>, JsValue> {
    if xs.len() != ys.len() || xs.len() != zs.len() {
        return Err(coded_err(
            "xs, ys, zs must have the same length",
            "mismatched_arrays",
        ));
    }
    Ok(xs
        .iter()
        .zip(ys)
        .zip(zs)
        .map(|((x, y), z)| Coord3D::new(*x as Real, *y as Real, *z as Real))
        .collect())
}

/// Build a `Vec<Real>` from a `&[f64]` slice (casts at the boundary).
fn to_real_vec(values: &[f64]) -> Vec<Real> {
    values.iter().map(|v| *v as Real).collect()
}

/// Build an `Anisotropy3D` from the five GSLib parameters. `(0, 0, 0,
/// 1, 1)` is the identity ("no rotation, no stretch"); see
/// `crate::interop::gslib_anisotropy` for the convention details.
fn parse_anisotropy(
    ang1: f64,
    ang2: f64,
    ang3: f64,
    anis1: f64,
    anis2: f64,
) -> Result<Anisotropy3D, JsValue> {
    from_gslib(GslibAnisotropy {
        ang1,
        ang2,
        ang3,
        anis1,
        anis2,
    })
    .map_err(kriging_err_to_js)
}

/// Apply a `Neighborhood3D` to a freshly-built model. Returns the same
/// model with the neighborhood attached if either parameter is set,
/// or unchanged if both are None.
fn maybe_with_neighborhood_ok(
    model: OrdinaryKrigingModel3D,
    max_radius: Option<f64>,
    max_neighbors: Option<usize>,
) -> Result<OrdinaryKrigingModel3D, JsValue> {
    match (max_radius, max_neighbors) {
        (None, None) => Ok(model),
        (r, k) => {
            if let Some(r) = r
                && (!r.is_finite() || r <= 0.0)
            {
                return Err(coded_err(
                    "maxRadius must be finite and positive",
                    "invalid_input",
                ));
            }
            let k_nz = match k {
                Some(0) => {
                    return Err(coded_err(
                        "maxNeighbors must be > 0 when provided",
                        "invalid_input",
                    ));
                }
                Some(k) => Some(NonZeroUsize::new(k).unwrap()),
                None => None,
            };
            Ok(model.with_neighborhood(Neighborhood3D {
                max_radius: r,
                max_neighbors: k_nz,
            }))
        }
    }
}

/// Split a `Vec<Prediction3D>` into parallel Vecs of value, variance,
/// condition_number, and used_nugget_inflation flags. Returns JS-side
/// arrays.
fn predictions_to_arrays(preds: Vec<Prediction3D>) -> JsValue {
    let n = preds.len();
    let mut values = Vec::with_capacity(n);
    let mut variances = Vec::with_capacity(n);
    let mut condition_numbers = Vec::with_capacity(n);
    let mut used_inflation = Vec::with_capacity(n);
    for p in preds {
        values.push(p.value as f64);
        variances.push(p.variance as f64);
        condition_numbers.push(p.condition_number);
        used_inflation.push(if p.used_nugget_inflation { 1.0 } else { 0.0 });
    }
    let obj = Object::new();
    let _ = set_object_field(
        &obj,
        "values",
        &Float64Array::from(values.as_slice()).into(),
    );
    let _ = set_object_field(
        &obj,
        "variances",
        &Float64Array::from(variances.as_slice()).into(),
    );
    let _ = set_object_field(
        &obj,
        "conditionNumbers",
        &Float64Array::from(condition_numbers.as_slice()).into(),
    );
    let _ = set_object_field(
        &obj,
        "usedNuggetInflation",
        &Float64Array::from(used_inflation.as_slice()).into(),
    );
    obj.into()
}

// ---------- Ordinary kriging 3D ----------

#[wasm_bindgen]
pub struct WasmOrdinaryKriging3D {
    inner: OrdinaryKrigingModel3D,
}

#[wasm_bindgen]
impl WasmOrdinaryKriging3D {
    /// Build an OK3D model from flat typed-array inputs. The
    /// anisotropy is supplied in GSLib parameter form
    /// `(ang1, ang2, ang3, anis1, anis2)`; pass `(0, 0, 0, 1, 1)` for
    /// the identity. The variogram is a spec object
    /// `{ variogramType, nugget, sill, range, shape?, shape2? }`.
    /// Optional `maxRadius` and `maxNeighbors` build an
    /// anisotropy-aware kd-tree neighborhood; passing neither uses
    /// all samples.
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        xs: &[f64],
        ys: &[f64],
        zs: &[f64],
        values: &[f64],
        ang1: f64,
        ang2: f64,
        ang3: f64,
        anis1: f64,
        anis2: f64,
        variogram: JsValue,
        max_radius: Option<f64>,
        max_neighbors: Option<usize>,
    ) -> Result<WasmOrdinaryKriging3D, JsValue> {
        if values.len() != xs.len() {
            return Err(coded_err(
                "values must have the same length as xs/ys/zs",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords_3d(xs, ys, zs)?;
        let dataset =
            PlanarDataset3D::new(coords, to_real_vec(values)).map_err(kriging_err_to_js)?;
        let anisotropy = parse_anisotropy(ang1, ang2, ang3, anis1, anis2)?;
        let variogram = variogram_model_from_js(variogram)?;
        let model = OrdinaryKrigingModel3D::new(dataset, anisotropy, variogram)
            .map_err(kriging_err_to_js)?;
        let model = maybe_with_neighborhood_ok(model, max_radius, max_neighbors)?;
        Ok(Self { inner: model })
    }

    /// Predict a single target. Returns `{ value, variance,
    /// conditionNumber, usedNuggetInflation }`.
    #[wasm_bindgen]
    pub fn predict(&self, x: f64, y: f64, z: f64) -> Result<JsValue, JsValue> {
        let target = Coord3D::new(x as Real, y as Real, z as Real);
        let p = self.inner.predict(target).map_err(kriging_err_to_js)?;
        let obj = Object::new();
        set_object_field(&obj, "value", &JsValue::from_f64(p.value as f64))?;
        set_object_field(&obj, "variance", &JsValue::from_f64(p.variance as f64))?;
        set_object_field(
            &obj,
            "conditionNumber",
            &JsValue::from_f64(p.condition_number),
        )?;
        set_object_field(
            &obj,
            "usedNuggetInflation",
            &JsValue::from_bool(p.used_nugget_inflation),
        )?;
        Ok(obj.into())
    }

    /// Predict a batch of targets. Returns `{ values, variances,
    /// conditionNumbers, usedNuggetInflation }` as parallel
    /// Float64Arrays.
    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(&self, xs: &[f64], ys: &[f64], zs: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords_3d(xs, ys, zs)?;
        let preds = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        Ok(predictions_to_arrays(preds))
    }
}

// ---------- Simple kriging 3D ----------

#[wasm_bindgen]
pub struct WasmSimpleKriging3D {
    inner: SimpleKrigingModel3D,
}

#[wasm_bindgen]
impl WasmSimpleKriging3D {
    /// Build an SK3D model from flat typed-array inputs. SK requires
    /// a known global `mean`. See [`WasmOrdinaryKriging3D::from_arrays`]
    /// for argument semantics.
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(
        xs: &[f64],
        ys: &[f64],
        zs: &[f64],
        values: &[f64],
        mean: f64,
        ang1: f64,
        ang2: f64,
        ang3: f64,
        anis1: f64,
        anis2: f64,
        variogram: JsValue,
    ) -> Result<WasmSimpleKriging3D, JsValue> {
        if values.len() != xs.len() {
            return Err(coded_err(
                "values must have the same length as xs/ys/zs",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords_3d(xs, ys, zs)?;
        let dataset =
            PlanarDataset3D::new(coords, to_real_vec(values)).map_err(kriging_err_to_js)?;
        let anisotropy = parse_anisotropy(ang1, ang2, ang3, anis1, anis2)?;
        let variogram = variogram_model_from_js(variogram)?;
        let model = SimpleKrigingModel3D::new(dataset, anisotropy, variogram, mean as Real)
            .map_err(kriging_err_to_js)?;
        Ok(Self { inner: model })
    }

    #[wasm_bindgen]
    pub fn predict(&self, x: f64, y: f64, z: f64) -> Result<JsValue, JsValue> {
        let target = Coord3D::new(x as Real, y as Real, z as Real);
        let p = self.inner.predict(target).map_err(kriging_err_to_js)?;
        let obj = Object::new();
        set_object_field(&obj, "value", &JsValue::from_f64(p.value as f64))?;
        set_object_field(&obj, "variance", &JsValue::from_f64(p.variance as f64))?;
        set_object_field(
            &obj,
            "conditionNumber",
            &JsValue::from_f64(p.condition_number),
        )?;
        set_object_field(
            &obj,
            "usedNuggetInflation",
            &JsValue::from_bool(p.used_nugget_inflation),
        )?;
        Ok(obj.into())
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(&self, xs: &[f64], ys: &[f64], zs: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords_3d(xs, ys, zs)?;
        let preds = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        Ok(predictions_to_arrays(preds))
    }
}

// ---------- Universal kriging 3D (linear trend) ----------

#[wasm_bindgen]
pub struct WasmUniversalKriging3D {
    inner: UniversalKrigingModel3D,
}

#[wasm_bindgen]
impl WasmUniversalKriging3D {
    /// Build a UK3D model with the linear trend basis `[1, x, y, z]`.
    /// Quadratic + arbitrary callback trends are deferred to v2; v1
    /// supports only linear.
    #[wasm_bindgen(js_name = fromArraysLinear)]
    pub fn from_arrays_linear(
        xs: &[f64],
        ys: &[f64],
        zs: &[f64],
        values: &[f64],
        ang1: f64,
        ang2: f64,
        ang3: f64,
        anis1: f64,
        anis2: f64,
        variogram: JsValue,
    ) -> Result<WasmUniversalKriging3D, JsValue> {
        if values.len() != xs.len() {
            return Err(coded_err(
                "values must have the same length as xs/ys/zs",
                "mismatched_arrays",
            ));
        }
        let coords = to_coords_3d(xs, ys, zs)?;
        let dataset =
            PlanarDataset3D::new(coords, to_real_vec(values)).map_err(kriging_err_to_js)?;
        let anisotropy = parse_anisotropy(ang1, ang2, ang3, anis1, anis2)?;
        let variogram = variogram_model_from_js(variogram)?;
        let model = UniversalKrigingModel3D::new(dataset, anisotropy, variogram, Trend3D::Linear)
            .map_err(kriging_err_to_js)?;
        Ok(Self { inner: model })
    }

    #[wasm_bindgen]
    pub fn predict(&self, x: f64, y: f64, z: f64) -> Result<JsValue, JsValue> {
        let target = Coord3D::new(x as Real, y as Real, z as Real);
        let p = self.inner.predict(target).map_err(kriging_err_to_js)?;
        let obj = Object::new();
        set_object_field(&obj, "value", &JsValue::from_f64(p.value as f64))?;
        set_object_field(&obj, "variance", &JsValue::from_f64(p.variance as f64))?;
        set_object_field(
            &obj,
            "conditionNumber",
            &JsValue::from_f64(p.condition_number),
        )?;
        set_object_field(
            &obj,
            "usedNuggetInflation",
            &JsValue::from_bool(p.used_nugget_inflation),
        )?;
        Ok(obj.into())
    }

    #[wasm_bindgen(js_name = predictBatch)]
    pub fn predict_batch(&self, xs: &[f64], ys: &[f64], zs: &[f64]) -> Result<JsValue, JsValue> {
        let coords = to_coords_3d(xs, ys, zs)?;
        let preds = self
            .inner
            .predict_batch(&coords)
            .map_err(kriging_err_to_js)?;
        Ok(predictions_to_arrays(preds))
    }
}

// ---------- Sequential Gaussian simulation ----------

/// Run 3-D SGS, calling `onRealization(idx, Float64Array)` once per
/// realization. The closure may return falsy/`undefined` to continue
/// or a truthy value to abort. The Float64Array passed to the
/// callback is a fresh JS array; callers may retain it.
///
/// Output is in data space (post NST back-transform). Pass
/// `scoreSpace = true` to get raw normal scores instead.
///
/// Grid layout: `nx × ny × nz` cells, row-major along x first, then
/// y, then z (so linear index `i + nx*j + nx*ny*k`).
///
/// **WASM is single-threaded** (no rayon); we use the serial
/// streaming path so the JS closure is invoked synchronously per
/// realization, in realization-index order.
#[wasm_bindgen(js_name = gaussianSimulation3D)]
pub fn wasm_gaussian_simulation_3d(
    sample_xs: &[f64],
    sample_ys: &[f64],
    sample_zs: &[f64],
    sample_values: &[f64],
    ang1: f64,
    ang2: f64,
    ang3: f64,
    anis1: f64,
    anis2: f64,
    variogram: JsValue,
    nx: usize,
    ny: usize,
    nz: usize,
    origin_x: f64,
    origin_y: f64,
    origin_z: f64,
    spacing_x: f64,
    spacing_y: f64,
    spacing_z: f64,
    seed: u64,
    n_realizations: usize,
    score_space: bool,
    on_realization: &Function,
) -> Result<(), JsValue> {
    if sample_values.len() != sample_xs.len() {
        return Err(coded_err(
            "sampleValues must have the same length as sampleXs/Ys/Zs",
            "mismatched_arrays",
        ));
    }
    let coords = to_coords_3d(sample_xs, sample_ys, sample_zs)?;
    let dataset =
        PlanarDataset3D::new(coords, to_real_vec(sample_values)).map_err(kriging_err_to_js)?;
    let anisotropy = parse_anisotropy(ang1, ang2, ang3, anis1, anis2)?;
    let variogram = variogram_model_from_js(variogram)?;
    let model = SgsModel3D::new(dataset, anisotropy, variogram).map_err(sgs_err_to_js)?;
    let grid = Grid3D::new(
        nx,
        ny,
        nz,
        Coord3D::new(origin_x as Real, origin_y as Real, origin_z as Real),
        Coord3D::new(spacing_x as Real, spacing_y as Real, spacing_z as Real),
    )
    .map_err(kriging_err_to_js)?;
    let output_space = if score_space {
        SgsOutputSpace::ScoreSpace
    } else {
        SgsOutputSpace::DataSpace
    };
    let this = JsValue::NULL;

    gaussian_simulation_3d_stream_with(
        &model,
        &grid,
        seed,
        n_realizations,
        output_space,
        |idx, gv| {
            // Copy `gv` into a fresh Float64Array because the engine
            // reuses its grid buffer between realizations; the JS
            // closure may retain the array.
            let gv_f64: Vec<f64> = gv.iter().map(|v| *v as f64).collect();
            let arr = Float64Array::from(gv_f64.as_slice());
            let result = on_realization.call2(&this, &JsValue::from_f64(idx as f64), &arr.into());
            match result {
                Ok(v) if v.is_truthy() => {
                    Err(SgsError::CallbackAborted("closure returned truthy".into()))
                }
                Ok(_) => Ok(()),
                Err(_) => Err(SgsError::CallbackAborted(
                    "closure threw an exception".into(),
                )),
            }
        },
    )
    .map_err(sgs_err_to_js)
}

// ---------- Directional variogram (3-D) ----------

/// Compute a 3-D experimental directional variogram using gamv-style
/// lag-centred binning and a cone+bandwidth direction filter.
///
/// Pass `atol = 90.0, dtol = 90.0, bandh = bandv = 1.0e10` for an
/// omnidirectional variogram (the engine sets the `omni` flag
/// automatically and double-counts pairs to match GSLib `gamv`).
///
/// Returns `Object { distances, semivariances, nPairs }` as parallel
/// `Float64Array`s aligned by lag index; empty lags are omitted (so
/// the array length is the count of non-empty bins, ≤ `nLags`).
#[wasm_bindgen(js_name = computeDirectionalVariogram3D)]
pub fn wasm_compute_directional_variogram_3d(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    values: &[f64],
    xlag: f64,
    xltol: f64,
    n_lags: usize,
    azm_deg: f64,
    atol_deg: f64,
    bandh: f64,
    dip_deg: f64,
    dtol_deg: f64,
    bandv: f64,
) -> Result<JsValue, JsValue> {
    if values.len() != xs.len() {
        return Err(coded_err(
            "values must have the same length as xs/ys/zs",
            "mismatched_arrays",
        ));
    }
    let coords = to_coords_3d(xs, ys, zs)?;
    let dataset = PlanarDataset3D::new(coords, to_real_vec(values)).map_err(kriging_err_to_js)?;
    let filter = GslibDirection {
        azm_deg,
        atol_deg,
        bandh,
        dip_deg,
        dtol_deg,
        bandv,
    }
    .to_filter()
    .map_err(kriging_err_to_js)?;
    let config = DirectionalConfig3D {
        xlag: PositiveReal::try_new(xlag as Real).map_err(kriging_err_to_js)?,
        xltol: PositiveReal::try_new(xltol as Real).map_err(kriging_err_to_js)?,
        n_lags,
        estimator: EmpiricalEstimator::Classical,
    };
    let ev =
        compute_directional_variogram_3d(&dataset, &filter, &config).map_err(kriging_err_to_js)?;

    let distances: Vec<f64> = ev.distances.iter().map(|d| *d as f64).collect();
    let semivariances: Vec<f64> = ev.semivariances.iter().map(|g| *g as f64).collect();
    let n_pairs: Vec<f64> = ev.n_pairs.iter().map(|n| *n as f64).collect();

    let obj = Object::new();
    set_object_field(
        &obj,
        "distances",
        &Float64Array::from(distances.as_slice()).into(),
    )?;
    set_object_field(
        &obj,
        "semivariances",
        &Float64Array::from(semivariances.as_slice()).into(),
    )?;
    set_object_field(
        &obj,
        "nPairs",
        &Float64Array::from(n_pairs.as_slice()).into(),
    )?;
    Ok(obj.into())
}

fn sgs_err_to_js(err: SgsError) -> JsValue {
    let (msg, code) = match &err {
        SgsError::CallbackAborted(s) => (s.clone(), "callback_aborted"),
        SgsError::InvalidInput(s) => (s.clone(), "invalid_input"),
        SgsError::InsufficientSamples => (err.to_string(), "insufficient_data"),
    };
    let obj = Object::new();
    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("message"),
        &JsValue::from_str(&msg),
    );
    let _ = Reflect::set(&obj, &JsValue::from_str("code"), &JsValue::from_str(code));
    obj.into()
}

// ---------- 3-D spherical joint fit ----------

/// Joint least-squares fit of a 3-D anisotropic spherical variogram
/// across three axis-aligned experimental variograms (major / minor /
/// vertical). Each input is `(distances, semivariances, nPairs)` as
/// parallel typed arrays. n_pairs weight the bins; bins with more
/// pairs contribute more to the loss.
///
/// Returns `Object { nugget, sill, rangeMajor, rangeMinor,
/// rangeVertical, residuals }`. The five parameters together
/// constitute a 3-D anisotropic spherical model that can drive
/// kriging via the existing Anisotropy3D + spherical VariogramModel
/// types.
#[wasm_bindgen(js_name = fitSpherical3DJoint)]
pub fn wasm_fit_spherical_3d_joint(
    major_distances: &[f64],
    major_semivariances: &[f64],
    major_n_pairs: &[f64],
    minor_distances: &[f64],
    minor_semivariances: &[f64],
    minor_n_pairs: &[f64],
    vertical_distances: &[f64],
    vertical_semivariances: &[f64],
    vertical_n_pairs: &[f64],
) -> Result<JsValue, JsValue> {
    let major =
        build_empirical_variogram("major", major_distances, major_semivariances, major_n_pairs)?;
    let minor =
        build_empirical_variogram("minor", minor_distances, minor_semivariances, minor_n_pairs)?;
    let vertical = build_empirical_variogram(
        "vertical",
        vertical_distances,
        vertical_semivariances,
        vertical_n_pairs,
    )?;
    let fit = crate::variogram::fitting::fit_spherical_3d_joint(&major, &minor, &vertical)
        .map_err(kriging_err_to_js)?;
    spherical_3d_fit_to_js(&fit)
}

/// Two-stage spherical fit. Fits nugget + sill + range_vertical on
/// the vertical experimental alone, then holds those and fits the
/// horizontal ranges -- avoids the joint fit's tendency to drive the
/// nugget toward zero when horizontal axes have noisy short-lag bins
/// (typical drillhole pattern; see Rust docs on
/// `fit_spherical_3d_two_stage`). Arguments and return shape match
/// `fitSpherical3DJoint`.
#[wasm_bindgen(js_name = fitSpherical3DTwoStage)]
pub fn wasm_fit_spherical_3d_two_stage(
    major_distances: &[f64],
    major_semivariances: &[f64],
    major_n_pairs: &[f64],
    minor_distances: &[f64],
    minor_semivariances: &[f64],
    minor_n_pairs: &[f64],
    vertical_distances: &[f64],
    vertical_semivariances: &[f64],
    vertical_n_pairs: &[f64],
    // data_variance: sample variance of the underlying data. Anchors
    // the stage-1 sill so the vertical's not-yet-plateaued curve
    // doesn't slide the fitter into a degenerate low-nugget /
    // high-sill solution. Pass 0 to opt out (treat sill as fully free).
    data_variance: f64,
) -> Result<JsValue, JsValue> {
    let major =
        build_empirical_variogram("major", major_distances, major_semivariances, major_n_pairs)?;
    let minor =
        build_empirical_variogram("minor", minor_distances, minor_semivariances, minor_n_pairs)?;
    let vertical = build_empirical_variogram(
        "vertical",
        vertical_distances,
        vertical_semivariances,
        vertical_n_pairs,
    )?;
    let fit = crate::variogram::fitting::fit_spherical_3d_two_stage(
        &major,
        &minor,
        &vertical,
        data_variance as Real,
    )
    .map_err(kriging_err_to_js)?;
    spherical_3d_fit_to_js(&fit)
}

/// Refit spherical model with nugget held at a user-supplied value.
/// Fits sill plus three ranges. Used after a two-stage fit when the
/// user has overridden the auto-derived nugget by reading it off the
/// vertical's short-lag intercept manually.
#[wasm_bindgen(js_name = fitSpherical3DFixedNugget)]
pub fn wasm_fit_spherical_3d_fixed_nugget(
    major_distances: &[f64],
    major_semivariances: &[f64],
    major_n_pairs: &[f64],
    minor_distances: &[f64],
    minor_semivariances: &[f64],
    minor_n_pairs: &[f64],
    vertical_distances: &[f64],
    vertical_semivariances: &[f64],
    vertical_n_pairs: &[f64],
    nugget: f64,
) -> Result<JsValue, JsValue> {
    let major =
        build_empirical_variogram("major", major_distances, major_semivariances, major_n_pairs)?;
    let minor =
        build_empirical_variogram("minor", minor_distances, minor_semivariances, minor_n_pairs)?;
    let vertical = build_empirical_variogram(
        "vertical",
        vertical_distances,
        vertical_semivariances,
        vertical_n_pairs,
    )?;
    let fit = crate::variogram::fitting::fit_spherical_3d_with_fixed_nugget(
        &major,
        &minor,
        &vertical,
        nugget as Real,
    )
    .map_err(kriging_err_to_js)?;
    spherical_3d_fit_to_js(&fit)
}

/// Shared helper: convert JS-side per-axis arrays into an
/// `EmpiricalVariogram`. Centralised so the three fit entry points
/// stay symmetric.
fn build_empirical_variogram(
    name: &str,
    distances: &[f64],
    semivariances: &[f64],
    n_pairs: &[f64],
) -> Result<crate::variogram::empirical::EmpiricalVariogram, JsValue> {
    if distances.len() != semivariances.len() || distances.len() != n_pairs.len() {
        return Err(coded_err(
            &format!(
                "{name} variogram arrays must have matching lengths (got {} / {} / {})",
                distances.len(),
                semivariances.len(),
                n_pairs.len(),
            ),
            "mismatched_arrays",
        ));
    }
    Ok(crate::variogram::empirical::EmpiricalVariogram {
        distances: distances.iter().map(|d| *d as Real).collect(),
        semivariances: semivariances.iter().map(|g| *g as Real).collect(),
        n_pairs: n_pairs.iter().map(|n| n.max(0.0) as usize).collect(),
    })
}

/// Shared helper: convert a Rust `Spherical3DJointFit` into the JS
/// object shape exposed by all three fit entry points.
fn spherical_3d_fit_to_js(
    fit: &crate::variogram::fitting::Spherical3DJointFit,
) -> Result<JsValue, JsValue> {
    let obj = Object::new();
    set_object_field(&obj, "nugget", &JsValue::from_f64(fit.nugget as f64))?;
    set_object_field(&obj, "sill", &JsValue::from_f64(fit.sill as f64))?;
    set_object_field(
        &obj,
        "rangeMajor",
        &JsValue::from_f64(fit.range_major as f64),
    )?;
    set_object_field(
        &obj,
        "rangeMinor",
        &JsValue::from_f64(fit.range_minor as f64),
    )?;
    set_object_field(
        &obj,
        "rangeVertical",
        &JsValue::from_f64(fit.range_vertical as f64),
    )?;
    set_object_field(&obj, "residuals", &JsValue::from_f64(fit.residuals as f64))?;
    Ok(obj.into())
}

/// Fit a 1-D spherical model to a precomputed empirical variogram.
/// Used by the anisotropy search to score candidate orientations by
/// their fitted range -- more robust to single-bin noise spikes than
/// a threshold-crossing heuristic. Cheaper than the joint 3-D fit
/// because there are only 3 parameters and one axis worth of data.
///
/// Returns `{ nugget, sill, range, residuals }` (no "rangeMajor" etc.
/// since this is 1-D).
#[wasm_bindgen(js_name = fitSpherical1D)]
pub fn wasm_fit_spherical_1d(
    distances: &[f64],
    semivariances: &[f64],
    n_pairs: &[f64],
) -> Result<JsValue, JsValue> {
    let empirical = build_empirical_variogram("axis", distances, semivariances, n_pairs)?;
    let fit = crate::variogram::fitting::fit_variogram(
        &empirical,
        crate::variogram::models::VariogramType::Spherical,
    )
    .map_err(kriging_err_to_js)?;
    let (nugget, sill, range) = fit.model.params();
    let obj = Object::new();
    set_object_field(&obj, "nugget", &JsValue::from_f64(nugget as f64))?;
    set_object_field(&obj, "sill", &JsValue::from_f64(sill as f64))?;
    set_object_field(&obj, "range", &JsValue::from_f64(range as f64))?;
    set_object_field(&obj, "residuals", &JsValue::from_f64(fit.residuals as f64))?;
    Ok(obj.into())
}
