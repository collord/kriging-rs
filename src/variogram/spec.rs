//! A serializable variogram specification — the single object that carries every variogram
//! parameter across a (de)serialization boundary.
//!
//! Passing variograms as one [`VariogramSpec`] object (rather than positional
//! `type, nugget, sill, range, shape, shape2` scalars) means a new parameter is **additive**:
//! add one optional field here and one line in [`VariogramSpec::build`]. No intermediate
//! function signature or call site changes. The WASM/TS boundary and the nested-variogram and
//! kriging option structs all deserialize into this one type.

use serde::Deserialize;

use crate::Real;
use crate::error::KrigingError;

use super::models::{VariogramModel, VariogramType};

/// Parametric variogram description. On the wire the fields are `camelCase`
/// (`variogramType`, `nugget`, `sill`, `range`, optional `shape`, `shape2`); `type` is accepted
/// as an alias for `variogramType`. `shape` is the primary shape (Stable α, Matérn/CH ν, Power
/// exponent); `shape2` is the Confluent Hypergeometric tail-decay α.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariogramSpec {
    #[serde(alias = "type")]
    pub variogram_type: String,
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
    #[serde(default)]
    pub shape: Option<f64>,
    #[serde(default)]
    pub shape2: Option<f64>,
}

impl VariogramSpec {
    /// Convenience constructor (mainly for building a spec from already-separated parameters).
    pub fn new(
        variogram_type: impl Into<String>,
        nugget: f64,
        sill: f64,
        range: f64,
        shape: Option<f64>,
        shape2: Option<f64>,
    ) -> Self {
        Self {
            variogram_type: variogram_type.into(),
            nugget,
            sill,
            range,
            shape,
            shape2,
        }
    }

    /// Resolve the `variogramType` string to a [`VariogramType`], or `None` if unrecognized.
    pub fn resolve_type(&self) -> Option<VariogramType> {
        Some(match self.variogram_type.to_ascii_lowercase().as_str() {
            "spherical" => VariogramType::Spherical,
            "exponential" => VariogramType::Exponential,
            "gaussian" => VariogramType::Gaussian,
            "cubic" => VariogramType::Cubic,
            "stable" => VariogramType::Stable,
            "matern" => VariogramType::Matern,
            "power" => VariogramType::Power,
            "holeeffect" | "hole_effect" | "hole-effect" => VariogramType::HoleEffect,
            "confluenthypergeometric"
            | "confluent_hypergeometric"
            | "confluent-hypergeometric"
            | "ch" => VariogramType::ConfluentHypergeometric,
            _ => return None,
        })
    }

    /// Build a [`VariogramModel`] for an already-resolved type. This is the **single place**
    /// variogram parameters are interpreted: dispatch on how many shapes the family needs.
    pub fn build(&self, vt: VariogramType) -> Result<VariogramModel, KrigingError> {
        let (nugget, sill, range) = (self.nugget as Real, self.sill as Real, self.range as Real);
        match (vt, self.shape) {
            (VariogramType::ConfluentHypergeometric, Some(s)) => VariogramModel::new_with_shapes(
                nugget,
                sill,
                range,
                vt,
                s as Real,
                self.shape2.map(|a| a as Real),
            ),
            (VariogramType::Stable, Some(s))
            | (VariogramType::Matern, Some(s))
            | (VariogramType::Power, Some(s)) => {
                VariogramModel::new_with_shape(nugget, sill, range, vt, s as Real)
            }
            _ => VariogramModel::new(nugget, sill, range, vt),
        }
    }

    /// Resolve the type and build the [`VariogramModel`]. Errors with
    /// [`KrigingError::FittingError`] on an unrecognized type, or propagates the constructor's
    /// validation error.
    pub fn to_model(&self) -> Result<VariogramModel, KrigingError> {
        let vt = self.resolve_type().ok_or_else(|| {
            KrigingError::FittingError(format!("unknown variogram type: {}", self.variogram_type))
        })?;
        self.build(vt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn deserializes_camelcase_and_optional_shapes() {
        // Single-shape family, no shape2.
        let json = r#"{"variogramType":"matern","nugget":0.1,"sill":1.0,"range":10.0,"shape":1.5}"#;
        let spec: VariogramSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.variogram_type, "matern");
        assert_eq!(spec.shape, Some(1.5));
        assert_eq!(spec.shape2, None);
        let model = spec.to_model().unwrap();
        assert_eq!(model.variogram_type(), VariogramType::Matern);
        assert_relative_eq!(model.shape().unwrap(), 1.5, epsilon = 1e-6);

        // Two-shape family carries shape2.
        let ch = r#"{"variogramType":"confluenthypergeometric","nugget":0.0,"sill":2.0,"range":5.0,"shape":0.75,"shape2":2.0}"#;
        let spec: VariogramSpec = serde_json::from_str(ch).unwrap();
        let model = spec.to_model().unwrap();
        assert_relative_eq!(model.shape2().unwrap(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn accepts_type_alias_and_rejects_unknown() {
        let aliased = r#"{"type":"exponential","nugget":0.0,"sill":1.0,"range":1.0}"#;
        let spec: VariogramSpec = serde_json::from_str(aliased).unwrap();
        assert_eq!(spec.resolve_type(), Some(VariogramType::Exponential));

        let bad = r#"{"variogramType":"nope","nugget":0.0,"sill":1.0,"range":1.0}"#;
        let spec: VariogramSpec = serde_json::from_str(bad).unwrap();
        assert!(spec.resolve_type().is_none());
        assert!(spec.to_model().is_err());
    }
}
