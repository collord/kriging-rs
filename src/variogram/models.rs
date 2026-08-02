use crate::Real;
use crate::error::KrigingError;
use std::sync::OnceLock;

/// Matérn semivariance: γ(h) = nugget + partial_sill * (1 - (2^(1-ν)/Γ(ν)) * x^ν * K_ν(x)) with x = h√(2ν)/range.
//
// `puruspe` only exposes `f64` Bessel/gamma functions, so we always promote the inputs to
// `f64` and demote the final result back to `Real`. Under the `f64` Cargo feature the
// promotion is a no-op cast that clippy flags as unnecessary; the cast is required in the
// default `f32` build.
#[allow(clippy::unnecessary_cast)]
fn matern_semivariance(d: Real, nugget: Real, partial_sill: Real, range: Real, nu: Real) -> Real {
    if d <= 0.0 {
        return nugget;
    }
    let nu_f64 = nu as f64;
    let x_f64 = (d as f64) * (2.0 * nu_f64).sqrt() / (range as f64);
    if x_f64 <= 0.0 {
        return nugget;
    }
    let (_i_nu, k_nu) = puruspe::bessel::Inu_Knu(nu_f64, x_f64);
    let gamma_nu = puruspe::gamma::gamma(nu_f64);
    let factor = (2.0_f64).powf(1.0 - nu_f64) / gamma_nu * x_f64.powf(nu_f64) * k_nu;
    let correlation = factor.clamp(0.0, 1.0);
    nugget + partial_sill * (1.0 - (correlation as Real))
}

/// Double-exponential ("exp-sinh") quadrature abscissae/weight-factors for `∫₀^∞`, cached.
///
/// Each entry is `(t_k, φ_k)` where `t_k = exp((π/2)·sinh(k·step))` is the node in `t`-space
/// and `φ_k = t_k · (π/2)·cosh(k·step)` is `dt/dx`. The map makes both tails decay
/// doubly-exponentially, so a single fixed rule is accurate for the whole practical range of
/// `z`. Nodes whose `t_k` is not finite are dropped (their true contribution underflows to 0).
fn exp_sinh_nodes() -> &'static Vec<(f64, f64)> {
    static NODES: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    NODES.get_or_init(|| {
        use std::f64::consts::FRAC_PI_2;
        let step = 1.0 / 32.0;
        let half_n = 128_i32; // x ∈ [−4, 4]
        let mut nodes = Vec::with_capacity((2 * half_n + 1) as usize);
        for k in -half_n..=half_n {
            let x = f64::from(k) * step;
            let t = (FRAC_PI_2 * x.sinh()).exp();
            let phi = t * FRAC_PI_2 * x.cosh();
            if t.is_finite() && phi.is_finite() {
                nodes.push((t, phi));
            }
        }
        nodes
    })
}

/// Natural log of the confluent hypergeometric integral
/// `∫₀^∞ e^{−z t} · t^{a−1} · (1 + t)^{b−a−1} dt` for `z > 0`, via double-exponential
/// quadrature (see [`exp_sinh_nodes`]) with a log-sum-exp accumulation.
///
/// Working in log space keeps the CH correlation stable for large `α`, where the integrand's
/// per-node magnitude and the `Γ` prefactor individually overflow/underflow `f64` even though
/// their combination is `O(1)`. Returns `-∞` when every node underflows.
fn conf_hypergeom_ln_integral(a: f64, b: f64, z: f64) -> f64 {
    let step: f64 = 1.0 / 32.0;
    let exp1 = a - 1.0;
    let exp2 = b - a - 1.0;
    let mut max_log = f64::NEG_INFINITY;
    let mut logs: Vec<f64> = Vec::with_capacity(exp_sinh_nodes().len());
    for &(t, phi) in exp_sinh_nodes() {
        let term_log = -z * t + exp1 * t.ln() + exp2 * (1.0 + t).ln() + phi.ln();
        if term_log.is_finite() {
            if term_log > max_log {
                max_log = term_log;
            }
            logs.push(term_log);
        }
    }
    if !max_log.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = logs.iter().map(|l| (l - max_log).exp()).sum();
    max_log + sum.ln() + step.ln()
}

/// Confluent hypergeometric function of the second kind `U(a, b, z)` (Tricomi) for `a > 0`,
/// `z > 0`:
///
/// `U(a, b, z) = 1/Γ(a) · ∫₀^∞ e^{−z t} · t^{a−1} · (1 + t)^{b−a−1} dt`.
///
/// Both preconditions hold for every CH covariance evaluation (`a = α > 0`, and
/// `z = (r/β)²/2 > 0` whenever `r > 0`). For very large `a` the result underflows to `0`;
/// the CH correlation avoids that by staying in log space (see
/// [`confluent_hypergeometric_semivariance`]).
///
/// Only used by tests to check the quadrature against closed-form values; the CH covariance
/// path uses [`conf_hypergeom_ln_integral`] directly.
#[cfg(test)]
fn conf_hypergeom_u(a: f64, b: f64, z: f64) -> f64 {
    (conf_hypergeom_ln_integral(a, b, z) - puruspe::gamma::ln_gamma(a)).exp()
}

/// Confluent hypergeometric (CH) semivariance:
/// `γ(r) = nugget + partial_sill · (1 − ρ(r))` with correlation
/// `ρ(r) = [Γ(ν+α)/Γ(ν)] · U(α, 1−ν, (r/β)²/2)`, smoothness `ν > 0`, tail decay `α > 0`.
///
/// `ρ(0) = 1`, and as `α → ∞` the CH family converges to Matérn; unlike Matérn it has
/// polynomial (long-range) tails. Like [`matern_semivariance`], inputs are promoted to
/// `f64` because `puruspe` is `f64`-only, and the correlation is clamped to `[0, 1]`.
#[allow(clippy::unnecessary_cast)]
fn confluent_hypergeometric_semivariance(
    d: Real,
    nugget: Real,
    partial_sill: Real,
    range: Real,
    nu: Real,
    alpha: Real,
) -> Real {
    if d <= 0.0 {
        return nugget;
    }
    let nu_f64 = nu as f64;
    let alpha_f64 = alpha as f64;
    let ratio = (d as f64) / (range as f64);
    let z = 0.5 * ratio * ratio;
    if z <= 0.0 {
        return nugget;
    }
    // ρ(r) = [Γ(ν+α)/Γ(ν)] · U(α, 1−ν, z), computed as
    // exp(lnΓ(ν+α) − lnΓ(ν) + ln∫ − lnΓ(α)) so large α stays representable.
    let ln_correlation = puruspe::gamma::ln_gamma(nu_f64 + alpha_f64)
        - puruspe::gamma::ln_gamma(nu_f64)
        + conf_hypergeom_ln_integral(alpha_f64, 1.0 - nu_f64, z)
        - puruspe::gamma::ln_gamma(alpha_f64);
    let correlation = ln_correlation.exp().clamp(0.0, 1.0);
    nugget + partial_sill * (1.0 - (correlation as Real))
}

/// Parametric variogram family used to construct a [`VariogramModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariogramType {
    Spherical,
    Exponential,
    Gaussian,
    Cubic,
    /// Power-law family; requires shape parameter `alpha` in (0, 2].
    Stable,
    /// Matérn family; requires smoothness parameter `nu` > 0.
    Matern,
    /// Unbounded power-law model `γ(h) = nugget + slope · h^exponent`, `exponent ∈ (0, 2)`.
    /// Has no sill; use with care in kriging (ordinary kriging systems remain solvable).
    Power,
    /// Damped hole-effect model `γ(h) = nugget + (sill − nugget)·(1 − sin(π·h/range)/(π·h/range))`.
    /// Useful for pseudo-periodic fields.
    HoleEffect,
    /// Confluent hypergeometric family; requires smoothness `nu > 0` and tail-decay `alpha > 0`.
    /// Generalizes Matérn (recovered as `alpha → ∞`) but with polynomial long-range tails.
    ConfluentHypergeometric,
}

/// Parametric variogram (nugget, sill, range, and optional shape).
///
/// Build with [`VariogramModel::new`] from a [`VariogramType`]; use with
/// [`OrdinaryKrigingModel::new`](crate::OrdinaryKrigingModel::new) or
/// [`BinomialKrigingModel::new`](crate::BinomialKrigingModel::new).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariogramModel {
    Spherical {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Exponential {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Gaussian {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Cubic {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    Stable {
        nugget: Real,
        sill: Real,
        range: Real,
        alpha: Real,
    },
    Matern {
        nugget: Real,
        sill: Real,
        range: Real,
        nu: Real,
    },
    /// Unbounded power-law model `γ(h) = nugget + slope·h^exponent`, `exponent ∈ (0, 2)`.
    /// `sill` and `range` fields carry `slope` and `exponent` respectively so that the rest
    /// of the library can treat `VariogramModel` uniformly via [`params`](Self::params).
    ///
    /// Note: for the power model, `sill` is not a true plateau — it is reused as the slope
    /// coefficient. Use [`new_power`](Self::new_power) for a named constructor.
    Power {
        nugget: Real,
        slope: Real,
        exponent: Real,
    },
    /// Hole-effect model with a pseudo-periodic dip; bounded above by `sill`.
    HoleEffect {
        nugget: Real,
        sill: Real,
        range: Real,
    },
    /// Confluent hypergeometric model with smoothness `nu` and tail-decay `alpha`; bounded
    /// above by `sill`. Build with [`new_with_shapes`](Self::new_with_shapes) supplying both
    /// shape parameters, or [`new`](Self::new) for defaults (`nu = 0.5`, `alpha = 1.0`).
    ConfluentHypergeometric {
        nugget: Real,
        sill: Real,
        range: Real,
        nu: Real,
        alpha: Real,
    },
}

impl VariogramModel {
    /// Constructs a variogram model with validated parameters: `nugget >= 0`, `sill > nugget`, `range > 0`, all finite.
    pub fn new(
        nugget: Real,
        sill: Real,
        range: Real,
        model_type: VariogramType,
    ) -> Result<Self, KrigingError> {
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(KrigingError::FittingError(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        if !sill.is_finite() || sill <= nugget {
            return Err(KrigingError::FittingError(
                "sill must be finite and greater than nugget".to_string(),
            ));
        }
        if !range.is_finite() || range <= 0.0 {
            return Err(KrigingError::FittingError(
                "range must be finite and positive".to_string(),
            ));
        }
        Ok(match model_type {
            VariogramType::Spherical => VariogramModel::Spherical {
                nugget,
                sill,
                range,
            },
            VariogramType::Exponential => VariogramModel::Exponential {
                nugget,
                sill,
                range,
            },
            VariogramType::Gaussian => VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            },
            VariogramType::Cubic => VariogramModel::Cubic {
                nugget,
                sill,
                range,
            },
            VariogramType::Stable => VariogramModel::Stable {
                nugget,
                sill,
                range,
                alpha: 1.0,
            },
            VariogramType::Matern => VariogramModel::Matern {
                nugget,
                sill,
                range,
                nu: 0.5,
            },
            VariogramType::Power => {
                // Default to exponent 1 (linear in distance); treat `sill` as slope.
                VariogramModel::Power {
                    nugget,
                    slope: sill - nugget,
                    exponent: 1.0,
                }
            }
            VariogramType::HoleEffect => VariogramModel::HoleEffect {
                nugget,
                sill,
                range,
            },
            VariogramType::ConfluentHypergeometric => VariogramModel::ConfluentHypergeometric {
                nugget,
                sill,
                range,
                nu: 0.5,
                alpha: 1.0,
            },
        })
    }

    /// Construct a power-law variogram `γ(h) = nugget + slope·h^exponent`. `slope > 0`,
    /// `exponent ∈ (0, 2)`, all values finite.
    pub fn new_power(nugget: Real, slope: Real, exponent: Real) -> Result<Self, KrigingError> {
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(KrigingError::FittingError(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        if !slope.is_finite() || slope <= 0.0 {
            return Err(KrigingError::FittingError(
                "power slope must be finite and positive".to_string(),
            ));
        }
        if !exponent.is_finite() || exponent <= 0.0 || exponent >= 2.0 {
            return Err(KrigingError::FittingError(
                "power exponent must be in (0, 2)".to_string(),
            ));
        }
        Ok(VariogramModel::Power {
            nugget,
            slope,
            exponent,
        })
    }

    /// Constructs a variogram model with an explicit shape parameter for Stable (alpha) or Matérn (nu).
    /// For other model types, `shape` is ignored. Stable: alpha in (0, 2]. Matérn: nu > 0.
    ///
    /// For two-shape families (Confluent Hypergeometric) use
    /// [`new_with_shapes`](Self::new_with_shapes); calling this with a CH `model_type` uses the
    /// supplied `shape` as `nu` and the default `alpha = 1.0`.
    pub fn new_with_shape(
        nugget: Real,
        sill: Real,
        range: Real,
        model_type: VariogramType,
        shape: Real,
    ) -> Result<Self, KrigingError> {
        Self::new_with_shapes(nugget, sill, range, model_type, shape, None)
    }

    /// Constructs a variogram model with up to two explicit shape parameters.
    ///
    /// `shape1` is the primary shape (Stable: alpha ∈ (0, 2]; Matérn: nu > 0; Power: exponent
    /// ∈ (0, 2); Confluent Hypergeometric: smoothness nu > 0). `shape2` is only consulted by
    /// two-shape families — Confluent Hypergeometric uses it as the tail-decay alpha > 0
    /// (defaulting to `1.0` when `None`). For all other model types `shape1`/`shape2` are
    /// ignored, matching [`new`](Self::new).
    pub fn new_with_shapes(
        nugget: Real,
        sill: Real,
        range: Real,
        model_type: VariogramType,
        shape1: Real,
        shape2: Option<Real>,
    ) -> Result<Self, KrigingError> {
        if !nugget.is_finite() || nugget < 0.0 {
            return Err(KrigingError::FittingError(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        if !sill.is_finite() || sill <= nugget {
            return Err(KrigingError::FittingError(
                "sill must be finite and greater than nugget".to_string(),
            ));
        }
        if !range.is_finite() || range <= 0.0 {
            return Err(KrigingError::FittingError(
                "range must be finite and positive".to_string(),
            ));
        }
        Ok(match model_type {
            VariogramType::Spherical => VariogramModel::Spherical {
                nugget,
                sill,
                range,
            },
            VariogramType::Exponential => VariogramModel::Exponential {
                nugget,
                sill,
                range,
            },
            VariogramType::Gaussian => VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            },
            VariogramType::Cubic => VariogramModel::Cubic {
                nugget,
                sill,
                range,
            },
            VariogramType::Stable => {
                if !shape1.is_finite() || shape1 <= 0.0 || shape1 > 2.0 {
                    return Err(KrigingError::FittingError(
                        "Stable shape (alpha) must be in (0, 2]".to_string(),
                    ));
                }
                VariogramModel::Stable {
                    nugget,
                    sill,
                    range,
                    alpha: shape1,
                }
            }
            VariogramType::Matern => {
                if !shape1.is_finite() || shape1 <= 0.0 {
                    return Err(KrigingError::FittingError(
                        "Matérn shape (nu) must be positive".to_string(),
                    ));
                }
                VariogramModel::Matern {
                    nugget,
                    sill,
                    range,
                    nu: shape1,
                }
            }
            VariogramType::Power => {
                // Reuse `shape1` as the exponent; `sill` is reinterpreted as slope (after
                // subtracting the nugget), matching the default constructor's convention.
                if !shape1.is_finite() || shape1 <= 0.0 || shape1 >= 2.0 {
                    return Err(KrigingError::FittingError(
                        "power exponent must be in (0, 2)".to_string(),
                    ));
                }
                VariogramModel::Power {
                    nugget,
                    slope: sill - nugget,
                    exponent: shape1,
                }
            }
            VariogramType::HoleEffect => VariogramModel::HoleEffect {
                nugget,
                sill,
                range,
            },
            VariogramType::ConfluentHypergeometric => {
                if !shape1.is_finite() || shape1 <= 0.0 {
                    return Err(KrigingError::FittingError(
                        "Confluent Hypergeometric smoothness (nu) must be positive".to_string(),
                    ));
                }
                let alpha = shape2.unwrap_or(1.0);
                if !alpha.is_finite() || alpha <= 0.0 {
                    return Err(KrigingError::FittingError(
                        "Confluent Hypergeometric tail decay (alpha) must be positive".to_string(),
                    ));
                }
                VariogramModel::ConfluentHypergeometric {
                    nugget,
                    sill,
                    range,
                    nu: shape1,
                    alpha,
                }
            }
        })
    }

    pub fn variogram_type(&self) -> VariogramType {
        match self {
            Self::Spherical { .. } => VariogramType::Spherical,
            Self::Exponential { .. } => VariogramType::Exponential,
            Self::Gaussian { .. } => VariogramType::Gaussian,
            Self::Cubic { .. } => VariogramType::Cubic,
            Self::Stable { .. } => VariogramType::Stable,
            Self::Matern { .. } => VariogramType::Matern,
            Self::Power { .. } => VariogramType::Power,
            Self::HoleEffect { .. } => VariogramType::HoleEffect,
            Self::ConfluentHypergeometric { .. } => VariogramType::ConfluentHypergeometric,
        }
    }

    pub fn params(&self) -> (Real, Real, Real) {
        match self {
            Self::Spherical {
                nugget,
                sill,
                range,
            }
            | Self::Exponential {
                nugget,
                sill,
                range,
            }
            | Self::Gaussian {
                nugget,
                sill,
                range,
            }
            | Self::Cubic {
                nugget,
                sill,
                range,
            }
            | Self::Stable {
                nugget,
                sill,
                range,
                ..
            }
            | Self::Matern {
                nugget,
                sill,
                range,
                ..
            }
            | Self::HoleEffect {
                nugget,
                sill,
                range,
            }
            | Self::ConfluentHypergeometric {
                nugget,
                sill,
                range,
                ..
            } => (*nugget, *sill, *range),
            // For the power model, `sill` carries slope and `range` carries exponent. This
            // keeps `params()` a simple getter; the actual semivariance computation branches
            // on the enum variant.
            Self::Power {
                nugget,
                slope,
                exponent,
            } => (*nugget, *slope + *nugget, *exponent),
        }
    }

    /// Primary shape parameter: Stable (alpha), Matérn (nu), Power (exponent), or Confluent
    /// Hypergeometric (nu). Returns `None` for 3-parameter bounded models.
    pub fn shape(&self) -> Option<Real> {
        match self {
            Self::Stable { alpha, .. } => Some(*alpha),
            Self::Matern { nu, .. } => Some(*nu),
            Self::Power { exponent, .. } => Some(*exponent),
            Self::ConfluentHypergeometric { nu, .. } => Some(*nu),
            _ => None,
        }
    }

    /// Secondary shape parameter, present only for two-shape families. For Confluent
    /// Hypergeometric this is the tail-decay `alpha`; all other models return `None`.
    pub fn shape2(&self) -> Option<Real> {
        match self {
            Self::ConfluentHypergeometric { alpha, .. } => Some(*alpha),
            _ => None,
        }
    }

    /// Semivariance at `distance`. Assumes `distance >= 0` (e.g. from haversine); clamps negative input to 0.
    pub fn semivariance(&self, distance: Real) -> Real {
        let d = distance.max(0.0);
        let (nugget, sill, range) = self.params();
        let partial_sill = sill - nugget;
        let r = range.max(Real::EPSILON);

        match self {
            Self::Spherical { .. } => {
                if d >= range {
                    sill
                } else {
                    let x = d / r;
                    nugget + partial_sill * (1.5 * x - 0.5 * x.powi(3))
                }
            }
            Self::Exponential { .. } => nugget + partial_sill * (1.0 - (-3.0 * d / r).exp()),
            Self::Gaussian { .. } => {
                nugget + partial_sill * (1.0 - (-3.0 * (d * d) / (r * r)).exp())
            }
            Self::Cubic { .. } => {
                if d >= range {
                    sill
                } else {
                    let x = d / r;
                    let poly = 7.0 * x * x - 8.5 * x.powi(3) + 3.5 * x.powi(5) - 0.5 * x.powi(7);
                    nugget + partial_sill * poly
                }
            }
            Self::Stable { alpha, .. } => {
                let x = (d / r).powf(*alpha);
                nugget + partial_sill * (1.0 - (-x).exp())
            }
            Self::Matern { nu, .. } => matern_semivariance(d, nugget, partial_sill, r, *nu),
            Self::Power {
                slope, exponent, ..
            } => nugget + slope * d.powf(*exponent),
            Self::HoleEffect { .. } => {
                if d <= 0.0 {
                    nugget
                } else {
                    // Damped hole-effect: γ(h) = nugget + partial_sill · (1 − sinc(π·h/r))
                    // with sinc(x) = sin(x)/x. Bounded above by sill.
                    let x = std::f64::consts::PI as Real * d / r;
                    let sinc = x.sin() / x;
                    nugget + partial_sill * (1.0 - sinc)
                }
            }
            Self::ConfluentHypergeometric { nu, alpha, .. } => {
                confluent_hypergeometric_semivariance(d, nugget, partial_sill, r, *nu, *alpha)
            }
        }
    }

    /// Covariance `C(h) = C(0) − γ(h)`. For bounded models `C(0) = sill`; for the unbounded
    /// [power model](VariogramModel::Power), `C(0)` is not defined, so this method returns
    /// `0` there and kriging should use the [`semivariance`](Self::semivariance) form.
    pub fn covariance(&self, distance: Real) -> Real {
        if matches!(self, Self::Power { .. }) {
            // Ordinary kriging of an intrinsic (unbounded) model is usually formulated via
            // semivariances. We return a dummy value; callers that explicitly support Power
            // should branch on the type.
            return 0.0;
        }
        let (_, sill, _) = self.params();
        sill - self.semivariance(distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spherical_hits_sill_at_range() {
        let model = VariogramModel::new(0.1, 1.0, 10.0, VariogramType::Spherical).unwrap();
        assert_relative_eq!(model.semivariance(10.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(model.semivariance(20.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn exponential_and_gaussian_start_at_nugget() {
        let exp = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Exponential).unwrap();
        let gauss = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Gaussian).unwrap();
        assert_relative_eq!(exp.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(gauss.semivariance(0.0), 0.2, epsilon = 1e-6);
    }

    #[test]
    fn covariance_complements_semivariance() {
        let model = VariogramModel::new(0.1, 1.0, 5.0, VariogramType::Exponential).unwrap();
        let d = 2.2;
        assert_relative_eq!(
            model.covariance(d) + model.semivariance(d),
            1.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn cubic_hits_sill_at_range() {
        let model = VariogramModel::new(0.1, 1.0, 10.0, VariogramType::Cubic).unwrap();
        assert_relative_eq!(model.semivariance(10.0), 1.0, epsilon = 1e-5);
        assert_relative_eq!(model.semivariance(20.0), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn cubic_stable_matern_start_at_nugget() {
        let cubic = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Cubic).unwrap();
        let stable = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Stable).unwrap();
        let matern = VariogramModel::new(0.2, 1.2, 15.0, VariogramType::Matern).unwrap();
        assert_relative_eq!(cubic.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(stable.semivariance(0.0), 0.2, epsilon = 1e-6);
        assert_relative_eq!(matern.semivariance(0.0), 0.2, epsilon = 1e-6);
    }

    #[test]
    fn stable_with_alpha_one_increases_to_sill() {
        let stable =
            VariogramModel::new_with_shape(0.1, 2.0, 10.0, VariogramType::Stable, 1.0).unwrap();
        let (nugget, sill, _) = stable.params();
        assert_relative_eq!(stable.semivariance(0.0), nugget, epsilon = 1e-6);
        let mut prev = nugget;
        for d in [1.0, 3.0, 5.0, 10.0, 20.0] {
            let g = stable.semivariance(d);
            assert!(
                g >= prev && g <= sill,
                "stable semivariance should increase toward sill"
            );
            prev = g;
        }
        assert_relative_eq!(stable.semivariance(100.0), sill, epsilon = 1e-4);
    }

    #[test]
    fn shape_returns_none_for_three_param_models() {
        let m = VariogramModel::new(0.0, 1.0, 1.0, VariogramType::Cubic).unwrap();
        assert_eq!(m.shape(), None);
    }

    #[test]
    fn shape_returns_some_for_stable_and_matern() {
        let stable =
            VariogramModel::new_with_shape(0.0, 1.0, 1.0, VariogramType::Stable, 1.5).unwrap();
        let matern =
            VariogramModel::new_with_shape(0.0, 1.0, 1.0, VariogramType::Matern, 2.5).unwrap();
        assert_relative_eq!(stable.shape().unwrap(), 1.5, epsilon = 1e-6);
        assert_relative_eq!(matern.shape().unwrap(), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn power_model_grows_without_a_sill() {
        let m = VariogramModel::new_power(0.1, 0.5, 1.0).unwrap();
        let g1 = m.semivariance(1.0);
        let g10 = m.semivariance(10.0);
        let g100 = m.semivariance(100.0);
        assert!(g1 < g10 && g10 < g100);
        assert_relative_eq!(m.semivariance(0.0), 0.1, epsilon = 1e-6);
    }

    #[test]
    fn power_model_rejects_invalid_exponent() {
        assert!(VariogramModel::new_power(0.0, 1.0, 0.0).is_err());
        assert!(VariogramModel::new_power(0.0, 1.0, 2.0).is_err());
        assert!(VariogramModel::new_power(0.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn hole_effect_starts_at_nugget_and_oscillates_below_sill() {
        let m = VariogramModel::new(0.1, 2.0, 10.0, VariogramType::HoleEffect).unwrap();
        assert_relative_eq!(m.semivariance(0.0), 0.1, epsilon = 1e-5);
        // At d = range/2 the hole-effect crosses above the midpoint.
        let mid = m.semivariance(5.0);
        assert!(mid > 0.1 && mid < 2.0);
    }

    #[test]
    fn conf_hypergeom_u_matches_known_identities() {
        // Identity: U(a, a+1, z) = z^{-a}.
        assert_relative_eq!(
            conf_hypergeom_u(2.0, 3.0, 1.5),
            1.5_f64.powf(-2.0),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            conf_hypergeom_u(3.0, 4.0, 0.75),
            0.75_f64.powf(-3.0),
            epsilon = 1e-6
        );
        // U(1, 1, 1) = e · E_1(1) = 0.5963473623...
        assert_relative_eq!(
            conf_hypergeom_u(1.0, 1.0, 1.0),
            0.596_347_362_3,
            epsilon = 1e-6
        );
    }

    #[test]
    fn confluent_hypergeometric_starts_at_nugget_and_rises_toward_sill() {
        let m = VariogramModel::new_with_shapes(
            0.1,
            2.0,
            10.0,
            VariogramType::ConfluentHypergeometric,
            0.5,
            Some(1.0),
        )
        .unwrap();
        assert_relative_eq!(m.semivariance(0.0), 0.1, epsilon = 1e-6);
        let mut prev = m.semivariance(0.0);
        for d in [1.0, 3.0, 5.0, 10.0, 50.0, 500.0] {
            let g = m.semivariance(d);
            assert!(
                g >= prev - 1e-6 && g <= 2.0 + 1e-6,
                "CH semivariance should increase monotonically toward the sill (d={d}, g={g})"
            );
            prev = g;
        }
        // Far-field approaches (but, with heavy tails, only slowly reaches) the sill.
        assert!(m.semivariance(1.0e6) > 1.9);
    }

    #[test]
    fn confluent_hypergeometric_shapes_round_trip() {
        let m = VariogramModel::new_with_shapes(
            0.0,
            1.0,
            5.0,
            VariogramType::ConfluentHypergeometric,
            1.5,
            Some(2.5),
        )
        .unwrap();
        assert_relative_eq!(m.shape().unwrap(), 1.5, epsilon = 1e-6);
        assert_relative_eq!(m.shape2().unwrap(), 2.5, epsilon = 1e-6);
        assert_eq!(m.variogram_type(), VariogramType::ConfluentHypergeometric);
    }

    #[test]
    fn confluent_hypergeometric_rejects_nonpositive_shapes() {
        assert!(
            VariogramModel::new_with_shapes(
                0.0,
                1.0,
                5.0,
                VariogramType::ConfluentHypergeometric,
                0.0,
                Some(1.0)
            )
            .is_err()
        );
        assert!(
            VariogramModel::new_with_shapes(
                0.0,
                1.0,
                5.0,
                VariogramType::ConfluentHypergeometric,
                1.0,
                Some(-1.0)
            )
            .is_err()
        );
    }

    #[test]
    fn confluent_hypergeometric_larger_alpha_has_thinner_tails() {
        // α is the tail-decay parameter: for fixed ν, a larger α makes the correlation fall
        // off faster, so the semivariance sits closer to the sill at a fixed far distance.
        let make = |alpha: Real| {
            VariogramModel::new_with_shapes(
                0.0,
                1.0,
                10.0,
                VariogramType::ConfluentHypergeometric,
                1.0,
                Some(alpha),
            )
            .unwrap()
        };
        let far = 60.0;
        let g_small = make(0.5).semivariance(far);
        let g_large = make(3.0).semivariance(far);
        assert!(
            g_large > g_small,
            "larger α should approach the sill faster at d={far}: α=0.5→{g_small}, α=3→{g_large}"
        );
    }

    #[test]
    fn confluent_hypergeometric_larger_nu_is_smoother_near_origin() {
        // ν is the smoothness parameter: for fixed α, a larger ν yields a flatter behavior
        // near the origin, hence a smaller semivariance at short lags.
        let make = |nu: Real| {
            VariogramModel::new_with_shapes(
                0.0,
                1.0,
                10.0,
                VariogramType::ConfluentHypergeometric,
                nu,
                Some(1.0),
            )
            .unwrap()
        };
        let near = 0.5;
        let g_rough = make(0.5).semivariance(near);
        let g_smooth = make(2.0).semivariance(near);
        assert!(
            g_smooth < g_rough,
            "larger ν should be smoother near origin at d={near}: ν=0.5→{g_rough}, ν=2→{g_smooth}"
        );
    }

    #[test]
    fn confluent_hypergeometric_covariance_complements_semivariance() {
        let m = VariogramModel::new_with_shapes(
            0.1,
            1.0,
            5.0,
            VariogramType::ConfluentHypergeometric,
            0.75,
            Some(1.5),
        )
        .unwrap();
        let d = 2.2;
        assert_relative_eq!(m.covariance(d) + m.semivariance(d), 1.0, epsilon = 1e-5);
    }
}
