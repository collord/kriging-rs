use crate::Real;
use crate::error::KrigingError;
use crate::variogram::empirical::EmpiricalVariogram;
use crate::variogram::models::{VariogramModel, VariogramType};

#[derive(Debug, Clone)]
pub struct FitResult {
    pub model: VariogramModel,
    pub residuals: Real,
}

fn model_from_params(
    nugget: Real,
    sill: Real,
    range: Real,
    model_type: VariogramType,
    shape: Option<Real>,
) -> VariogramModel {
    match shape {
        None => VariogramModel::new(nugget, sill, range, model_type)
            .expect("grid ensures nugget >= 0, sill > nugget, range > 0"),
        Some(s) => VariogramModel::new_with_shape(nugget, sill, range, model_type, s)
            .expect("grid ensures valid shape for Stable/Matérn"),
    }
}

/// Fits a parametric variogram by minimizing weighted sum of squared residuals over a 5×5×5 grid.
///
/// The empirical variogram must be non-empty and have matching-length distance/semivariance/n_pairs
/// arrays (e.g. from [`compute_empirical_variogram`](crate::compute_empirical_variogram)). Returns
/// [`KrigingError::FittingError`] if these preconditions are violated.
///
/// The grid spans plausible scales around data-derived guesses (sill, range, nugget). Accuracy is
/// limited by grid resolution: the best point may be 20–40% away from the continuous optimum in
/// sill/range. For typical empirical variograms (noisy, few bins) this is usually acceptable; for
/// noiseless synthetic data the grid can pick a different local minimum than the true parameters.
pub fn fit_variogram(
    empirical: &EmpiricalVariogram,
    model_type: VariogramType,
) -> Result<FitResult, KrigingError> {
    if empirical.semivariances.is_empty()
        || empirical.distances.is_empty()
        || empirical.semivariances.len() != empirical.distances.len()
        || empirical.semivariances.len() != empirical.n_pairs.len()
    {
        return Err(KrigingError::FittingError(
            "empirical variogram is empty or has mismatched arrays".to_string(),
        ));
    }
    let sill_guess = empirical
        .semivariances
        .iter()
        .copied()
        .fold(0.0 as Real, |a, b| a.max(b))
        .max(Real::EPSILON);
    let range_guess = empirical
        .distances
        .iter()
        .copied()
        .fold(0.0 as Real, |a, b| a.max(b))
        .max(Real::EPSILON);
    let nugget_guess = empirical.semivariances[0].min(sill_guess * 0.5).max(0.0);

    let shape_values: Option<&[Real]> = match model_type {
        VariogramType::Stable => Some(&[0.5, 1.0, 1.5, 2.0]),
        VariogramType::Matern => Some(&[0.5, 1.0, 2.0, 3.0]),
        // Power exponent must lie in (0, 2); sample plausible values (avoiding the endpoints).
        VariogramType::Power => Some(&[0.5, 1.0, 1.5, 1.9]),
        _ => None,
    };

    let mut best = None::<FitResult>;
    for nugget_frac in [0.0, 0.05, 0.1, 0.2, 0.3] {
        for sill_scale in [0.7, 0.9, 1.0, 1.1, 1.3] {
            for range_scale in [0.4, 0.7, 1.0, 1.4, 1.8] {
                let nugget = (nugget_guess * (1.0 + nugget_frac)).min(sill_guess * sill_scale);
                let sill = (sill_guess * sill_scale).max(nugget + 1e-9);
                let range = (range_guess * range_scale).max(1e-9);
                let shapes: Vec<Option<Real>> = match shape_values {
                    None => vec![None],
                    Some(slices) => slices.iter().copied().map(Some).collect(),
                };
                for shape in shapes {
                    let model = model_from_params(nugget, sill, range, model_type, shape);
                    let residuals = weighted_residuals(empirical, model);
                    let candidate = FitResult { model, residuals };
                    best = Some(match best {
                        None => candidate,
                        Some(ref curr) if residuals < curr.residuals => candidate,
                        Some(curr) => curr,
                    });
                }
            }
        }
    }
    let best = best.expect("grid has at least one iteration");
    // Refine the grid minimum with a few Nelder–Mead iterations over (nugget, sill, range)
    // (shape stays fixed at whatever the grid picked). This typically recovers the continuous
    // optimum from a nearby grid point while staying numerically cheap.
    Ok(refine_nelder_mead(empirical, model_type, best))
}

/// A light Nelder–Mead simplex over `(nugget, sill, range)` starting from an existing fit.
/// Shape (for Stable/Matérn/Power) is held fixed — the grid has already sampled it. Only valid
/// candidates (those satisfying the model's constructor preconditions) are evaluated.
fn refine_nelder_mead(
    empirical: &EmpiricalVariogram,
    model_type: VariogramType,
    start: FitResult,
) -> FitResult {
    let shape = start.model.shape();
    let (n0, s0, r0) = start.model.params();
    let build = |p: [Real; 3]| -> Option<VariogramModel> {
        let (nugget, sill, range) = (p[0], p[1], p[2]);
        if !(nugget.is_finite() && sill.is_finite() && range.is_finite()) {
            return None;
        }
        if nugget < 0.0 || range <= 0.0 {
            return None;
        }
        match model_type {
            VariogramType::Power => VariogramModel::new_power(nugget, sill, range).ok(),
            _ => match shape {
                Some(s) => VariogramModel::new_with_shape(nugget, sill, range, model_type, s).ok(),
                None => VariogramModel::new(nugget, sill, range, model_type).ok(),
            },
        }
    };
    let eval = |p: [Real; 3]| -> Real {
        match build(p) {
            Some(m) => weighted_residuals(empirical, m),
            None => Real::INFINITY,
        }
    };
    let step_n = (s0 * 0.05).max(1e-6);
    let step_s = (s0 * 0.1).max(1e-6);
    let step_r = (r0 * 0.1).max(1e-6);
    let mut simplex: [([Real; 3], Real); 4] = [
        ([n0, s0, r0], start.residuals),
        ([n0 + step_n, s0, r0], 0.0),
        ([n0, s0 + step_s, r0], 0.0),
        ([n0, s0, r0 + step_r], 0.0),
    ];
    for entry in simplex.iter_mut().skip(1) {
        entry.1 = eval(entry.0);
    }
    for _ in 0..64 {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best, worst) = (simplex[0], simplex[3]);
        if !worst.1.is_finite() && !best.1.is_finite() {
            break;
        }
        // Centroid of all but worst.
        let c = [
            (simplex[0].0[0] + simplex[1].0[0] + simplex[2].0[0]) / 3.0,
            (simplex[0].0[1] + simplex[1].0[1] + simplex[2].0[1]) / 3.0,
            (simplex[0].0[2] + simplex[1].0[2] + simplex[2].0[2]) / 3.0,
        ];
        let reflect = [
            c[0] + (c[0] - worst.0[0]),
            c[1] + (c[1] - worst.0[1]),
            c[2] + (c[2] - worst.0[2]),
        ];
        let r_val = eval(reflect);
        if r_val < simplex[2].1 && r_val >= best.1 {
            simplex[3] = (reflect, r_val);
            continue;
        }
        if r_val < best.1 {
            let expand = [
                c[0] + 2.0 * (c[0] - worst.0[0]),
                c[1] + 2.0 * (c[1] - worst.0[1]),
                c[2] + 2.0 * (c[2] - worst.0[2]),
            ];
            let e_val = eval(expand);
            simplex[3] = if e_val < r_val {
                (expand, e_val)
            } else {
                (reflect, r_val)
            };
            continue;
        }
        let contract = [
            c[0] + 0.5 * (worst.0[0] - c[0]),
            c[1] + 0.5 * (worst.0[1] - c[1]),
            c[2] + 0.5 * (worst.0[2] - c[2]),
        ];
        let k_val = eval(contract);
        if k_val < worst.1 {
            simplex[3] = (contract, k_val);
            continue;
        }
        // Shrink toward best.
        for slot in simplex.iter_mut().skip(1) {
            let p = [
                best.0[0] + 0.5 * (slot.0[0] - best.0[0]),
                best.0[1] + 0.5 * (slot.0[1] - best.0[1]),
                best.0[2] + 0.5 * (slot.0[2] - best.0[2]),
            ];
            *slot = (p, eval(p));
        }
    }
    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (p, r) = simplex[0];
    match build(p) {
        Some(m) if r < start.residuals => FitResult {
            model: m,
            residuals: r,
        },
        _ => start,
    }
}

pub(crate) fn weighted_residuals(emp: &EmpiricalVariogram, model: VariogramModel) -> Real {
    emp.distances
        .iter()
        .zip(emp.semivariances.iter())
        .zip(emp.n_pairs.iter())
        .map(|((d, y), w)| {
            let diff = y - model.semivariance(*d);
            (*w as Real) * diff * diff
        })
        .sum()
}

// -----------------------------------------------------------------------------
// 3-D spherical joint fit across three axis-aligned experimental variograms.
// -----------------------------------------------------------------------------

/// Parameters of a 3-D anisotropic spherical variogram model: a shared
/// `nugget` and `sill`, and an independent effective range along each
/// of the major, minor, and vertical axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spherical3DJointFit {
    pub nugget: Real,
    pub sill: Real,
    pub range_major: Real,
    pub range_minor: Real,
    pub range_vertical: Real,
    /// Weighted sum of squared residuals at the returned parameter set.
    pub residuals: Real,
}

#[inline]
fn spherical_kernel(t: Real) -> Real {
    if t >= 1.0 {
        1.0
    } else if t <= 0.0 {
        0.0
    } else {
        1.5 * t - 0.5 * t * t * t
    }
}

#[inline]
fn spherical_gamma(distance: Real, nugget: Real, sill: Real, range: Real) -> Real {
    if range <= 0.0 {
        return nugget;
    }
    nugget + (sill - nugget) * spherical_kernel(distance / range)
}

fn axis_residuals(empirical: &EmpiricalVariogram, nugget: Real, sill: Real, range: Real) -> Real {
    empirical
        .distances
        .iter()
        .zip(empirical.semivariances.iter())
        .zip(empirical.n_pairs.iter())
        .map(|((d, y), w)| {
            let diff = y - spherical_gamma(*d, nugget, sill, range);
            (*w as Real) * diff * diff
        })
        .sum()
}

fn joint_residuals(
    major: &EmpiricalVariogram,
    minor: &EmpiricalVariogram,
    vertical: &EmpiricalVariogram,
    nugget: Real,
    sill: Real,
    range_major: Real,
    range_minor: Real,
    range_vertical: Real,
) -> Real {
    axis_residuals(major, nugget, sill, range_major)
        + axis_residuals(minor, nugget, sill, range_minor)
        + axis_residuals(vertical, nugget, sill, range_vertical)
}

/// Initial guess for `(nugget, sill, range)` from a single experimental
/// variogram: sill ≈ max(γ̂), range ≈ max(distance), nugget ≈ γ̂[0].
fn initial_axis_guess(empirical: &EmpiricalVariogram) -> (Real, Real, Real) {
    let sill = empirical
        .semivariances
        .iter()
        .copied()
        .fold(0.0 as Real, |a, b| a.max(b))
        .max(Real::EPSILON);
    let range = empirical
        .distances
        .iter()
        .copied()
        .fold(0.0 as Real, |a, b| a.max(b))
        .max(Real::EPSILON);
    let nugget = empirical
        .semivariances
        .first()
        .copied()
        .unwrap_or(0.0)
        .min(sill * 0.5)
        .max(0.0);
    (nugget, sill, range)
}

/// Validate that a parameter vector is in the feasible region for the
/// 3-D spherical model.
#[inline]
fn joint_params_ok(p: &[Real; 5]) -> bool {
    if p.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let (nugget, sill, r_major, r_minor, r_vertical) = (p[0], p[1], p[2], p[3], p[4]);
    nugget >= 0.0
        && sill > nugget
        && r_major > 0.0
        && r_minor > 0.0
        && r_vertical > 0.0
}

/// Joint least-squares fit of a 3-D spherical variogram model to three
/// axis-aligned experimental variograms.
///
/// The model has 5 parameters: shared `(nugget, sill)` and per-axis
/// ranges `(range_major, range_minor, range_vertical)`. Optimization is
/// weighted Nelder–Mead in 5 dimensions, seeded from per-axis isotropic
/// guesses and aggregated by median sill and median nugget.
///
/// All three input variograms must be non-empty with matching-length
/// arrays. Each is weighted by its own `n_pairs` so bins with more
/// pairs contribute more to the loss.
pub fn fit_spherical_3d_joint(
    major: &EmpiricalVariogram,
    minor: &EmpiricalVariogram,
    vertical: &EmpiricalVariogram,
) -> Result<Spherical3DJointFit, KrigingError> {
    for (name, ev) in [("major", major), ("minor", minor), ("vertical", vertical)] {
        if ev.distances.is_empty() {
            return Err(KrigingError::FittingError(format!(
                "{name} experimental variogram is empty"
            )));
        }
        if ev.distances.len() != ev.semivariances.len()
            || ev.distances.len() != ev.n_pairs.len()
        {
            return Err(KrigingError::FittingError(format!(
                "{name} experimental variogram has mismatched array lengths"
            )));
        }
    }

    // Seed from per-axis guesses. Take the median of the three (nugget,
    // sill) candidates as the shared starting point; each axis's range
    // guess feeds the corresponding dimension.
    let (n_maj, s_maj, r_maj) = initial_axis_guess(major);
    let (n_min, s_min, r_min) = initial_axis_guess(minor);
    let (n_vrt, s_vrt, r_vrt) = initial_axis_guess(vertical);
    let median3 = |mut a: [Real; 3]| {
        a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        a[1]
    };
    let nugget0 = median3([n_maj, n_min, n_vrt]);
    let sill0_raw = median3([s_maj, s_min, s_vrt]);
    let sill0 = sill0_raw.max(nugget0 + Real::EPSILON);

    let mut start: [Real; 5] = [nugget0, sill0, r_maj, r_min, r_vrt];

    let eval = |p: [Real; 5]| -> Real {
        if !joint_params_ok(&p) {
            return Real::INFINITY;
        }
        joint_residuals(major, minor, vertical, p[0], p[1], p[2], p[3], p[4])
    };

    let mut start_val = eval(start);
    if !start_val.is_finite() {
        // Seed produced an infeasible point; fall back to symmetric
        // ranges around the average axis extent.
        let r_avg = (r_maj + r_min + r_vrt) / 3.0;
        start = [
            nugget0.max(0.0),
            sill0.max(nugget0 + Real::EPSILON),
            r_avg.max(Real::EPSILON),
            r_avg.max(Real::EPSILON),
            r_avg.max(Real::EPSILON),
        ];
        start_val = eval(start);
        if !start_val.is_finite() {
            return Err(KrigingError::FittingError(
                "could not find a feasible starting point for 3-D spherical fit".to_string(),
            ));
        }
    }

    // Initial simplex: start + perturbations along each dimension. Step
    // sizes are a small fraction of each parameter's magnitude with
    // floors so degenerate scales don't collapse the simplex.
    let step_nugget = (start[1] * 0.05).max(1e-6);
    let step_sill = (start[1] * 0.1).max(1e-6);
    let step_r_major = (start[2] * 0.1).max(1e-6);
    let step_r_minor = (start[3] * 0.1).max(1e-6);
    let step_r_vertical = (start[4] * 0.1).max(1e-6);
    let mut simplex: [([Real; 5], Real); 6] = [
        (start, start_val),
        ([start[0] + step_nugget, start[1], start[2], start[3], start[4]], 0.0),
        ([start[0], start[1] + step_sill, start[2], start[3], start[4]], 0.0),
        ([start[0], start[1], start[2] + step_r_major, start[3], start[4]], 0.0),
        ([start[0], start[1], start[2], start[3] + step_r_minor, start[4]], 0.0),
        ([start[0], start[1], start[2], start[3], start[4] + step_r_vertical], 0.0),
    ];
    for entry in simplex.iter_mut().skip(1) {
        entry.1 = eval(entry.0);
    }

    // Standard Nelder–Mead reflect / expand / contract / shrink with
    // 128 iterations -- empirically enough to converge for 5 dims on
    // smooth quadratic-ish loss surfaces.
    let centroid = |s: &[([Real; 5], Real); 6], skip: usize| -> [Real; 5] {
        let mut c = [0.0 as Real; 5];
        let mut n = 0;
        for (i, entry) in s.iter().enumerate() {
            if i == skip {
                continue;
            }
            for j in 0..5 {
                c[j] += entry.0[j];
            }
            n += 1;
        }
        for j in 0..5 {
            c[j] /= n as Real;
        }
        c
    };

    for _ in 0..128 {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let best = simplex[0];
        let second_worst = simplex[4];
        let worst = simplex[5];

        if !best.1.is_finite() && !worst.1.is_finite() {
            break;
        }

        let c = centroid(&simplex, 5);

        // Reflect.
        let mut reflect = [0.0 as Real; 5];
        for j in 0..5 {
            reflect[j] = c[j] + (c[j] - worst.0[j]);
        }
        let r_val = eval(reflect);

        if r_val < second_worst.1 && r_val >= best.1 {
            simplex[5] = (reflect, r_val);
            continue;
        }

        if r_val < best.1 {
            // Expand.
            let mut expand = [0.0 as Real; 5];
            for j in 0..5 {
                expand[j] = c[j] + 2.0 * (c[j] - worst.0[j]);
            }
            let e_val = eval(expand);
            simplex[5] = if e_val < r_val {
                (expand, e_val)
            } else {
                (reflect, r_val)
            };
            continue;
        }

        // Contract.
        let mut contract = [0.0 as Real; 5];
        for j in 0..5 {
            contract[j] = c[j] + 0.5 * (worst.0[j] - c[j]);
        }
        let cn_val = eval(contract);
        if cn_val < worst.1 {
            simplex[5] = (contract, cn_val);
            continue;
        }

        // Shrink toward best.
        for i in 1..6 {
            let mut shrunk = [0.0 as Real; 5];
            for j in 0..5 {
                shrunk[j] = best.0[j] + 0.5 * (simplex[i].0[j] - best.0[j]);
            }
            simplex[i] = (shrunk, eval(shrunk));
        }
    }

    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let (p, residuals) = simplex[0];
    if !residuals.is_finite() {
        return Err(KrigingError::FittingError(
            "3-D spherical joint fit did not converge to a finite residual".to_string(),
        ));
    }
    Ok(Spherical3DJointFit {
        nugget: p[0],
        sill: p[1],
        range_major: p[2],
        range_minor: p[3],
        range_vertical: p[4],
        residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_variogram_rejects_empty_empirical() {
        let empirical = EmpiricalVariogram {
            distances: vec![],
            semivariances: vec![],
            n_pairs: vec![],
        };
        let result = fit_variogram(&empirical, VariogramType::Exponential);
        assert!(result.is_err(), "empty empirical must be rejected");
    }

    #[test]
    fn fit_variogram_returns_finite_solution() {
        let empirical = EmpiricalVariogram {
            distances: vec![10.0, 20.0, 30.0, 40.0],
            semivariances: vec![0.2, 0.4, 0.6, 0.75],
            n_pairs: vec![8, 9, 7, 6],
        };
        let fit = fit_variogram(&empirical, VariogramType::Exponential).expect("fit should work");
        assert!(fit.residuals.is_finite());
        let (_, sill, range) = fit.model.params();
        assert!(sill > 0.0);
        assert!(range > 0.0);
    }

    #[test]
    fn fit_synthetic_exponential_returns_valid_params() {
        let true_model = VariogramModel::new(0.1, 2.0, 25.0, VariogramType::Exponential).unwrap();
        let distances = vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];
        let semivariances: Vec<Real> = distances
            .iter()
            .map(|&d| true_model.semivariance(d))
            .collect();
        let n_pairs = vec![10, 12, 11, 9, 8, 7, 6, 5];
        let empirical = EmpiricalVariogram {
            distances,
            semivariances,
            n_pairs,
        };
        let fit = fit_variogram(&empirical, VariogramType::Exponential).expect("fit should work");
        assert!(fit.residuals.is_finite());
        let (nugget, sill, range) = fit.model.params();
        assert!(nugget >= 0.0, "nugget {} should be non-negative", nugget);
        assert!(
            sill > nugget,
            "sill {} should exceed nugget {}",
            sill,
            nugget
        );
        assert!(range > 0.0, "range {} should be positive", range);
    }

    #[test]
    fn fit_spherical_and_gaussian_return_finite() {
        let empirical = EmpiricalVariogram {
            distances: vec![10.0, 20.0, 30.0, 40.0],
            semivariances: vec![0.2, 0.4, 0.6, 0.75],
            n_pairs: vec![8, 9, 7, 6],
        };
        for vt in [VariogramType::Spherical, VariogramType::Gaussian] {
            let fit = fit_variogram(&empirical, vt).expect("fit should work");
            assert!(fit.residuals.is_finite());
            let (_, sill, range) = fit.model.params();
            assert!(sill > 0.0);
            assert!(range > 0.0);
        }
    }

    #[test]
    fn nelder_mead_refinement_does_not_worsen_grid_fit() {
        // True exponential with parameters between grid points, so the grid itself will
        // land at a suboptimal point and refinement should tighten the fit.
        let true_model = VariogramModel::new(0.07, 1.83, 27.3, VariogramType::Exponential).unwrap();
        let distances: Vec<Real> = (1..=20).map(|i| i as Real * 2.5).collect();
        let semivariances: Vec<Real> = distances
            .iter()
            .map(|&d| true_model.semivariance(d))
            .collect();
        let n_pairs = vec![10usize; distances.len()];
        let empirical = EmpiricalVariogram {
            distances: distances.clone(),
            semivariances: semivariances.clone(),
            n_pairs: n_pairs.clone(),
        };
        let refined = fit_variogram(&empirical, VariogramType::Exponential).unwrap();
        // Compare against the residuals of the best grid-only fit by rerunning in a local
        // wrapper with refinement disabled (we just recompute the grid search inline).
        let sill_guess = empirical
            .semivariances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        let range_guess = empirical
            .distances
            .iter()
            .copied()
            .fold(0.0 as Real, Real::max);
        let nugget_guess = empirical.semivariances[0].min(sill_guess * 0.5).max(0.0);
        let mut grid_best = Real::INFINITY;
        for nf in [0.0, 0.05, 0.1, 0.2, 0.3] {
            for ss in [0.7, 0.9, 1.0, 1.1, 1.3] {
                for rs in [0.4, 0.7, 1.0, 1.4, 1.8] {
                    let nug = (nugget_guess * (1.0 + nf)).min(sill_guess * ss);
                    let sill = (sill_guess * ss).max(nug + 1e-9);
                    let range = (range_guess * rs).max(1e-9);
                    let m =
                        VariogramModel::new(nug, sill, range, VariogramType::Exponential).unwrap();
                    let r = weighted_residuals(&empirical, m);
                    if r < grid_best {
                        grid_best = r;
                    }
                }
            }
        }
        assert!(
            refined.residuals <= grid_best * 1.000001,
            "refined residuals {} should not exceed grid-best {}",
            refined.residuals,
            grid_best
        );
    }

    #[test]
    fn fit_cubic_stable_matern_return_finite() {
        let empirical = EmpiricalVariogram {
            distances: vec![10.0, 20.0, 30.0, 40.0],
            semivariances: vec![0.2, 0.4, 0.6, 0.75],
            n_pairs: vec![8, 9, 7, 6],
        };
        for vt in [
            VariogramType::Cubic,
            VariogramType::Stable,
            VariogramType::Matern,
        ] {
            let fit = fit_variogram(&empirical, vt).expect("fit should work");
            assert!(fit.residuals.is_finite());
            let (nugget, sill, range) = fit.model.params();
            assert!(nugget >= 0.0);
            assert!(sill > nugget);
            assert!(range > 0.0);
            if let Some(shape) = fit.model.shape() {
                assert!(shape.is_finite());
                assert!(shape > 0.0);
            }
        }
    }

    // ---- 3D spherical joint fit ---------------------------------------

    /// Synthesize an axis-aligned experimental variogram by sampling
    /// the true spherical model at given lag distances; n_pairs = 10 at
    /// every bin (uniform weighting).
    fn synth_spherical(
        distances: &[Real],
        nugget: Real,
        sill: Real,
        range: Real,
    ) -> EmpiricalVariogram {
        let semivariances: Vec<Real> = distances
            .iter()
            .map(|d| spherical_gamma(*d, nugget, sill, range))
            .collect();
        EmpiricalVariogram {
            distances: distances.to_vec(),
            semivariances,
            n_pairs: vec![10; distances.len()],
        }
    }

    #[test]
    fn joint_fit_recovers_isotropic_truth_on_noiseless_data() {
        let lags: Vec<Real> = (1..=10).map(|i| i as Real * 5.0).collect();
        let truth = (0.1 as Real, 1.0 as Real, 40.0 as Real);
        let major = synth_spherical(&lags, truth.0, truth.1, truth.2);
        let minor = synth_spherical(&lags, truth.0, truth.1, truth.2);
        let vertical = synth_spherical(&lags, truth.0, truth.1, truth.2);

        let fit = fit_spherical_3d_joint(&major, &minor, &vertical).unwrap();
        approx::assert_relative_eq!(fit.nugget as f64, truth.0 as f64, epsilon = 1e-2);
        approx::assert_relative_eq!(fit.sill as f64, truth.1 as f64, epsilon = 1e-2);
        approx::assert_relative_eq!(fit.range_major as f64, truth.2 as f64, epsilon = 1.0);
        approx::assert_relative_eq!(fit.range_minor as f64, truth.2 as f64, epsilon = 1.0);
        approx::assert_relative_eq!(fit.range_vertical as f64, truth.2 as f64, epsilon = 1.0);
    }

    #[test]
    fn joint_fit_recovers_anisotropic_ranges() {
        // Three independent ranges, common nugget and sill.
        let nugget: Real = 0.05;
        let sill: Real = 1.0;
        let lags_major: Vec<Real> = (1..=10).map(|i| i as Real * 5.0).collect();
        let lags_minor: Vec<Real> = (1..=10).map(|i| i as Real * 3.0).collect();
        let lags_vertical: Vec<Real> = (1..=10).map(|i| i as Real * 1.0).collect();
        let major = synth_spherical(&lags_major, nugget, sill, 40.0);
        let minor = synth_spherical(&lags_minor, nugget, sill, 25.0);
        let vertical = synth_spherical(&lags_vertical, nugget, sill, 8.0);

        let fit = fit_spherical_3d_joint(&major, &minor, &vertical).unwrap();
        approx::assert_relative_eq!(fit.nugget as f64, nugget as f64, epsilon = 5e-2);
        approx::assert_relative_eq!(fit.sill as f64, sill as f64, epsilon = 5e-2);
        approx::assert_relative_eq!(fit.range_major as f64, 40.0, epsilon = 2.0);
        approx::assert_relative_eq!(fit.range_minor as f64, 25.0, epsilon = 2.0);
        approx::assert_relative_eq!(fit.range_vertical as f64, 8.0, epsilon = 2.0);
    }

    #[test]
    fn joint_fit_rejects_empty_axis_variogram() {
        let lags: Vec<Real> = (1..=5).map(|i| i as Real).collect();
        let nonempty = synth_spherical(&lags, 0.0, 1.0, 5.0);
        let empty = EmpiricalVariogram {
            distances: vec![],
            semivariances: vec![],
            n_pairs: vec![],
        };
        assert!(fit_spherical_3d_joint(&empty, &nonempty, &nonempty).is_err());
        assert!(fit_spherical_3d_joint(&nonempty, &empty, &nonempty).is_err());
        assert!(fit_spherical_3d_joint(&nonempty, &nonempty, &empty).is_err());
    }
}
