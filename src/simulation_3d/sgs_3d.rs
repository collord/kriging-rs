//! 3-D sequential Gaussian simulation with streaming realization API.
//!
//! Algorithm (per v3 §"3D sequential Gaussian simulation"):
//!
//! 1. Normal-score-transform the sample data.
//! 2. Define the simulation grid.
//! 3. Build the conditioning kd-tree (anisotropy-aware) over sample
//!    locations and add their normal scores.
//! 4. Generate a seeded random path through the grid cells.
//! 5. For each cell in path order:
//!    a. Find the `max_neighbors` nearest neighbours in the growing
//!       conditioning tree.
//!    b. Build a simple-kriging system (mean = 0 in score space) and
//!       solve via the M8 robust solver.
//!    c. Draw `u ~ N(0, 1)`; the cell's simulated score is `μ + σ · u`.
//!    d. Add the simulated cell back into the conditioning tree.
//! 6. After the full path: back-transform the grid scores through the
//!    NST and **yield to the caller via the closure**. Reset the
//!    conditioning state for the next realization.
//!
//! ## Streaming, not collecting
//!
//! The full realization grid is yielded to the caller and **discarded
//! immediately**. The engine retains no realizations between calls.
//! Callers who need all realizations in memory can accumulate them in
//! the closure themselves, but the v0 SGS API is single-realization-
//! at-a-time per the v3 §"Streaming realization API" architectural
//! commitment.
//!
//! ## Solver-failure handling
//!
//! Per v3's "no silent solver failures" gate: when the SK solver
//! refuses to factor a node's system (e.g. fewer than 2 neighbours, or
//! a covariance matrix that can't be rescued by nugget inflation),
//! we **skip the cell** (leaving it at NaN in the realization). The
//! closure receives a `&[Real]` slice with NaN at unsimulated cells;
//! callers can decide whether to treat that as a hard error.
//!
//! ## Parallel variant
//!
//! [`gaussian_simulation_3d_stream_parallel`] runs realizations
//! concurrently on rayon's thread pool (native only). Each realization
//! gets a deterministic seed derived from the base seed and its
//! realization index, so same-seed-bit-identical-realization still
//! holds *per realization*. The closure is called in realization-index
//! order; up to `rayon::current_num_threads()` realizations are held
//! in memory at peak (vs 1 for the serial path).
//!
//! Note: parallel and serial paths produce **different bytes** for the
//! same base seed because the parallel path derives each
//! realization's seed independently (`mix(base_seed, r)`) while the
//! serial path threads one RNG through all realizations.

use crate::Real;
use crate::anisotropy_3d::Anisotropy3D;
use crate::coord_3d::Coord3D;
use crate::error::KrigingError;
use crate::kriging::diagnostics::SolverFailure;
use crate::kriging::solver::{SolverConfig, solve_simple_kriging_3d};
use crate::neighborhood::kdtree_3d::MutableKdTree3D;
use crate::planar_dataset_3d::PlanarDataset3D;
use crate::simulation::Rng;
use crate::simulation_3d::Grid3D;
use crate::simulation_3d::nst::NormalScoreTransform;
use crate::variogram::models::VariogramModel;

/// Tuning knobs for SGS.
#[derive(Debug, Clone, Copy)]
pub struct SgsConfig {
    /// Maximum number of conditioning neighbours used for each cell's
    /// SK system. Larger values give more accurate simulations at the
    /// cost of larger per-cell kriging matrices. Default 24 (a common
    /// GSLib default; trades accuracy and speed reasonably).
    pub max_neighbors: usize,
    /// Robust-solver configuration (condition threshold, nugget
    /// inflation, retry budget). Default is [`SolverConfig::default`].
    pub solver: SolverConfig,
}

impl Default for SgsConfig {
    fn default() -> Self {
        Self {
            max_neighbors: 24,
            solver: SolverConfig::default(),
        }
    }
}

/// SGS errors. These are surfaced *before* simulation starts (input
/// validation); per-cell solver failures don't kill the realization
/// — they leave NaN at that cell and continue.
#[derive(Debug, Clone, PartialEq)]
pub enum SgsError {
    /// The closure callback returned an error; SGS aborts.
    CallbackAborted(String),
    /// Input was invalid (zero realizations, bad grid, etc.).
    InvalidInput(String),
    /// Sample dataset has fewer than 2 points (not enough to fit NST).
    InsufficientSamples,
}

impl std::fmt::Display for SgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SgsError::CallbackAborted(s) => write!(f, "SGS aborted by callback: {s}"),
            SgsError::InvalidInput(s) => write!(f, "SGS invalid input: {s}"),
            SgsError::InsufficientSamples => {
                write!(f, "SGS needs at least 2 sample points")
            }
        }
    }
}

impl std::error::Error for SgsError {}

impl From<KrigingError> for SgsError {
    fn from(e: KrigingError) -> Self {
        SgsError::InvalidInput(format!("{e}"))
    }
}

/// Pre-built SGS model: holds the sample data, anisotropy, variogram,
/// and pre-fitted NST. Built once and reused across realizations.
#[derive(Debug, Clone)]
pub struct SgsModel3D {
    sample_coords: Vec<Coord3D>,
    /// Normal-score-transformed sample values (kriging happens in score
    /// space; back-transform happens at the very end of each
    /// realization).
    sample_scores: Vec<Real>,
    nst: NormalScoreTransform,
    anisotropy: Anisotropy3D,
    /// Variogram model in *score space*. Caller is responsible for
    /// fitting this on normal-scored sample residuals if they care
    /// about variogram fidelity; for the M11 deliverable we accept
    /// whatever variogram the caller provides.
    variogram_score: VariogramModel,
}

impl SgsModel3D {
    /// Build an SGS model from a sample dataset, anisotropy, and a
    /// variogram model fitted in score space.
    pub fn new(
        dataset: PlanarDataset3D,
        anisotropy: Anisotropy3D,
        variogram_score: VariogramModel,
    ) -> Result<Self, SgsError> {
        let (sample_coords, values) = dataset.into_parts();
        if sample_coords.len() < 2 {
            return Err(SgsError::InsufficientSamples);
        }
        let nst = NormalScoreTransform::fit(&values);
        let sample_scores: Vec<Real> = values.iter().map(|v| nst.forward(*v)).collect();
        Ok(Self {
            sample_coords,
            sample_scores,
            nst,
            anisotropy,
            variogram_score,
        })
    }

    /// Borrow the sample coordinates.
    pub fn sample_coords(&self) -> &[Coord3D] {
        &self.sample_coords
    }

    /// Borrow the normal-scored sample values.
    pub fn sample_scores(&self) -> &[Real] {
        &self.sample_scores
    }

    /// Borrow the NST so callers can back-transform other quantities
    /// (e.g. per-cell means accumulated across realizations).
    pub fn nst(&self) -> &NormalScoreTransform {
        &self.nst
    }
}

/// Output space for SGS realizations: data-space (post NST
/// back-transform) or score-space (pre back-transform). Most users want
/// `DataSpace`; validation gates that need to compare against
/// score-space kriging predictions use `ScoreSpace`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SgsOutputSpace {
    /// Yield data-space values via NST back-transform. Default.
    DataSpace,
    /// Yield raw normal scores; skip the NST back-transform.
    ScoreSpace,
}

/// Run sequential Gaussian simulation, **streaming** each realization
/// to the caller via the closure. Convenience wrapper for the default
/// data-space output; use [`gaussian_simulation_3d_stream_with`] for
/// explicit control over [`SgsOutputSpace`].
pub fn gaussian_simulation_3d_stream<F>(
    model: &SgsModel3D,
    grid: &Grid3D,
    seed: u64,
    n_realizations: usize,
    on_realization: F,
) -> Result<(), SgsError>
where
    F: FnMut(usize, &[Real]) -> Result<(), SgsError>,
{
    gaussian_simulation_3d_stream_with(
        model,
        grid,
        seed,
        n_realizations,
        SgsOutputSpace::DataSpace,
        on_realization,
    )
}

/// Streaming SGS with explicit output-space control. The closure
/// receives `(realization_index, &[Real] flattened grid)` and may
/// return `Err` to abort. Cells where the per-node kriging system
/// failed appear as NaN regardless of output space.
///
/// The grid buffer is **reused across realizations**; callers must not
/// store the slice past the end of the closure call (it's overwritten
/// on the next iteration).
pub fn gaussian_simulation_3d_stream_with<F>(
    model: &SgsModel3D,
    grid: &Grid3D,
    seed: u64,
    n_realizations: usize,
    output_space: SgsOutputSpace,
    mut on_realization: F,
) -> Result<(), SgsError>
where
    F: FnMut(usize, &[Real]) -> Result<(), SgsError>,
{
    if n_realizations == 0 {
        return Err(SgsError::InvalidInput("n_realizations must be > 0".into()));
    }

    let config = SgsConfig::default();
    let n_cells = grid.n_cells();

    // Reusable scratch buffers (allocate once, reuse across realizations).
    let mut path: Vec<usize> = Vec::with_capacity(n_cells);
    let mut grid_scores = vec![Real::NAN; n_cells];
    let mut grid_values = vec![Real::NAN; n_cells];

    // Single RNG threaded explicitly through the whole loop; this is
    // the serial path's determinism guarantee.
    let mut rng = Rng::new(seed);

    for r in 0..n_realizations {
        run_one_realization(
            model,
            grid,
            &config,
            &mut rng,
            &mut path,
            &mut grid_scores,
        );
        let output_slice =
            finalize_output(output_space, model, &grid_scores, &mut grid_values);
        on_realization(r, output_slice)?;
    }

    Ok(())
}

/// Run SGS realizations in parallel on rayon's thread pool. Native
/// only — WASM falls back to the serial path.
///
/// Each realization derives its own RNG from the base seed using
/// splitmix64 mixing: `realization_seed(r) = mix(base_seed, r)`. The
/// closure is invoked in **realization-index order**; up to
/// `rayon::current_num_threads()` realizations are held in memory at
/// peak (vs 1 for the serial path).
///
/// Same-seed-bit-identical-realization still holds *per realization
/// index*: running the same input twice produces the same bytes per
/// realization. However, **bytes do not match the serial path** for
/// the same base seed (different per-realization seed-derivation
/// schemes).
#[cfg(not(target_arch = "wasm32"))]
pub fn gaussian_simulation_3d_stream_parallel<F>(
    model: &SgsModel3D,
    grid: &Grid3D,
    seed: u64,
    n_realizations: usize,
    output_space: SgsOutputSpace,
    mut on_realization: F,
) -> Result<(), SgsError>
where
    F: FnMut(usize, &[Real]) -> Result<(), SgsError>,
{
    use rayon::prelude::*;

    if n_realizations == 0 {
        return Err(SgsError::InvalidInput("n_realizations must be > 0".into()));
    }

    let n_cells = grid.n_cells();

    // Process in chunks of `chunk_size` realizations so memory is
    // bounded (chunk_size * n_cells * sizeof(Real)) instead of
    // n_realizations * that. The chunk equals the thread count so each
    // thread gets one realization per chunk on average.
    let chunk_size = rayon::current_num_threads().max(1);

    let mut idx = 0;
    while idx < n_realizations {
        let end = (idx + chunk_size).min(n_realizations);
        let realizations: Vec<Vec<Real>> = (idx..end)
            .into_par_iter()
            .map(|r| {
                let mut rng = Rng::new(splitmix64_mix(seed, r as u64));
                let config = SgsConfig::default();
                let mut path: Vec<usize> = Vec::with_capacity(n_cells);
                let mut grid_scores = vec![Real::NAN; n_cells];
                run_one_realization(
                    model,
                    grid,
                    &config,
                    &mut rng,
                    &mut path,
                    &mut grid_scores,
                );
                let mut grid_values = vec![Real::NAN; n_cells];
                let _ = finalize_output(
                    output_space,
                    model,
                    &grid_scores,
                    &mut grid_values,
                );
                match output_space {
                    SgsOutputSpace::DataSpace => grid_values,
                    SgsOutputSpace::ScoreSpace => grid_scores,
                }
            })
            .collect();

        for (offset, grid_buf) in realizations.into_iter().enumerate() {
            on_realization(idx + offset, &grid_buf)?;
        }
        idx = end;
    }

    Ok(())
}

/// WASM (single-threaded) fallback for the parallel SGS entry point.
/// Delegates straight to the serial path so callers can `use` the
/// parallel function name unconditionally without `cfg`-gating their
/// own code.
#[cfg(target_arch = "wasm32")]
pub fn gaussian_simulation_3d_stream_parallel<F>(
    model: &SgsModel3D,
    grid: &Grid3D,
    seed: u64,
    n_realizations: usize,
    output_space: SgsOutputSpace,
    on_realization: F,
) -> Result<(), SgsError>
where
    F: FnMut(usize, &[Real]) -> Result<(), SgsError>,
{
    gaussian_simulation_3d_stream_with(
        model,
        grid,
        seed,
        n_realizations,
        output_space,
        on_realization,
    )
}

/// splitmix64 used to derive a per-realization seed from a base seed
/// and a realization index. Different from the xoshiro RNG itself —
/// this is just the diffusion function used for seed derivation, so
/// `(seed, 0)` and `(seed, 1)` produce uncorrelated streams.
///
/// Only used by the parallel SGS path, which is `cfg(not(wasm32))`.
#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn splitmix64_mix(base: u64, idx: u64) -> u64 {
    let mut z = base.wrapping_add(idx).wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Simulate one realization into `grid_scores`. The caller owns the
/// RNG and scratch buffers so this function allocates nothing on the
/// realization-hot path beyond the per-cell SK system Vecs.
fn run_one_realization(
    model: &SgsModel3D,
    grid: &Grid3D,
    config: &SgsConfig,
    rng: &mut Rng,
    path: &mut Vec<usize>,
    grid_scores: &mut [Real],
) {
    let n_cells = grid.n_cells();
    let n_samples = model.sample_coords.len();

    // 1) Random path: Fisher-Yates shuffle. Deterministic given RNG state.
    path.clear();
    path.extend(0..n_cells);
    for i in (1..n_cells).rev() {
        let r_u = rng.next_u64() % ((i as u64) + 1);
        path.swap(i, r_u as usize);
    }

    // 2) Reset score buffer to NaN for this realization.
    for s in grid_scores.iter_mut() {
        *s = Real::NAN;
    }

    // 3) Build conditioning kd-tree pre-allocated for the final
    //    size (samples + grid cells). Also a parallel Vec<Real> of
    //    conditioning scores so the SK solver has the values for
    //    its known set of indices.
    let mut tree = MutableKdTree3D::with_capacity(model.anisotropy, n_samples + n_cells);
    let mut cond_coords: Vec<Coord3D> = Vec::with_capacity(n_samples + n_cells);
    let mut cond_scores: Vec<Real> = Vec::with_capacity(n_samples + n_cells);
    for (c, s) in model
        .sample_coords
        .iter()
        .zip(model.sample_scores.iter())
    {
        tree.add(*c);
        cond_coords.push(*c);
        cond_scores.push(*s);
    }

    // 4) Walk the path, simulating one cell at a time.
    for &cell_linear in path.iter() {
        let target = grid.cell_center_linear(cell_linear);
        let neighbours = tree.nearest_n(target, config.max_neighbors);
        if neighbours.len() < 2 {
            // Consume the RNG anyway so realizations stay deterministic
            // regardless of which solver outcomes occur.
            let _ = rng.next_standard_normal();
            continue;
        }
        let n_neigh = neighbours.len();
        let mut samples = Vec::with_capacity(n_neigh);
        let mut values = Vec::with_capacity(n_neigh);
        for nb in &neighbours {
            samples.push(cond_coords[nb.index]);
            values.push(cond_scores[nb.index]);
        }

        let prediction_result = solve_simple_kriging_3d(
            &samples,
            &values,
            0.0 as Real, // SK mean in score space is zero by construction.
            target,
            &model.anisotropy,
            &model.variogram_score,
            &config.solver,
        );

        let u = rng.next_standard_normal();
        let simulated_score = match prediction_result {
            Ok(pred) => {
                let sigma = (pred.variance as f64).max(0.0).sqrt() as Real;
                pred.value + sigma * u
            }
            Err(SolverFailure::NonFiniteWeights)
            | Err(SolverFailure::PoorlyConditioned { .. })
            | Err(SolverFailure::NonSingularEvenAfterInflation { .. }) => {
                // No-silent-failure gate: leave NaN at this cell rather
                // than producing a garbage value. The realization continues.
                Real::NAN
            }
        };

        grid_scores[cell_linear] = simulated_score;

        if simulated_score.is_finite() {
            tree.add(target);
            cond_coords.push(target);
            cond_scores.push(simulated_score);
        }
    }
}

/// Translate `grid_scores` into the caller's chosen output space.
/// Returns a reference into either `grid_scores` (ScoreSpace) or the
/// freshly back-transformed `grid_values` (DataSpace).
fn finalize_output<'a>(
    output_space: SgsOutputSpace,
    model: &SgsModel3D,
    grid_scores: &'a [Real],
    grid_values: &'a mut [Real],
) -> &'a [Real] {
    match output_space {
        SgsOutputSpace::DataSpace => {
            for (out, s) in grid_values.iter_mut().zip(grid_scores.iter()) {
                *out = if s.is_finite() {
                    model.nst.backward(*s)
                } else {
                    Real::NAN
                };
            }
            grid_values
        }
        SgsOutputSpace::ScoreSpace => grid_scores,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::models::VariogramType;

    fn small_dataset() -> PlanarDataset3D {
        PlanarDataset3D::new(
            vec![
                Coord3D::new(0.0, 0.0, 0.0),
                Coord3D::new(10.0, 0.0, 0.0),
                Coord3D::new(0.0, 10.0, 0.0),
                Coord3D::new(10.0, 10.0, 0.0),
                Coord3D::new(0.0, 0.0, 10.0),
            ],
            vec![1.0, 3.0, 5.0, 7.0, 9.0],
        )
        .unwrap()
    }

    fn small_grid() -> Grid3D {
        Grid3D::new(
            5,
            5,
            3,
            Coord3D::new(0.0, 0.0, 0.0),
            Coord3D::new(2.5, 2.5, 5.0),
        )
        .unwrap()
    }

    fn variogram() -> VariogramModel {
        // In score space, the data has unit-ish variance. Use a
        // reasonable exponential.
        VariogramModel::new(0.0, 1.0, 10.0, VariogramType::Exponential).unwrap()
    }

    #[test]
    fn produces_finite_grid_for_small_problem() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        let mut grids = Vec::new();
        gaussian_simulation_3d_stream(&model, &grid, 42, 1, |idx, gv| {
            assert_eq!(idx, 0);
            grids.push(gv.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(grids.len(), 1);
        assert_eq!(grids[0].len(), grid.n_cells());
        // All cells should be finite (a 5-sample, 75-cell run with our
        // exponential variogram has no rank-deficient conditioning sets
        // in practice).
        for (i, v) in grids[0].iter().enumerate() {
            assert!(v.is_finite(), "cell {i} is NaN");
        }
    }

    #[test]
    fn same_seed_produces_bit_identical_realization() {
        // The determinism gate: identical seed and identical inputs ->
        // identical realization, byte-for-byte.
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        let mut first: Vec<Real> = Vec::new();
        gaussian_simulation_3d_stream(&model, &grid, 12345, 1, |_, gv| {
            first.extend_from_slice(gv);
            Ok(())
        })
        .unwrap();
        let mut second: Vec<Real> = Vec::new();
        gaussian_simulation_3d_stream(&model, &grid, 12345, 1, |_, gv| {
            second.extend_from_slice(gv);
            Ok(())
        })
        .unwrap();
        assert_eq!(first.len(), second.len());
        for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "cell {i} differs: {a} vs {b}",
            );
        }
    }

    #[test]
    fn different_seed_produces_different_realization() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        let mut a: Vec<Real> = Vec::new();
        gaussian_simulation_3d_stream(&model, &grid, 1, 1, |_, gv| {
            a.extend_from_slice(gv);
            Ok(())
        })
        .unwrap();
        let mut b: Vec<Real> = Vec::new();
        gaussian_simulation_3d_stream(&model, &grid, 2, 1, |_, gv| {
            b.extend_from_slice(gv);
            Ok(())
        })
        .unwrap();
        // At least one cell should differ; almost certainly all of them.
        let any_diff = a.iter().zip(b.iter()).any(|(x, y)| x != y);
        assert!(any_diff, "different seeds should produce different output");
    }

    #[test]
    fn closure_abort_propagates_as_error() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        let result = gaussian_simulation_3d_stream(&model, &grid, 42, 3, |idx, _| {
            if idx == 1 {
                Err(SgsError::CallbackAborted("test abort".into()))
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(SgsError::CallbackAborted(_))));
    }

    #[test]
    fn zero_realizations_errors() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        let result = gaussian_simulation_3d_stream(&model, &grid, 42, 0, |_, _| Ok(()));
        assert!(matches!(result, Err(SgsError::InvalidInput(_))));
    }

    #[test]
    fn realizations_stay_within_data_range() {
        // NST backward clamps to the empirical sample range; therefore
        // every simulated cell must be in [min_sample, max_sample].
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        gaussian_simulation_3d_stream(&model, &grid, 9999, 1, |_, gv| {
            for v in gv {
                assert!(*v >= 1.0 - 1e-6 && *v <= 9.0 + 1e-6,
                    "value {v} outside sample range [1, 9]");
            }
            Ok(())
        })
        .unwrap();
    }

    // ---------- Parallel SGS tests (native only) ----------

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_same_seed_per_realization_is_bit_identical() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();

        let mut first: Vec<Vec<Real>> = Vec::new();
        gaussian_simulation_3d_stream_parallel(
            &model,
            &grid,
            7777,
            4,
            SgsOutputSpace::DataSpace,
            |_, gv| {
                first.push(gv.to_vec());
                Ok(())
            },
        )
        .unwrap();
        let mut second: Vec<Vec<Real>> = Vec::new();
        gaussian_simulation_3d_stream_parallel(
            &model,
            &grid,
            7777,
            4,
            SgsOutputSpace::DataSpace,
            |_, gv| {
                second.push(gv.to_vec());
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(first.len(), 4);
        assert_eq!(second.len(), 4);
        for r in 0..4 {
            assert_eq!(first[r].len(), second[r].len(), "realization {r} length");
            for (i, (a, b)) in first[r].iter().zip(second[r].iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "realization {r}, cell {i}: a={a} b={b}",
                );
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_callback_fires_in_realization_index_order() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();

        let mut indices: Vec<usize> = Vec::new();
        gaussian_simulation_3d_stream_parallel(
            &model,
            &grid,
            42,
            10,
            SgsOutputSpace::ScoreSpace,
            |idx, _| {
                indices.push(idx);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(indices, (0..10).collect::<Vec<_>>());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_different_realization_indices_produce_different_grids() {
        // Same base seed, different realization indices. Each
        // realization must produce a *different* grid (no collision
        // in the per-realization seed derivation).
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();

        let mut grids: Vec<Vec<Real>> = Vec::new();
        gaussian_simulation_3d_stream_parallel(
            &model,
            &grid,
            13579,
            3,
            SgsOutputSpace::ScoreSpace,
            |_, gv| {
                grids.push(gv.to_vec());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(grids.len(), 3);
        let same_01 = grids[0] == grids[1];
        let same_02 = grids[0] == grids[2];
        assert!(!same_01, "realizations 0 and 1 should differ");
        assert!(!same_02, "realizations 0 and 2 should differ");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn parallel_callback_abort_propagates() {
        let model =
            SgsModel3D::new(small_dataset(), Anisotropy3D::identity(), variogram())
                .unwrap();
        let grid = small_grid();
        let result = gaussian_simulation_3d_stream_parallel(
            &model,
            &grid,
            1,
            5,
            SgsOutputSpace::DataSpace,
            |idx, _| {
                if idx == 2 {
                    Err(SgsError::CallbackAborted("test".into()))
                } else {
                    Ok(())
                }
            },
        );
        assert!(matches!(result, Err(SgsError::CallbackAborted(_))));
    }
}
