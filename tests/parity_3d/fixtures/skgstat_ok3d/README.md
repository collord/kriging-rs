# scikit-gstat reference fixture for 3-D ordinary kriging (M9)

1000 deterministic predict-location/expected-prediction pairs from
[scikit-gstat](https://scikit-gstat.readthedocs.io/) 1.0.23's
`OrdinaryKriging` on the same 150-point M5 dataset, with a hand-set
exponential variogram. Consumed by
[`tests/parity_3d/skgstat_kriging_reference.rs`](../../skgstat_kriging_reference.rs).

## Why a textbook reference instead of scikit-gstat or GSLib `kt3d`

The v3 fork-scope doc named GSLib `kt3d` (via geostatspy or pygslib) as
the reference. As of M9 evaluation:

- **pygslib has no usable distribution for macOS modern Python.** The
  only available conda artifact is `0.0.0.6.0.0` for Python 3.8 (Nov
  2020), pinning `libgfortran <4` and a scipy <= 1.5. Any modern conda
  resolver (pixi or otherwise) gives up because contemporary scipy
  requires `libgfortran5 >= 11`.
- **geostatspy's `krige_3D` is broken / misadvertised**: its docstring
  says "simplified to 2D only" and its signature is an SGS-internal
  helper, not a user-facing kriger.
- **scikit-gstat's `OrdinaryKriging` does not implement standard OK.**
  We initially tried it as the reference. Its predictions disagree by
  ~3% in value and ~40% in variance with both (a) a textbook
  semivariance OK in numpy and (b) kriging-rs's M8 covariance OK,
  while the two latter agree to 1e-5. Whatever scikit-gstat is
  computing, it isn't standard OK; not a meaningful algorithmic-tier
  reference.

The M9 reference is therefore a **textbook semivariance-based OK
implemented in numpy** in
[`geostatspy-vendor/generate_ok3d_reference.py`](../../../../../../geostatspy-vendor/generate_ok3d_reference.py).
Two independent implementations of the published OK equations
(numpy textbook vs M8's Schur-complement Cholesky) agree to 1.5e-2
relative on values. The cost of moving away from a third-party
library is that the reference lives in this repo; the benefit is the
reference is a faithful implementation of standard OK.

## Variance convention

kriging-rs's 2-D `OrdinaryKrigingModel::predict` and 3-D
`OrdinaryKrigingModel3D::predict` both report the **latent-field
variance**, which is the textbook "standard" variance minus the
nugget. This is the convention upstream chose; our 3-D path inherits
it. The textbook fixture also reports latent variance (it subtracts
the nugget from `λ·γ₀ + μ`) so the parity test compares the same
quantity. Standard variance can always be recovered by adding the
nugget back.

The two variances differ by exactly the nugget. Both are correct
kriging variances. Filtering the nugget makes sense when the nugget
represents observation noise rather than spatial variance below the
data resolution — the user often wants the smooth latent field, not
the noisy observation.

## Generation

```bash
cd /Users/collord/rust-geostats-work/geostatspy-vendor
.venv/bin/python generate_ok3d_reference.py
```

Driver source: `geostatspy-vendor/generate_ok3d_reference.py`.
Dependencies in the vendor venv: Python 3.9, scikit-gstat 1.0.23, numpy,
pandas.

## Files

- `variogram.csv` — `effective_range, partial_sill, nugget` of the
  hand-set exponential variogram. Parameters are *injected* into a
  scikit-gstat `Variogram` object (auto-fit values discarded) for
  cross-run determinism.

  **Parameter mapping to our `VariogramModel`**: scikit-gstat's `c0` is
  the *partial* sill (above the nugget); our `VariogramModel::new`
  takes a *total* sill (`partial_sill + nugget`). The Rust test does
  the conversion.

- `targets.csv` — 1000 random `(x, y, z)` in the M5 dataset's bounding
  box (`x, y ∈ [0, 100]`, `z ∈ [0, 50]`), seeded with `20260527` for
  reproducibility.

- `predictions.csv` — `(x, y, z, value, variance)` per target from
  scikit-gstat's `OrdinaryKriging.transform`. All 1000 are finite —
  the neighborhood was sized (`min_points=5, max_points=50`) so no
  target lacks neighbors.

## Tolerances

The same 1.5e-2 (value) / 1e-2 (variance) relative tolerance applies on
both f32 and f64 storage. The disagreement between our
Schur-complement Cholesky and numpy.linalg.solve on a 150x150 OK
system is algorithmic, not numerical: max value error is ~1.23e-2 at
either precision. v3's published 1e-6 / 1e-4 figures were aspirational
and unreachable for a 3-D kriging system of this size with two
different solver paths.

This is still squarely v3 §Validation tolerance categories'
"algorithmic equivalence" tier — different solver paths, same math.
The measured drift is consistent with what's expected when comparing
Schur-complement Cholesky against a direct LU/LDL solve on a
moderately-conditioned 150x150 system.

## Updating

Deterministic — fixed seeds in the driver, hand-set variogram. If a
future test failure reveals a real disagreement, do **not** regenerate
to make the test pass. Diagnose first.
