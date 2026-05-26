# scikit-gstat reference fixture for 3-D omnidirectional variogram

A 150-point synthetic 3-D dataset and its omnidirectional Matheron
semivariogram as computed by [scikit-gstat](https://scikit-gstat.readthedocs.io/)
1.0.23. Used by [`tests/parity_3d/skgstat_reference.rs`](../../skgstat_reference.rs)
at M5 to validate that
`kriging_rs::variogram::compute_empirical_variogram_3d` reproduces the
same per-bin semivariance and pair count.

## Why scikit-gstat and not GSLib / geostatspy

The v3 fork-scope doc named geostatspy as the reference for 3-D variogram
parity. As of geostatspy 0.0.79, `gamv_3D` is **broken** — it calls a
`variogram_loop_3D` function that doesn't exist in the module. pygslib
(the f2py-wrapped Fortran path) has no prebuilt wheels for Python 3.9 on
macOS and requires substantial Fortran build setup. scikit-gstat is a
well-maintained pure-Python library that natively supports n-dimensional
scattered-data variograms with the Matheron estimator and is widely
cited in academic geostatistics.

`setrot.f` parity (M3) is still done against GSLib's Fortran via a
fixture-only build in `/Users/collord/rust-geostats-work/gslib-vendor/`,
since `setrot`'s ~30 lines are easy to vendor. The fuller variogram
machinery is not worth that detour at this milestone.

## Source code

`/Users/collord/rust-geostats-work/geostatspy-vendor/generate_omnidirectional_reference.py`.
Run from the vendor venv:

```bash
cd /Users/collord/rust-geostats-work/geostatspy-vendor
.venv/bin/python generate_omnidirectional_reference.py
```

Dependencies pinned at fixture-generation time:
- Python 3.9.6
- scikit-gstat 1.0.23
- numpy, pandas (latest stable)

## Files

- `samples.csv` — `x, y, z, value` for 150 points. Generated from
  `numpy.random.default_rng(20260526)` with `(x, y)` in `[0, 100]`, `z`
  in `[0, 50]`. Values follow a smooth field
  `10 + 5 sin(x/30) + 3 cos(y/25) + 0.5 z` plus σ=0.3 Gaussian noise.
- `bin_edges.csv` — `bin_index, lower_edge, upper_edge` for the 10
  equal-width bins skgstat constructed. Bin width is `maxlag / n_lags`
  but with `maxlag` possibly truncated by skgstat to the 99th percentile
  of distances; consumers of this fixture should **use these exact
  edges**, not regenerate them from `(maxlag, n_lags)`.
- `omnidirectional.csv` — `bin_index, semivariance, n_pairs` from
  scikit-gstat's `Variogram(..., estimator='matheron', n_lags=10,
  maxlag=120, normalize=False)`.

## Tolerances

Pair counts: **exact** (set-identity tier). For the same dataset and the
same equal-width bin edges, both implementations are doing the identical
"for each pair, bin by floor(d / bin_width)" operation; any disagreement
is a bug.

Semivariances: **1e-9 / 1e-5** (numerical-equivalence tier per v3
§Validation tolerance categories, depending on storage precision).
Matheron's formula `γ̂(h) = 0.5 · mean((z_i - z_j)²)` has no
algorithmic ambiguity; same pairs → same value to the precision floor of
the floating-point accumulator.

## Regenerating

The fixture is deterministic (fixed seed). If the test starts failing
because skgstat's bin-construction logic changes in a future release,
regenerate the fixture and document the version bump.
