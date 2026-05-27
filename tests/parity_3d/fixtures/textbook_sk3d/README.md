# Textbook simple kriging reference fixture (M10)

1000 deterministic predict-location/expected-prediction pairs from a
**textbook semivariance-form simple kriging** implementation in numpy
on the same 150-point M5 dataset, with the same hand-set exponential
variogram as the M9 OK fixture.

SK assumes a known global mean. The fixture uses the **empirical mean
of the sample values** (`mean = 25.010480`) so it's deterministic and
reproducible.

See [`../textbook_ok3d/README.md`](../textbook_ok3d/README.md) for the
chain of reasoning on why we use textbook implementations rather than
scikit-gstat / pygslib / geostatspy.

## SK math

System: `K λ = k₀` (no Lagrangian).

- `K[i, j] = C(d(i, j))` for i ≠ j; `K[i, i] = C(0)` (the partial sill
  under our latent-field convention).
- `k₀[i] = C(d(target, i))`.
- Predicted: `m + λ · (z − m)`.
- Variance (latent): `partial_sill − λ · k₀`.

## Files

- `variogram.csv` — `effective_range, partial_sill, nugget, mean`.
  Mean is the empirical sample-value mean.
- `targets.csv` — 1000 random `(x, y, z)` from
  `numpy.random.default_rng(20260527)`.
- `predictions.csv` — `(x, y, z, value, variance)` from the textbook
  SK reference.

## Tolerance

1.5e-2 relative on values, 1e-2 on variance — same as M9 OK because the
two solvers share the same Cholesky path; SK is structurally simpler
(no Schur complement, no μ) so drift is in the same envelope.

## Generation

```bash
cd /Users/collord/rust-geostats-work/geostatspy-vendor
.venv/bin/python generate_sk3d_reference.py
```
