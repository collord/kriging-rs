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

- `K[i, j] = C(d(i, j))` for i ≠ j; `K[i, i] = C(0)`, the total sill
  `partial_sill + nugget` (`γ(0) = 0`, so the nugget is on the diagonal).
- `k₀[i] = C(d(target, i))`.
- Predicted: `m + λ · (z − m)`.
- Variance (standard, nugget included): `C(0) − λ · k₀`.

## Files

- `variogram.csv` — `effective_range, partial_sill, nugget, mean`.
  Mean is the empirical sample-value mean.
- `targets.csv` — 1000 random `(x, y, z)` from
  `numpy.random.default_rng(20260527)`.
- `predictions.csv` — `(x, y, z, value, variance)` from the textbook
  SK reference, using all 150 samples.

## Tolerance

Both sides solve the same equations, so they differ only by rounding:
measured maximum relative error is ~6e-7 with `f32` storage and ~5e-15
with `f64`. The test uses 1e-5 and 1e-10 respectively.

## Generation

```bash
python tests/parity_3d/fixtures/generate_textbook_kriging.py
```

Regenerates the OK, SK and UK fixtures together; see
[`../textbook_ok3d/README.md`](../textbook_ok3d/README.md#nugget-and-variance-convention) for why
the original fixture was replaced.
