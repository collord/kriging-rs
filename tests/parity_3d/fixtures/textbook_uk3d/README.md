# Textbook universal kriging (linear trend) reference fixture (M10)

1000 deterministic predict-location/expected-prediction pairs from a
**textbook semivariance-form universal kriging** implementation in
numpy with the linear trend basis `[1, x, y, z]`. Same 150-point M5
dataset and same hand-set exponential variogram as M9 OK.

See [`../textbook_ok3d/README.md`](../textbook_ok3d/README.md) for the
chain of reasoning on why we use textbook implementations rather than
scikit-gstat / pygslib / geostatspy.

## UK math (linear trend)

System: `(n+4) × (n+4)` bordered by the trend basis:

```text
  [ Γ   F ] [λ]   [γ₀]
  [ Fᵀ  0 ] [μ] = [f₀]
```

- `F[i, :] = [1, x_i, y_i, z_i]` for the linear basis.
- `f₀ = [1, x_target, y_target, z_target]`.
- Predicted: `λᵀ z`.
- `Γ[i, i] = γ(0) = 0`; the nugget enters every off-diagonal and `γ₀` entry.
- Variance (standard, semivariance form, nugget included): `λᵀ γ₀ + μᵀ f₀`.

The 4 Lagrangian rows enforce trend unbiasedness on the kriging
weights (one constraint per basis function): `Σ λᵢ f_k(xᵢ) = f_k(x₀)`
for each `k`.

## Files

- `variogram.csv` — `effective_range, partial_sill, nugget`.
- `targets.csv` — 1000 random `(x, y, z)` from
  `numpy.random.default_rng(20260527)`.
- `predictions.csv` — `(x, y, z, value, variance)` from the textbook
  UK reference, using all 150 samples.

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
