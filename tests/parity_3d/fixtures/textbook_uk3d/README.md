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
- Variance (standard, semivariance form): `λᵀ γ₀ + μᵀ f₀`.
- Variance (latent, our convention): standard − nugget.

The 4 Lagrangian rows enforce trend unbiasedness on the kriging
weights (one constraint per basis function): `Σ λᵢ f_k(xᵢ) = f_k(x₀)`
for each `k`.

## Files

- `variogram.csv` — `effective_range, partial_sill, nugget`.
- `targets.csv` — 1000 random `(x, y, z)` from
  `numpy.random.default_rng(20260527)`.
- `predictions.csv` — `(x, y, z, value, variance)` from the textbook
  UK reference.

## Tolerance

2e-2 relative on both value and variance. UK has structurally more
drift than OK because it adds 4 Lagrangians plus a 4×4 trend system
on top of the same Cholesky path; the extra back-substitutions
accumulate slightly more rounding. Measured ~1.5e-2 in practice with
2e-2 leaving a small safety margin.

## Generation

```bash
cd /Users/collord/rust-geostats-work/geostatspy-vendor
.venv/bin/python generate_uk3d_reference.py
```
