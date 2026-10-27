# Textbook ordinary kriging reference fixture (M9)

1000 deterministic predict-location/expected-prediction pairs from a
**textbook semivariance-form OK** implementation in numpy on the same
150-point M5 dataset, with a hand-set exponential variogram. Consumed
by [`tests/parity_3d/textbook_kriging_reference.rs`](../../textbook_kriging_reference.rs).

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
  We initially tried it as the reference and its predictions disagreed
  by ~3% in value and ~40% in variance; not a meaningful
  algorithmic-tier reference.

The reference is therefore a **textbook semivariance-based OK in
numpy**, in [`../generate_textbook_kriging.py`](../generate_textbook_kriging.py).

## Nugget and variance convention

`γ(0) = 0`; for `h > 0`, `γ(h) = nugget + partial_sill · (1 − exp(−3h/a))`.
The nugget is the jump just past the origin, so it sits on the
kriging-matrix diagonal. The reported variance is the standard kriging
variance `λᵀγ₀ + μ`, nugget included — the same quantity kriging-rs
reports.

> **History.** The original fixture (generated outside the repo)
> evaluated `γ(0) = nugget` on the diagonal and then subtracted the
> nugget from the variance, documenting this as a "latent-field"
> convention. That reproduced a kriging-rs bug rather than a convention:
> with the nugget missing from the diagonal, the weights ignore it and
> the variance *falls* as the nugget rises. It also used all 150 samples
> while the Rust test applies `within_radius(a)`, which is where the
> ~1.2e-2 value disagreement previously attributed to "different solver
> paths" came from. Both are fixed; the in-repo generator reproduces the
> old fixture to machine precision when given the old convention and
> neighborhood.

## Generation

```bash
python tests/parity_3d/fixtures/generate_textbook_kriging.py
```

Regenerates the OK, SK and UK fixtures together; numpy is the only
dependency.

## Files

- `variogram.csv` — `effective_range, partial_sill, nugget` of the
  hand-set exponential variogram.

  **Parameter mapping to our `VariogramModel`**: `partial_sill` is the
  sill above the nugget; our `VariogramModel::new` takes a *total* sill
  (`partial_sill + nugget`). The Rust test does the conversion.

- `targets.csv` — 1000 random `(x, y, z)` in the M5 dataset's bounding
  box (`x, y ∈ [0, 100]`, `z ∈ [0, 50]`), seeded with `20260527` for
  reproducibility.

- `predictions.csv` — `(x, y, z, value, variance)` per target, using
  every sample within `effective_range` of the target (matching
  `Neighborhood3D::within_radius`).

## Tolerances

Both sides solve the same equations on the same neighborhood, so they
differ only by rounding: measured maximum relative error is ~6e-7 with
`f32` storage and ~5e-15 with `f64`. The test uses 1e-5 and 1e-10
respectively.

## Updating

Deterministic — fixed seeds, hand-set variogram. If a future test
failure reveals a real disagreement, do **not** regenerate to make the
test pass. Diagnose first.
