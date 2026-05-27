# GSLib `gamv` reference fixture for 3-D directional variograms

Reference outputs from the GSLib `gamv` binary (vendored at
`gslib-vendor/gamv`) on the same 150-point dataset M5 uses for skgstat
parity. Consumed by
[`tests/parity_3d/gamv_reference.rs`](../../gamv_reference.rs) at M6.

## Why this exists alongside `skgstat_3d/`

M5 validated the **omnidirectional** Matheron variogram against
scikit-gstat (a well-maintained pure-Python implementation with native
n-dimensional support, equal-width binning). M6 validates the
**directional** variogram against the canonical GSLib reference
(`gamv`'s Fortran), which uses a different binning convention
(lag-centred) and a cone+bandwidth direction filter. The two fixtures
exercise distinct code paths and should not be conflated.

## Generation

```bash
cd /Users/collord/rust-geostats-work/geostatspy-vendor
.venv/bin/python ../gslib-vendor/generate_gamv_reference.py
```

The driver loads `samples.csv` from M5's skgstat_3d fixture, converts
to GSLib free-format `samples.gslib`, then runs `gamv` once per
direction in the table. Each run produces a `.par` file (gamv's
parameter input) and a `.out` file (gamv's text output) — both are
checked in alongside the parsed CSV so the full chain is reproducible.

GSLib version: `GAMV Version: 2.905` (binary in `gslib-vendor/`).
Driver toolchain: Python 3.9 venv at `geostatspy-vendor/.venv/` with
`pandas`.

## Files

- `samples.gslib` — GSLib free-format data file, same coords/values as
  `../skgstat_3d/samples.csv`.
- `directions.csv` — table of 4 directions with `(name, azm, atol,
  bandh, dip, dtol, bandv, xlag, xltol, nlag)`:
    - `omni`: atol=dtol=90 deg with infinite bandwidth → all pairs
      accepted.
    - `azim_north`: 22.5 deg cone north, 25-unit horizontal bandwidth.
    - `azim_east`: 22.5 deg cone east, 25-unit horizontal bandwidth.
    - `vertical`: 22.5 deg cone downward, 25-unit bandwidths.
- `<name>.par` — gamv parameter file (one per direction).
- `<name>.out` — raw gamv text output (one per direction).
- `dir_<name>.csv` — parsed `(lag_index, mean_distance, gamma, n_pairs)`.

## Quirks of gamv's output that matter for parity testing

1. **Lag indexing**: gamv writes `nlag + 2` rows. Row `lag_index=1` is
   the self-pair bin (`h <= EPSLON`), with `n_pairs = n_dataset` for
   non-omni directions and `2*n_dataset` for omni. Subsequent rows are
   centred at `(lag_index - 2) * xlag`. Our directional code starts
   from `j = i + 1` so we don't generate self-pairs; the parity test
   strips gamv's lag-1 row before comparing.

2. **Omni double-count**: when `atol >= 90 deg`, gamv accumulates each
   pair twice with reversed (tail, head) ordering
   (`variograms.f90:473-484`). For M6, the parity test uses only the
   three directional cases (atol = 22.5 deg) and skips omni; M5's
   skgstat fixture is the omni reference.

3. **Text-output precision**: gamv writes `gamma` with 5 decimal
   places (Fortran `f12.5` or similar). Parity tolerance for γ is
   `1e-4 relative` against both f32 and f64 storage — the fixture
   itself is the precision floor, not our floating-point arithmetic.

## Updating the fixture

The fixture is deterministic — the dataset uses a fixed seed and
gamv's output is reproducible. If the gamv binary or parameter table
changes, regenerate with the driver and document the bump here.
