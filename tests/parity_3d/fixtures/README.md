# Parity test fixtures

3D test data and pre-computed reference outputs for parity tests.

## Layout

- `walker_lake_3d.csv` or `stanford_vi_subset.csv` — the shared 3D dataset all parity tests run against. **Not yet populated.** Acquired at M5 when first parity test lands.
- `gamv_outputs/` — pre-computed GSLib `gamv` outputs (pair indices per bin, gamma values per bin) for the directional parity tests in M6.
- `kt3d_outputs/` — pre-computed GSLib `kt3d` predictions at 1000 fixed locations for the algorithmic-tier parity tests in M9/M10.

## How fixtures are generated

Each subdirectory will contain a `README.md` and a `generate.sh` (or `generate.py`) that documents the exact command used to produce the reference output, including software version, parameter file contents, and any random seeds. Reproducibility of the fixture itself matters as much as reproducibility of the kriging-rs output it validates against.
