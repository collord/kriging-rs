# `setrot` reference fixture

30 hand-curated cases of `(ang1, ang2, ang3, anis1, anis2)` paired with the
3×3 deformation matrix produced by GSLib's `dsetrot` subroutine
(double-precision variant of `setrot.f`).

Consumed by [`tests/parity_3d/gslib_roundtrip.rs`](../../gslib_roundtrip.rs)
at M3 to validate that `interop::gslib_anisotropy::from_gslib` reproduces
GSLib's matrix exactly (within tolerance — see below) and that
`to_gslib(from_gslib(p)) == p` round-trips within 1e-12.

## Source code

Built from [pygslib](https://github.com/opengeostat/pygslib) commit on
`master` (`for_code/setrot.f90`), which is itself an f2py-wrapped Fortran
90 port of the original 1996 GSLib v.2 `setrot.f`. The semantics of
`setrot` have not changed across GSLib versions; the choice of port is
not load-bearing as long as the conventions match the published GSLib
manual (Deutsch & Journel 1997).

The Fortran source and the driver that produced this CSV live in
`/Users/collord/rust-geostats-work/gslib-vendor/` (outside the fork —
the fork doesn't ship Fortran).

## How `cases.csv` was generated

```bash
cd /Users/collord/rust-geostats-work/gslib-vendor
gfortran -O0 -ffree-line-length-none \
    setrot.f90 setrot_reference_driver.f90 \
    -o setrot_reference_driver
./setrot_reference_driver > path/to/cases.csv
```

Toolchain: `gfortran` (Homebrew GCC 15.2.0_1) on Darwin 24.4.0
(arm64-binary running under Rosetta; emits x86_64 code path).

## Output format

Each row is one test case. Columns:

- `ang1, ang2, ang3, anis1, anis2` — input parameters (GSLib conventions:
  azimuth clockwise from north, dip positive downward, plunge,
  range_minor/range_major, range_vertical/range_major).
- `m11, m12, m13, m21, m22, m23, m31, m32, m33` — entries of the
  deformation matrix in row-major order. **Not** a pure rotation:
  rows 2 and 3 are pre-multiplied by `1/anis1` and `1/anis2`
  respectively, so the matrix encodes both rotation and anisotropic
  stretching in a single transform.

## Tolerance ceiling

GSLib's `setrot` uses `DEG2RAD = 3.141592654 / 180.0` — a deliberately
truncated π with only 9 significant digits. This applies even in the
double-precision `dsetrot` variant (see `setrot.f90:191`). As a
consequence:

- The fixture's `(0, 0, 0, 1, 1)` "identity" case is not literal identity;
  the off-diagonal residual is ~1.22e-8, which is `cos(π/2)` evaluated
  against GSLib's truncated π.
- **Matrix-parity tests should not use tolerances tighter than ~1e-7** when
  comparing GSLib output against any implementation that uses a
  full-precision π. Looser is fine; 1e-12 is unachievable by design.

The round-trip test (`from_gslib(to_gslib(p)) == p`) is unaffected by this
and uses the v3-mandated 1e-12 tolerance, because it never goes through
the truncated-π path.

## Curation strategy

The 30 cases span:

| Group | Cases | Purpose |
|---|---|---|
| Identity / near-identity | 1-3 | Sanity checks |
| Pure azimuth rotations | 4-8 | Including the `>= 270°` branch boundary |
| Pure dip | 9-12 | Mild, steep, near-vertical, upward |
| Pure plunge | 13-14 | Third-angle only |
| Axis-aligned anisotropy | 15-18 | Including pathological ratios (`0.05`, `0.02`) |
| Realistic geological cases | 19-23 | Plunging veins, tilted strata, etc. |
| Three-angle compositions | 24-27 | Stress the angle composition order |
| Branch boundaries | 28-30 | `ang1` near 270° |

If a future bug-fix or feature addition needs more cases, edit
`setrot_reference_driver.f90` and re-run; do **not** hand-edit the CSV.
