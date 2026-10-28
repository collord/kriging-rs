"""Regenerate the textbook 3-D kriging reference fixtures (OK, SK, UK).

Writes ``predictions.csv`` in ``textbook_ok3d/``, ``textbook_sk3d/`` and
``textbook_uk3d/`` from the shared 150-point dataset in ``skgstat_3d/`` and
each fixture's ``targets.csv`` / ``variogram.csv``. Only numpy is required:

    python tests/parity_3d/fixtures/generate_textbook_kriging.py

The equations are the published ones (Isaaks & Srivastava 1989 ch. 12;
Chilès & Delfiner 2012 §3.4), solved densely in float64:

- ``γ(0) = 0``; for ``h > 0``, ``γ(h) = nugget + partial_sill · (1 − exp(−3h/a))``.
  The nugget is the jump just past the origin, so it sits on the
  kriging-matrix diagonal (``C(0) = nugget + partial_sill``).
- Reported variance is the standard kriging variance of the estimate,
  nugget included.
- OK uses every sample within ``a`` of the target (matching
  ``Neighborhood3D::within_radius``); SK and UK use all samples.
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def exponential_gamma(h, nugget, partial_sill, effective_range):
    g = nugget + partial_sill * (1.0 - np.exp(-3.0 * h / effective_range))
    return np.where(h > 0.0, g, 0.0)


def ordinary(xs, zs, target, gamma, radius):
    h0 = np.linalg.norm(xs - target, axis=1)
    keep = h0 <= radius
    xs, zs, h0 = xs[keep], zs[keep], h0[keep]
    n = len(zs)
    a = np.ones((n + 1, n + 1))
    a[n, n] = 0.0
    a[:n, :n] = gamma(np.linalg.norm(xs[:, None] - xs[None], axis=2))
    b = np.append(gamma(h0), 1.0)
    sol = np.linalg.solve(a, b)
    return sol[:n] @ zs, sol @ b


def simple(xs, zs, target, gamma, sill, mean):
    def cov(h):
        return sill - gamma(h)

    k = cov(np.linalg.norm(xs[:, None] - xs[None], axis=2))
    k0 = cov(np.linalg.norm(xs - target, axis=1))
    lam = np.linalg.solve(k, k0)
    return mean + lam @ (zs - mean), sill - lam @ k0


def universal_linear(xs, zs, target, gamma):
    n = len(zs)
    f = np.column_stack([np.ones(n), xs])
    f0 = np.concatenate([[1.0], target])
    a = np.zeros((n + 4, n + 4))
    a[:n, :n] = gamma(np.linalg.norm(xs[:, None] - xs[None], axis=2))
    a[:n, n:] = f
    a[n:, :n] = f.T
    b = np.concatenate([gamma(np.linalg.norm(xs - target, axis=1)), f0])
    sol = np.linalg.solve(a, b)
    return sol[:n] @ zs, sol @ b


def predict_all(name, predict):
    targets = load_csv(HERE / name / "targets.csv")
    rows = [(*t, *predict(t)) for t in targets]
    out = HERE / name / "predictions.csv"
    with out.open("w", newline="\n") as fh:
        fh.write("x,y,z,value,variance\n")
        for row in rows:
            fh.write(",".join(repr(float(v)) for v in row) + "\n")
    print(f"wrote {len(rows)} rows to {out.relative_to(HERE)}")


def main():
    samples = load_csv(HERE / "skgstat_3d" / "samples.csv")
    xs, zs = samples[:, :3], samples[:, 3]

    a, ps, nug = load_csv(HERE / "textbook_ok3d" / "variogram.csv")[0]
    gamma = lambda h: exponential_gamma(h, nug, ps, a)  # noqa: E731
    predict_all("textbook_ok3d", lambda t: ordinary(xs, zs, t, gamma, a))

    a, ps, nug, mean = load_csv(HERE / "textbook_sk3d" / "variogram.csv")[0]
    gamma = lambda h: exponential_gamma(h, nug, ps, a)  # noqa: E731
    predict_all("textbook_sk3d", lambda t: simple(xs, zs, t, gamma, nug + ps, mean))

    a, ps, nug = load_csv(HERE / "textbook_uk3d" / "variogram.csv")[0]
    gamma = lambda h: exponential_gamma(h, nug, ps, a)  # noqa: E731
    predict_all("textbook_uk3d", lambda t: universal_linear(xs, zs, t, gamma))


if __name__ == "__main__":
    main()
