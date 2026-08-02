/**
 * Per-family variogram math backgrounders shown by the "Variogram model" help icon.
 *
 * Each entry carries the semivariance formula γ(h), a one-line description, the parameter
 * legend, and citations to the standard literature. Formulas use the "practical range"
 * convention (the range `a` is where ~95% of the sill is reached) for the exponential and
 * Gaussian families, matching the engine's parameterization.
 */

// Reusable citations (stable DOI / open-access links).
const CD2012 = {
  text: "Chilès & Delfiner (2012), Geostatistics: Modeling Spatial Uncertainty, 2nd ed., Wiley",
  url: "https://doi.org/10.1002/9781118136188",
};
const MATHERON1963 = {
  text: "Matheron (1963), Principles of Geostatistics, Economic Geology 58(8):1246–1266",
  url: "https://doi.org/10.2113/gsecongeo.58.8.1246",
};
const STEIN1999 = {
  text: "Stein (1999), Interpolation of Spatial Data: Some Theory for Kriging, Springer",
  url: "https://doi.org/10.1007/978-1-4612-1494-6",
};
const RW2006 = {
  text: "Rasmussen & Williams (2006), Gaussian Processes for Machine Learning, Ch. 4 (open access)",
  url: "https://gaussianprocess.org/gpml/",
};
const MA_BHADRA2023 = {
  text: "Ma & Bhadra (2023), Beyond Matérn: On a Class of Interpretable Confluent Hypergeometric Covariance Functions, JASA 118(543):2045–2058",
  url: "https://doi.org/10.1080/01621459.2022.2027775",
};

/** Shared parameter legend fragments. */
const BASE_PARAMS = "c₀ = nugget, s = sill, a = range, h = lag distance";

const VARIOGRAM_MATH = {
  spherical: {
    label: "Spherical",
    formula: "γ(h) = c₀ + (s − c₀)·[ 1.5·(h/a) − 0.5·(h/a)³ ],  0 < h < a\nγ(h) = s,  h ≥ a",
    note: "Reaches the sill exactly at the range a, then stays flat. A common default with a clean finite range.",
    params: BASE_PARAMS,
    citations: [MATHERON1963, CD2012],
  },
  exponential: {
    label: "Exponential",
    formula: "γ(h) = c₀ + (s − c₀)·[ 1 − exp(−3·h/a) ]",
    note: "Approaches the sill asymptotically; the factor 3 makes a a “practical range” where ≈95% of the sill is reached. Rough near the origin (linear behavior).",
    params: BASE_PARAMS,
    citations: [CD2012],
  },
  gaussian: {
    label: "Gaussian",
    formula: "γ(h) = c₀ + (s − c₀)·[ 1 − exp(−3·h²/a²) ]",
    note: "Parabolic near the origin ⇒ very smooth fields. Can make the kriging system ill-conditioned; a small nugget helps. Uses the practical-range (factor 3) convention.",
    params: BASE_PARAMS,
    citations: [CD2012, RW2006],
  },
  cubic: {
    label: "Cubic",
    formula:
      "γ(h) = c₀ + (s − c₀)·[ 7(h/a)² − 8.75(h/a)³ + 3.5(h/a)⁵ − 0.75(h/a)⁷ ],  h < a\nγ(h) = s,  h ≥ a",
    note: "Smooth like the Gaussian but with a finite range: reaches the sill exactly at a.",
    params: BASE_PARAMS,
    citations: [CD2012],
  },
  stable: {
    label: "Stable (powered exponential)",
    formula: "γ(h) = c₀ + (s − c₀)·[ 1 − exp(−(h/a)^α) ],  0 < α ≤ 2",
    note: "The exponent α tunes short-scale smoothness: α = 1 recovers the exponential, α = 2 the Gaussian shape.",
    params: `${BASE_PARAMS}, α = shape (smoothness)`,
    citations: [CD2012],
  },
  matern: {
    label: "Matérn",
    formula:
      "γ(h) = c₀ + (s − c₀)·[ 1 − (2^{1−ν} / Γ(ν))·xᵛ·K_ν(x) ],  x = √(2ν)·h/a",
    note: "ν sets the smoothness continuously: ν = 0.5 gives the exponential, ν → ∞ the Gaussian. Kᵥ is the modified Bessel function of the second kind, Γ the gamma function.",
    params: `${BASE_PARAMS}, ν = smoothness`,
    citations: [STEIN1999, RW2006],
  },
  confluenthypergeometric: {
    label: "Confluent hypergeometric",
    formula:
      "γ(h) = c₀ + (s − c₀)·[ 1 − ρ(h) ]\nρ(h) = (Γ(ν+α)/Γ(ν))·U(α, 1−ν, ½·(h/a)²)",
    note: "A two-parameter generalization of Matérn: ν sets short-scale smoothness, α the tail decay. As α → ∞ it recovers Matérn, but for finite α the tails are polynomial (long-range dependence). U is the confluent hypergeometric function of the second kind.",
    params: `${BASE_PARAMS}, ν = smoothness, α = tail decay`,
    citations: [MA_BHADRA2023, STEIN1999],
  },
};

/**
 * Renders the math backgrounder for one variogram family.
 *
 * @param {object} props
 * @param {string} props.model Model name (any case; e.g. "Matern", "exponential").
 */
export function VariogramMathHelp({ model }) {
  const entry = VARIOGRAM_MATH[String(model ?? "").toLowerCase()];
  if (!entry) {
    return (
      <p>
        The variogram γ(h) describes how similarity between points decays with distance. Pick a
        family, and its nugget, sill, and range are fitted to your data.
      </p>
    );
  }
  return (
    <>
      <p>
        The variogram γ(h) describes how dissimilarity grows with lag distance h. This is the{" "}
        <strong>{entry.label}</strong> family:
      </p>
      <code className="help-formula">{entry.formula}</code>
      <p>{entry.note}</p>
      <p className="help-params">{entry.params}</p>
      <div className="help-cite">
        <div className="help-cite-label">References</div>
        <ul>
          {entry.citations.map((c) => (
            <li key={c.url}>
              <a href={c.url} target="_blank" rel="noreferrer noopener">
                {c.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

export default VARIOGRAM_MATH;
