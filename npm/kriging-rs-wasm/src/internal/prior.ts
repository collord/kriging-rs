/**
 * Internal: helpers for normalizing the binomial Beta prior shape used by CV and SGS.
 *
 * The public API accepts an optional {@link BinomialPriorParams} object; the WASM
 * layer historically takes two `number | undefined` scalars. This helper keeps the
 * boundary conversion in one place so each call site just unpacks `{alpha, beta}`.
 *
 * @module
 */

import type { BinomialPriorParams } from "../types.js";

/** Default Beta(α, β) used when the caller omits a prior — matches Rust's `BinomialPrior::default()` (Beta(1, 1)). */
export const DEFAULT_BINOMIAL_PRIOR: Readonly<BinomialPriorParams> =
  Object.freeze({
    alpha: 1,
    beta: 1,
  });

/**
 * Resolve a user-supplied prior object to the positional `(alpha, beta)` form the
 * WASM functions expect. Returns `undefined` for both fields when `prior` is
 * omitted so the Rust side falls back to its default Beta(1, 1).
 */
export function resolveBinomialPrior(prior: BinomialPriorParams | undefined): {
  alpha: number | undefined;
  beta: number | undefined;
} {
  return { alpha: prior?.alpha, beta: prior?.beta };
}

/**
 * Resolve a user-supplied prior to a concrete `{ alpha, beta }` pair, falling
 * back to {@link DEFAULT_BINOMIAL_PRIOR}. Used when the TS side needs to
 * compute the same EB-smoothed logits the Rust kriger consumes (e.g. so the
 * variogram fit and the kriging system see the same field).
 */
export function resolveBinomialPriorOrDefault(
  prior: BinomialPriorParams | undefined
): BinomialPriorParams {
  return prior ?? DEFAULT_BINOMIAL_PRIOR;
}

/**
 * Empirical-Bayes smoothed logit `logit((s + α) / (n + α + β))` with the supplied
 * Beta prior. Mirrors `BinomialObservation::smoothed_logit_with_prior` in the
 * Rust crate so callers can pre-compute the same logit values the kriger sees.
 *
 * Stations with `n === 0` are pulled all the way to the prior mean; the result
 * is always finite as long as `α + β > 0`.
 */
export function smoothedLogit(
  successes: number,
  trials: number,
  prior: BinomialPriorParams = DEFAULT_BINOMIAL_PRIOR
): number {
  const { alpha, beta } = prior;
  const p = (successes + alpha) / (trials + alpha + beta);
  return Math.log(p / (1 - p));
}

/**
 * Vectorized {@link smoothedLogit} over equal-length count arrays. Returns a
 * `Float64Array` so the result can be handed straight to {@link fitVariogram}.
 */
export function smoothedLogits(
  successes: ArrayLike<number>,
  trials: ArrayLike<number>,
  prior: BinomialPriorParams = DEFAULT_BINOMIAL_PRIOR
): Float64Array {
  const n = successes.length;
  if (trials.length !== n) {
    throw new Error(
      `smoothedLogits: successes (${n}) and trials (${trials.length}) must have the same length`
    );
  }
  const out = new Float64Array(n);
  const { alpha, beta } = prior;
  const denomBase = alpha + beta;
  for (let i = 0; i < n; i++) {
    const p = (successes[i] + alpha) / (trials[i] + denomBase);
    out[i] = Math.log(p / (1 - p));
  }
  return out;
}
