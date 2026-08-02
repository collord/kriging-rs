/**
 * Projected (planar) binomial kriging on `(x, y)` coordinates with 2D anisotropy.
 *
 * The planar / anisotropic counterpart of {@link BinomialKriging}: count data
 * (`successes`, `trials`) is mapped to a smoothed logit scale (default
 * Beta(1, 1) prior, matching Rust) and ordinary kriging runs on those logits using
 * Euclidean (optionally anisotropy-deformed) distances. Predictions are
 * back-transformed to prevalences via the logistic function with a delta-method
 * variance approximation. See {@link BinomialKriging} for the geographic
 * (Haversine) flavor.
 *
 * @module
 */
import { KrigingError, wrapThrown } from "../errors.js";
import { toFloat64Array, toUint32Array } from "../internal/convert.js";
import {
  mapBinomialBatchArrayOutput,
  mapBinomialBuildNotes,
  mapBinomialPrediction,
  mapBinomialPredictionArray,
} from "../internal/mappers.js";
import { requireLoadedModule } from "../internal/module.js";
import type { WasmBinomialProjectedInstance } from "../internal/wasm-shapes.js";
import type {
  BinomialBatchArrayOutput,
  BinomialBuildNotes,
  BinomialPrediction,
  BinomialProjectedFromPrecomputedLogitsOptions,
  BinomialProjectedKrigingOptions,
  BinomialProjectedKrigingWithPriorOptions,
  NumericArrayInput,
} from "../types.js";

const PROJECTED_BINOMIAL_FREED =
  "BinomialProjectedKriging model has been freed";

/**
 * Projected / planar binomial kriging on `(x, y)` coordinates with 2-D anisotropy.
 * Fits ordinary kriging on the (smoothed) logit scale and back-transforms via
 * the logistic; suitable for prevalence mapping when the data are already
 * projected (e.g. meters in a local equal-area projection) and a directional
 * range matters.
 */
export class BinomialProjectedKriging {
  private inner: WasmBinomialProjectedInstance | null;

  /**
   * Build a projected binomial kriging model with default Beta(1, 1) prior.
   * Use {@link BinomialProjectedKriging.newWithPrior} for an explicit
   * Beta prior, or {@link BinomialProjectedKriging.fromPrecomputedLogits} when
   * the caller already has finite logit estimates.
   *
   * @throws {KrigingError} When the WASM module is not loaded, when inputs are
   *   invalid (mismatched array lengths, `trials[i] === 0`,
   *   `rangeRatio ∉ (0, 1]`, …), or when the WASM package was built without
   *   this class (code `backend_unavailable`).
   */
  constructor(options: BinomialProjectedKrigingOptions) {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    try {
      this.inner = ctor.fromArrays(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toUint32Array(options.successes),
        toUint32Array(options.trials),
        options.variogram,
        options.majorAngleDeg,
        options.rangeRatio
      );
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  private requireInner(): WasmBinomialProjectedInstance {
    if (this.inner === null) {
      throw new KrigingError(PROJECTED_BINOMIAL_FREED, { code: "model_freed" });
    }
    return this.inner;
  }

  /**
   * Build a projected binomial kriging model with an explicit Beta(`alpha`, `beta`)
   * prior on prevalence. Useful when counts are small.
   */
  static newWithPrior(
    options: BinomialProjectedKrigingWithPriorOptions
  ): BinomialProjectedKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const instance = Object.create(
      BinomialProjectedKriging.prototype
    ) as BinomialProjectedKriging;
    try {
      (
        instance as unknown as {
          inner: WasmBinomialProjectedInstance | null;
        }
      ).inner = ctor.fromArraysWithPrior(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toUint32Array(options.successes),
        toUint32Array(options.trials),
        options.variogram,
        options.majorAngleDeg,
        options.rangeRatio,
        options.prior.alpha,
        options.prior.beta
      );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /**
   * Build a projected binomial kriging model from caller-supplied logit values
   * (bypasses the empirical-Bayes shrinkage). All `logits` must be finite.
   */
  static fromPrecomputedLogits(
    options: BinomialProjectedFromPrecomputedLogitsOptions
  ): BinomialProjectedKriging {
    const mod = requireLoadedModule();
    const ctor = mod.WasmBinomialProjectedKriging;
    if (!ctor) {
      throw new KrigingError(
        "BinomialProjectedKriging is not available; rebuild the WASM package",
        { code: "backend_unavailable" }
      );
    }
    const instance = Object.create(
      BinomialProjectedKriging.prototype
    ) as BinomialProjectedKriging;
    try {
      (
        instance as unknown as {
          inner: WasmBinomialProjectedInstance | null;
        }
      ).inner = ctor.fromPrecomputedLogits(
        toFloat64Array(options.xs),
        toFloat64Array(options.ys),
        toFloat64Array(options.logits),
        options.variogram,
        options.majorAngleDeg,
        options.rangeRatio
      );
    } catch (e) {
      throw wrapThrown(e);
    }
    return instance;
  }

  /** Release WASM-held resources. Safe to call multiple times. */
  free(): void {
    if (this.inner === null) return;
    if (typeof this.inner.free === "function") this.inner.free();
    this.inner = null;
  }

  /** Explicit-resource-management disposer; calls {@link free}. */
  [Symbol.dispose](): void {
    this.free();
  }

  /** Build-time diagnostics (prior, dropped rows, logit inflation, …). */
  getBuildNotes(): BinomialBuildNotes {
    try {
      return mapBinomialBuildNotes(this.requireInner().getBuildNotes());
    } catch (e) {
      throw wrapThrown(e);
    }
  }

  /** Single-point prevalence prediction at planar `(x, y)`. */
  predict(x: number, y: number): BinomialPrediction {
    return mapBinomialPrediction(this.requireInner().predict(x, y));
  }

  /** Batch prevalence prediction at multiple `(x, y)` pairs. */
  predictBatch(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): BinomialPrediction[] {
    const out = this.requireInner().predictBatch(
      toFloat64Array(xs),
      toFloat64Array(ys)
    );
    return mapBinomialPredictionArray(out);
  }

  /** Batch prevalence prediction returning typed arrays. Prefer for large grids. */
  predictBatchArrays(
    xs: NumericArrayInput,
    ys: NumericArrayInput
  ): BinomialBatchArrayOutput {
    const out = this.requireInner().predictBatchArrays(
      toFloat64Array(xs),
      toFloat64Array(ys)
    );
    return mapBinomialBatchArrayOutput(out);
  }
}
