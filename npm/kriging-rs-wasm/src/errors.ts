/**
 * Error types and helpers for kriging-rs-wasm.
 *
 * @module
 */

import type { KrigingErrorCode } from "./types.js";

/**
 * Error thrown by the library when WASM operations fail (invalid inputs, model build failure, etc.).
 * The underlying cause is attached as `cause` when available. Optional `code` is a stable string
 * for UI-friendly messages (e.g. "singular_covariance", "mismatched_arrays").
 */
export class KrigingError extends Error {
  /** Stable code for this error type, when known; use for UI messages or branching. */
  readonly code?: KrigingErrorCode;

  constructor(
    message: string,
    options?: { cause?: unknown; code?: KrigingErrorCode }
  ) {
    super(message);
    this.name = "KrigingError";
    if (options?.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
    if (options?.code !== undefined) {
      this.code = options.code;
    }
  }
}

const KNOWN_ERROR_CODES: readonly KrigingErrorCode[] = [
  "not_loaded",
  "model_freed",
  "mismatched_arrays",
  "invalid_variogram",
  "invalid_bins",
  "singular_covariance",
  "too_few_points",
  "unknown_variogram",
  "invalid_input",
  "backend_unavailable",
  "internal_error",
  "insufficient_data",
  "callback_aborted",
];

/**
 * Extract an explicit stable error code from a thrown value. Rust bindings attach a `code`
 * property (stable string) via `coded_err`, so the TS layer can surface it directly instead
 * of string-matching the message.
 */
function extractErrorCode(value: unknown): KrigingErrorCode | undefined {
  if (typeof value === "object" && value !== null) {
    const code = (value as { code?: unknown }).code;
    if (
      typeof code === "string" &&
      (KNOWN_ERROR_CODES as readonly string[]).includes(code)
    ) {
      return code as KrigingErrorCode;
    }
  }
  return undefined;
}

function extractErrorMessage(value: unknown): string {
  if (value instanceof Error) return value.message;
  if (typeof value === "object" && value !== null) {
    const msg = (value as { message?: unknown }).message;
    if (typeof msg === "string") return msg;
  }
  return String(value);
}

/**
 * Wrap a thrown value in {@link KrigingError}, preserving the Rust-attached error code when
 * present. Falls back to passing the bare message through without a code.
 */
export function wrapThrown(value: unknown): KrigingError {
  const msg = extractErrorMessage(value);
  const code = extractErrorCode(value);
  return new KrigingError(msg, { cause: value, code });
}
