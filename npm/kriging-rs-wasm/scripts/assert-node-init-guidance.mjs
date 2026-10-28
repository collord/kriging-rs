/**
 * Run in a separate process (init state is per-process). Asserts that init() with no
 * bytes under Node fails with a not_loaded KrigingError naming the `kriging-rs-wasm/wasm`
 * export, and that init() succeeds with the bytes that export resolves to.
 * Exit code 0 on success; non-zero otherwise. Uses process.exitCode rather than
 * process.exit(): exiting while the failed fetch still holds handles trips a libuv
 * assertion on Windows.
 */
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import * as kriging from "../dist/index.js";

async function main() {
  try {
    await kriging.init();
    return "init() without bytes unexpectedly succeeded under Node";
  } catch (err) {
    if (err?.name !== "KrigingError" || err?.code !== "not_loaded") {
      return `unexpected error: ${err}`;
    }
    if (!String(err.message).includes("kriging-rs-wasm/wasm")) {
      return `error does not say how to load the binary: ${err.message}`;
    }
  }
  const wasmPath = fileURLToPath(import.meta.resolve("kriging-rs-wasm/wasm"));
  await kriging.init(await readFile(wasmPath));
  const fit = kriging.fitVariogramProjected({
    xs: [0, 1, 2, 3, 4],
    ys: [0, 1, 0, 1, 0],
    values: [1, 2, 1.5, 2.5, 2],
    variogramType: "exponential",
  });
  return Number.isFinite(fit.range)
    ? null
    : `non-finite fitted range: ${fit.range}`;
}

const problem = await main();
if (problem) {
  console.error(problem);
  process.exitCode = 1;
}
