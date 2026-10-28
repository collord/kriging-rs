import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// kriging-rs-wasm is a file: dependency that resolves outside this folder; the dev
// server must be allowed to serve its pkg/*.wasm.
const wasmPackage = fileURLToPath(new URL("../../npm/kriging-rs-wasm", import.meta.url));

export default defineConfig({
  server: { fs: { allow: [".", wasmPackage] } },
  // The kriging worker is an ES module (it imports the package).
  worker: { format: "es" },
  build: { target: "es2022" },
});
