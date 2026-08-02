import { useState, useEffect } from "react";
import init, { webgpuAvailable } from "kriging-rs-wasm";
import ErrorBanner from "./components/ErrorBanner";
import HelpTip from "./components/HelpTip";
import QuickDemos from "./components/QuickDemos";
import SurfaceDemo from "./components/SurfaceDemo";
import CompareView from "./components/CompareView";
import DataUpload from "./components/DataUpload";

export default function App() {
  const [wasmReady, setWasmReady] = useState(false);
  const [webgpuStatus, setWebgpuStatus] = useState("detecting…");
  const [error, setError] = useState(null);
  const [uploadedData, setUploadedData] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await init();
        if (cancelled) return;
        setWasmReady(true);
        try {
          const ok = await webgpuAvailable();
          setWebgpuStatus(ok ? "available" : "unavailable");
        } catch {
          setWebgpuStatus("unavailable");
        }
      } catch (e) {
        if (!cancelled) {
          setError(e?.message ?? String(e));
          setWebgpuStatus("init failed");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!wasmReady) {
    return (
      <div className="panel">
        <div className="loading">
          <span className="spinner" aria-hidden />
          <span>Loading WebAssembly module…</span>
        </div>
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      </div>
    );
  }

  return (
    <>
      <h1>kriging-rs Web Demo</h1>
      <p>
        This demo uses the <code>kriging-rs-wasm</code> package to run ordinary and
        binomial kriging in the browser. Upload your own CSV or use synthetic data.
      </p>

      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      <div className="panel">
        <p>
          <strong>WebGPU:</strong> {webgpuStatus} — use “Auto” in 2D Surface to prefer
          GPU when available.
          <HelpTip topic="webgpu" label="WebGPU status" />
        </p>
      </div>

      <DataUpload onUpload={setUploadedData} uploadedData={uploadedData} />

      <QuickDemos onError={setError} />

      <SurfaceDemo
        uploadedData={uploadedData}
        onError={setError}
        webgpuStatus={webgpuStatus}
      />

      <CompareView onError={setError} webgpuAvailable={webgpuStatus === "available"} />

      <footer style={{ marginTop: "2rem", fontSize: "0.85rem", color: "var(--text-muted)" }}>
        kriging-rs — spatial interpolation with ordinary and binomial kriging.
      </footer>
    </>
  );
}
