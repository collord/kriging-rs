import { useState, useRef } from "react";
import HelpTip from "./HelpTip";

/**
 * Parse CSV for kriging.
 * Expected columns (case-insensitive): lat, lon, and either "value" (ordinary) or "successes" and "trials" (binomial).
 */
function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return null;
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const latIdx = header.indexOf("lat");
  const lonIdx = header.indexOf("lon");
  const valueIdx = header.indexOf("value");
  const successIdx = header.indexOf("successes");
  const trialsIdx = header.indexOf("trials");
  if (latIdx === -1 || lonIdx === -1) return null;
  const hasValue = valueIdx >= 0;
  const hasBinomial = successIdx >= 0 && trialsIdx >= 0;
  if (!hasValue && !hasBinomial) return null;

  const lats = [];
  const lons = [];
  const values = hasValue ? [] : null;
  const successes = hasBinomial ? [] : null;
  const trials = hasBinomial ? [] : null;

  for (let i = 1; i < lines.length; i += 1) {
    const cells = lines[i].split(",").map((c) => c.trim());
    const lat = Number(cells[latIdx]);
    const lon = Number(cells[lonIdx]);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    lats.push(lat);
    lons.push(lon);
    if (hasValue) values.push(Number(cells[valueIdx]));
    if (hasBinomial) {
      successes.push(Math.max(0, Math.floor(Number(cells[successIdx]) || 0)));
      trials.push(Math.max(1, Math.floor(Number(cells[trialsIdx]) || 1)));
    }
  }

  if (lats.length < 3) return null;
  if (hasValue && values.some((v) => !Number.isFinite(v))) return null;
  if (hasBinomial) {
    for (let j = 0; j < successes.length; j += 1) {
      if (successes[j] > trials[j]) successes[j] = trials[j];
    }
  }

  return {
    lats,
    lons,
    values: values ?? undefined,
    successes: successes ?? undefined,
    trials: trials ?? undefined,
    mode: hasValue ? "ordinary" : "binomial",
  };
}

export default function DataUpload({ onUpload, uploadedData }) {
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef(null);

  function handleFile(e) {
    const file = e.target.files?.[0];
    setError(null);
    setFileName(null);
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseCsv(reader.result);
        if (!parsed) {
          setError(
            "CSV must have headers: lat, lon, and either value (ordinary) or successes and trials (binomial).",
          );
          onUpload?.(null);
          return;
        }
        setFileName(file.name);
        onUpload?.(parsed);
      } catch (err) {
        setError(err?.message ?? "Failed to parse CSV");
        onUpload?.(null);
      }
    };
    reader.readAsText(file);
  }

  function clearUpload() {
    setError(null);
    setFileName(null);
    onUpload?.(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="panel">
      <h2>
        Data
        <HelpTip topic="dataSection" label="Data" />
      </h2>
      <p>
        Upload a CSV with columns <code>lat</code>, <code>lon</code>, and either{" "}
        <code>value</code> (ordinary kriging) or <code>successes</code> and{" "}
        <code>trials</code> (binomial kriging). If no file is uploaded, the 2D Surface
        uses synthetic data.
      </p>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", flexWrap: "wrap" }}>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleFile}
          aria-label="Upload CSV"
        />
        <HelpTip topic="csvFormat" label="CSV format" />
        {uploadedData && (
          <button type="button" onClick={clearUpload}>
            Clear upload
          </button>
        )}
      </div>
      {fileName && (
        <p style={{ fontSize: "0.9rem", marginTop: "0.5rem", marginBottom: 0 }}>
          Using <strong>{fileName}</strong> ({uploadedData.mode}, {uploadedData.lats.length}{" "}
          points)
        </p>
      )}
      {error && (
        <p style={{ color: "var(--error-border)", fontSize: "0.9rem", marginTop: "0.5rem" }}>
          {error}
        </p>
      )}
    </div>
  );
}
