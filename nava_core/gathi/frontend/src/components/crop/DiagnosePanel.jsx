import React, { useRef, useState, useEffect } from "react";
import PlantSelector from "./PlantSelector.jsx";
import { apiFetch } from "../../lib/api.js";

function HistorySection({ plantId, onDeleted }) {
  const [events, setEvents] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await apiFetch(`/api/events?plant_id=${plantId}&limit=30`);
      setEvents((data.events || []).filter(e => e.event_type === "diagnose"));
    } catch { } finally { setLoading(false); }
  };

  const deleteEvent = async (id) => {
    if (!confirm("Delete this detection entry?")) return;
    setDeleting(id);
    try {
      await apiFetch(`/api/events/${id}`, { method: "DELETE" });
      setEvents(prev => prev.filter(e => e.id !== id));
      onDeleted?.();
    } catch { } finally { setDeleting(null); }
  };

  useEffect(() => { if (open) load(); }, [open, plantId]);

  return (
    <div className="history-section">
      <button className="history-toggle" onClick={() => setOpen(v => !v)}>
        📋 Detection History {open ? "▲" : "▼"}
      </button>
      {open && (
        <div className="history-list">
          {loading && <div className="text-sm text-muted">Loading…</div>}
          {!loading && events.length === 0 && <div className="text-sm text-muted">No history yet.</div>}
          {events.map(e => {
            const p = e.payload || {};
            const healthy = p.class_label?.toLowerCase().includes("healthy");
            const conf = p.confidence != null ? Math.round(p.confidence * 100) : null;
            return (
              <div key={e.id} className="history-item">
                <div className={`health-dot ${healthy ? "dot-green" : "dot-red"}`} style={{ flexShrink: 0 }} />
                <div className="history-item-body">
                  <span className="history-item-label">{p.class_label?.replace(/_/g, " ")}</span>
                  <span className="history-item-meta">{conf != null && `${conf}% · `}{p.reliability} · {(e.created_at || "").slice(0, 16)}</span>
                </div>
                <button className="history-del-btn" onClick={() => deleteEvent(e.id)} disabled={deleting === e.id}>×</button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function DiagnosePanel({ fieldId, cropId }) {
  const [plant, setPlant] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [clearing, setClearing] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [historyKey, setHistoryKey] = useState(0);
  const inputRef = useRef();
  const fileRef = useRef();

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    fileRef.current = file; setSelectedFileName(file.name); setResult(null); setError("");
  };

  const handleRun = async () => {
    if (!fileRef.current || !plant) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const form = new FormData();
      form.append("image", fileRef.current);
      form.append("plant_id", plant.id);
      form.append("crop_id", cropId);
      form.append("field_id", fieldId);
      const data = await apiFetch("/api/diagnose", { method: "POST", body: form });
      setResult(data);
      setHistoryKey(k => k + 1);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const clearHistory = async () => {
    if (!plant || !confirm(`Clear all detection history for "${plant.name}"?`)) return;
    setClearing(true);
    try { await apiFetch(`/api/plants/${plant.id}/history?event_type=diagnose`, { method: "DELETE" }); setResult(null); setHistoryKey(k => k + 1); }
    catch (e) { setError(e.message); } finally { setClearing(false); }
  };

  const reset = () => { setPlant(null); setResult(null); setError(""); fileRef.current = null; setSelectedFileName(""); };

  if (!plant) return <PlantSelector cropId={cropId} onSelect={setPlant} />;

  const confPct = result?.confidence != null ? Math.round(result.confidence * 100) : null;
  const isHealthy = result?.class_label?.toLowerCase().includes("healthy");
  const isReliable = result?.reliability === "RELIABLE";

  // Professional label: "tomato_late_blight" → "Late Blight", "tomato_healthy" → "Healthy"
  const formatLabel = (raw) => {
    if (!raw) return "—";
    return raw.replace(/^[a-z]+_/, "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  };
  const formatCrop = (raw) => {
    if (!raw) return "";
    const parts = raw.split("_");
    return parts[0].charAt(0).toUpperCase() + parts[0].slice(1);
  };

  return (
    <div className="tool-panel">
      <div className="tool-panel-header">
        <div className="row row-gap">
          <button className="btn btn-ghost btn-sm" onClick={reset}>← All Plants</button>
          <div className="plant-pill">🪴 {plant.name}</div>
        </div>
        <button className="btn btn-ghost btn-sm danger-hover" onClick={clearHistory} disabled={clearing}>
          {clearing ? "Clearing…" : "🗑 Clear All"}
        </button>
      </div>

      {error && <div className="notice notice-danger">{error}</div>}

      <div className="upload-bar">
        <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
          onChange={e => handleFile(e.target.files[0])} />
        <button className="btn btn-secondary" onClick={() => inputRef.current?.click()}>📂 Choose Leaf Image</button>
        {selectedFileName && <span className="file-name-pill">📄 {selectedFileName}</span>}
        <button className="btn btn-primary" onClick={handleRun} disabled={loading || !fileRef.current} style={{ marginLeft: "auto" }}>
          {loading ? <><span className="spinner-sm" style={{ marginRight: 8 }} />Analysing…</> : "Run Detection"}
        </button>
      </div>

      {loading && (
        <div className="result-row">
          <div className="result-skeleton-col" />
          <div className="result-skeleton-col" />
          <div className="result-skeleton-col" />
        </div>
      )}

      {result && (
        <>
          <div className="result-row">
            {/* ── Col 1: Professional diagnosis card ── */}
            <div className={`result-col diag-status-col ${isHealthy ? "diag-healthy" : "diag-disease"}`}>
              {/* Severity indicator bar at top */}
              <div className="diag-severity-bar" style={{
                background: isHealthy
                  ? "linear-gradient(90deg, #10b981, #34d399)"
                  : "linear-gradient(90deg, #dc2626, #f97316)"
              }} />

              <div className="diag-body">
                <div className="diag-status-tag">{isHealthy ? "HEALTHY" : "DISEASE DETECTED"}</div>

                <div className="diag-label">{formatLabel(result.class_label)}</div>
                {result.class_label && !isHealthy && (
                  <div className="diag-crop-tag">{formatCrop(result.class_label)}</div>
                )}

                {/* Confidence gauge */}
                <div className="diag-conf-block">
                  <div className="diag-conf-label">
                    <span>Model Confidence</span>
                    <span className="diag-conf-pct">{confPct}%</span>
                  </div>
                  <div className="diag-conf-track">
                    <div className="diag-conf-fill" style={{
                      width: `${confPct}%`,
                      background: confPct >= 80
                        ? (isHealthy ? "#10b981" : "#ef4444")
                        : "#f59e0b"
                    }} />
                  </div>
                </div>

                {/* Reliability chip */}
                <div className="diag-chip-row">
                  <span className={`diag-chip ${isReliable ? "chip-reliable" : "chip-unreliable"}`}>
                    {isReliable ? "● Reliable" : "● Low confidence"}
                  </span>
                  {!isHealthy && isReliable && (
                    <span className="diag-chip chip-action">Action needed</span>
                  )}
                </div>
              </div>
            </div>

            {/* ── Col 2: Original image ── */}
            {result.original_image_base64 ? (
              <div className="result-col result-img-col">
                <div className="result-image-label">Original Image</div>
                <img src={result.original_image_base64} alt="Original" className="result-img-compact" />
              </div>
            ) : <div className="result-col" />}

            {/* ── Col 3: Grad-CAM ── */}
            {result.gradcam_image_base64 ? (
              <div className="result-col result-img-col">
                <div className="result-image-label">Attention Map</div>
                <img src={result.gradcam_image_base64} alt="GradCAM" className="result-img-compact" />
                <div className="result-img-caption">Model focus region (Grad-CAM)</div>
              </div>
            ) : <div className="result-col" />}
          </div>

          {!isHealthy && isReliable && (
            <div className="notice notice-warning">
              <strong>Recommendation:</strong> Seek agronomist assessment for treatment options.
            </div>
          )}
        </>
      )}

      <HistorySection key={historyKey} plantId={plant.id} onDeleted={() => setHistoryKey(k => k + 1)} />
    </div>
  );
}
