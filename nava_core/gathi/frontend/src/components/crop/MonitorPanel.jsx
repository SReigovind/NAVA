import React, { useEffect, useRef, useState } from "react";
import PlantSelector from "./PlantSelector.jsx";
import { apiFetch } from "../../lib/api.js";

function getVnirTier(status) {
  const l = status?.toLowerCase() || "";
  if (l.includes("critical") || l.includes("severe")) return "critical";
  if (l.includes("warning") || l.includes("stress"))  return "warning";
  if (l.includes("calibrat")) return "calibrating";
  return "ok";
}

const TIER_META = {
  critical:    { label: "CRITICAL STRESS",  color: "#ef4444", track: "#7f1d1d", css: "vnir-critical" },
  warning:     { label: "STRESS DETECTED",  color: "#f97316", track: "#7c2d12", css: "vnir-warning"  },
  calibrating: { label: "CALIBRATING",      color: "#3b82f6", track: "#1e3a5f", css: "vnir-calibrate"},
  ok:          { label: "HEALTHY",          color: "#10b981", track: "#064e3b", css: "vnir-ok"       },
};

function HistorySection({ plantId, onDeleted }) {
  const [events, setEvents] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState(null);

  const formatLocalTime = (dateStr) => {
    if (!dateStr) return "";
    const d = new Date(dateStr.replace(" ", "T") + "Z");
    if (isNaN(d)) return dateStr.slice(0, 16);
    const pad = (n) => n.toString().padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };

  const load = async () => {
    setLoading(true);
    try {
      const data = await apiFetch(`/api/events?plant_id=${plantId}&limit=30`);
      setEvents((data.events || []).filter(e => e.event_type === "vnir"));
    } catch { } finally { setLoading(false); }
  };

  const deleteEvent = async (id) => {
    if (!confirm("Delete this monitoring entry?")) return;
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
        📋 VNIR History {open ? "▲" : "▼"}
      </button>
      {open && (
        <div className="history-list">
          {loading && <div className="text-sm text-muted">Loading…</div>}
          {!loading && events.length === 0 && <div className="text-sm text-muted">No VNIR history yet.</div>}
          {events.map(e => {
            const p = e.payload || {};
            const ok = p.status?.toLowerCase().includes("ok") || p.status?.toLowerCase().includes("calibrat");
            return (
              <div key={e.id} className="history-item">
                <div className={`health-dot ${ok ? "dot-green" : p.status ? "dot-red" : "dot-blue"}`} style={{ flexShrink: 0 }} />
                <div className="history-item-body">
                  <span className="history-item-label">{p.status}</span>
                  <span className="history-item-meta">ratio={p.ratio?.toFixed(4)} · {p.leaf_state} · {formatLocalTime(e.created_at)}</span>
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

export default function MonitorPanel({ fieldId, cropId }) {
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
      const data = await apiFetch("/api/vnir-upload", { method: "POST", body: form });
      setResult(data);
      setHistoryKey(k => k + 1);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleClear = async () => {
    if (!plant || !confirm(`Clear all VNIR history for "${plant.name}"?`)) return;
    setClearing(true);
    try { await apiFetch(`/api/plants/${plant.id}/history?event_type=vnir`, { method: "DELETE" }); setResult(null); setHistoryKey(k => k + 1); }
    catch (e) { setError(e.message); } finally { setClearing(false); }
  };

  const reset = () => { setPlant(null); setResult(null); setError(""); fileRef.current = null; setSelectedFileName(""); };

  if (!plant) return <PlantSelector cropId={cropId} onSelect={setPlant} />;

  const tier = result ? getVnirTier(result.status) : null;
  const meta = tier ? TIER_META[tier] : null;
  const isCal = tier === "calibrating";

  const DeltaRow = ({ label, val }) => {
    if (val == null) return null;
    const sign = val > 0 ? "+" : "";
    const bad  = Math.abs(val) > 5;
    return (
      <div className="vnir-delta-row">
        <span className="vnir-delta-label">{label}</span>
        <span className="vnir-delta-val" style={{ color: bad ? "#f87171" : "var(--text-secondary)" }}>
          {sign}{val.toFixed(1)}%
        </span>
      </div>
    );
  };

  return (
    <div className="tool-panel">
      <div className="tool-panel-header">
        <div className="row row-gap">
          <button className="btn btn-ghost btn-sm" onClick={reset}>← All Plants</button>
          <div className="plant-pill">🪴 {plant.name}</div>
          {plant.description && <span className="text-sm text-muted">{plant.description}</span>}
        </div>
        <button className="btn btn-ghost btn-sm danger-hover" onClick={handleClear} disabled={clearing}>
          {clearing ? "Clearing…" : "🗑 Clear All"}
        </button>
      </div>

      {error && <div className="notice notice-danger">{error}</div>}

      <div className="notice notice-info" style={{ fontSize: "0.8125rem" }}>
        <strong>Precautionary monitoring.</strong> ≥5 scans build a stress baseline. Supplements disease detection — does not replace it.
      </div>

      <div className="upload-bar">
        <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
          onChange={e => handleFile(e.target.files[0])} />
        <button className="btn btn-secondary" onClick={() => inputRef.current?.click()}>📂 Choose Image</button>
        {selectedFileName && <span className="file-name-pill">📄 {selectedFileName}</span>}
        <button className="btn btn-primary" onClick={handleRun} disabled={loading || !fileRef.current} style={{ marginLeft: "auto" }}>
          {loading ? <><span className="spinner-sm" style={{ marginRight: 8 }} />Analysing…</> : "Run VNIR Analysis"}
        </button>
      </div>

      {loading && (
        <div className="result-row">
          <div className="result-skeleton-col" />
          <div className="result-skeleton-col" />
          <div className="result-skeleton-col" />
        </div>
      )}

      {result && meta && (
        <div className="result-row">
          {/* ── Col 1: All text/stats ── */}
          <div className={`result-col vnir-status-col ${meta.css}`}>
            <div className="vnir-severity-bar" style={{ background: `linear-gradient(90deg, ${meta.color}cc, ${meta.color}55)` }} />

            <div className="vnir-body">
              <div className="vnir-status-tag">{meta.label}</div>
              <div className="vnir-status-text">{result.status}</div>

              {/* Raw measurements */}
              <div className="vnir-section-title">Measurements</div>
              <div className="vnir-metrics">
                <div className="vnir-metric-item">
                  <div className="vnir-metric-val">{result.ratio?.toFixed(4)}</div>
                  <div className="vnir-metric-key">VNIR Ratio</div>
                </div>
                <div className="vnir-metric-item">
                  <div className="vnir-metric-val">{result.avg_green?.toFixed(1)}</div>
                  <div className="vnir-metric-key">Avg Green</div>
                </div>
                <div className="vnir-metric-item">
                  <div className="vnir-metric-val">{result.avg_vnir?.toFixed(1)}</div>
                  <div className="vnir-metric-key">Avg VNIR</div>
                </div>
              </div>

              {result.leaf_state && (
                <div className="vnir-leaf-state">
                  Leaf state: <span>{result.leaf_state}</span>
                </div>
              )}

              {/* Deltas vs references */}
              {!isCal && (result.vs_baseline != null || result.vs_global != null) && (
                <>
                  <div className="vnir-section-title" style={{ marginTop: 12 }}>vs. Reference</div>
                  <DeltaRow label="Baseline"   val={result.vs_baseline} />
                  <DeltaRow label="Global avg" val={result.vs_global} />
                  <DeltaRow label="Rolling avg" val={result.vs_rolling} />
                  <DeltaRow label="Checkpoint"  val={result.vs_prev_checkpoint} />
                </>
              )}

              {isCal && (
                <div className="vnir-cal-note">
                  Baseline building — scan at least 5 images to enable stress comparisons.
                </div>
              )}
            </div>
          </div>

          {/* ── Col 2: HSV image ── */}
          {result.hsv_image_base64 ? (
            <div className="result-col result-img-col">
              <div className="result-image-label">HSV Analysis</div>
              <img src={result.hsv_image_base64} alt="HSV" className="result-img-compact" />
            </div>
          ) : <div className="result-col" />}

          {/* ── Col 3: VNIR stress map ── */}
          {result.vnir_image_base64 ? (
            <div className="result-col result-img-col">
              <div className="result-image-label">Stress Map</div>
              <img src={result.vnir_image_base64} alt="VNIR" className="result-img-compact" />
              <div className="result-img-caption">VNIR-derived stress index</div>
            </div>
          ) : <div className="result-col" />}
        </div>
      )}

      <HistorySection key={historyKey} plantId={plant.id} onDeleted={() => setHistoryKey(k => k + 1)} />
    </div>
  );
}
