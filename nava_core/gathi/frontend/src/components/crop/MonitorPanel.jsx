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

/**
 * Returns a message descriptor for leaf states where NIR analysis is not
 * meaningful (no leaf found, or leaf is visibly yellow/brown).
 * Returns null for normal GREEN leaf scans.
 */
function getInvalidLeafMessage(leafState, status) {
  if (leafState === "NONE") {
    return {
      icon: "🔍",
      title: "No Leaf Detected",
      body: "The camera could not isolate a leaf in this image. The HSV colour analyser looks for green or yellow-brown regions and found neither above the minimum area threshold.",
      tips: [
        "Make sure the leaf fills most of the frame.",
        "Use even, natural lighting — avoid heavy shadows or direct glare.",
        "Avoid busy or colourful backgrounds; plain soil or sky works best.",
        "Retake the photo closer to the leaf surface.",
      ],
      accent: "#6366f1",
    };
  }
  if (leafState === "YELLOW_BROWN") {
    return {
      icon: "🍂",
      title: "Severe Visual Stress Detected",
      body: "The leaf appears predominantly yellow or brown. At this stage the tissue has already undergone visible degradation, and the NIR reflectance model cannot produce reliable readings. Stress monitoring values are not shown.",
      tips: [
        "Photograph a still-green section of the plant for NIR analysis.",
        "Use the Disease Detection tab to identify the likely pathogen.",
        "Consult NAVA Chat for treatment recommendations.",
        "Consider clearing VNIR history and restarting the baseline once the plant recovers.",
      ],
      accent: "#f97316",
    };
  }
  return null;
}

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

function VnirCautionBlock() {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="vnir-caution-block">
      <div className="vnir-caution-header">
        <span className="vnir-caution-icon">⚠️</span>
        <span className="vnir-caution-summary">
          Stress monitoring · requires <strong>5+ healthy baseline scans</strong> to activate
        </span>
        <button className="vnir-caution-toggle" onClick={() => setOpen(v => !v)}>
          {open ? "Hide details ▲" : "How it works ▼"}
        </button>
      </div>

      {open && (
        <div className="vnir-caution-details">
          <div className="vnir-caution-item">
            <span className="vnir-caution-bullet">📸</span>
            <span><strong>Minimum 5 photos needed</strong> before stress comparisons activate. Until then the system is in calibration mode and cannot flag stress.</span>
          </div>
          <div className="vnir-caution-item">
            <span className="vnir-caution-bullet">🌱</span>
            <span><strong>First 5 photos must be from a healthy plant.</strong> The system uses these as its baseline reference — if early photos are taken during disease or stress, comparisons will be inaccurate.</span>
          </div>
          <div className="vnir-caution-item">
            <span className="vnir-caution-bullet">🕐</span>
            <span><strong>Take photos at a consistent time each day</strong> (e.g. always morning or always midday). Lighting and leaf water content change through the day and affect NIR reflectance.</span>
          </div>
          <div className="vnir-caution-item">
            <span className="vnir-caution-bullet">🔄</span>
            <span><strong>Clear monitoring data monthly</strong> or when the plant enters a new growth stage. NIR reflectance patterns change naturally as the plant matures — an old baseline will produce false stress alerts.</span>
          </div>
          <div className="vnir-caution-item">
            <span className="vnir-caution-bullet">⚗️</span>
            <span><strong>This monitoring system is experimental.</strong> Results are <em>proactive warnings only</em> — always visually inspect the plant and cross-check with the Disease Detection tab before drawing any conclusions.</span>
          </div>
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

      <VnirCautionBlock />

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

      {result && (() => {
        const invalidMsg = getInvalidLeafMessage(result.leaf_state, result.status);

        // ── Invalid leaf state: no-leaf or yellow/brown ──────────────────────
        if (invalidMsg) {
          return (
            <div className="result-row" style={{ alignItems: "flex-start" }}>
              {/* Message card */}
              <div className="result-col" style={{ flex: 2, minWidth: 0 }}>
                <div className="vnir-invalid-card" style={{ borderLeft: `4px solid ${invalidMsg.accent}` }}>
                  <div className="vnir-invalid-header">
                    <span className="vnir-invalid-icon">{invalidMsg.icon}</span>
                    <span className="vnir-invalid-title" style={{ color: invalidMsg.accent }}>{invalidMsg.title}</span>
                  </div>
                  <p className="vnir-invalid-body">{invalidMsg.body}</p>
                  <div className="vnir-invalid-tips-label">What to do:</div>
                  <ul className="vnir-invalid-tips">
                    {invalidMsg.tips.map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                </div>
              </div>

              {/* Still show HSV image so the user sees what the camera captured */}
              {result.hsv_image_base64 ? (
                <div className="result-col result-img-col">
                  <div className="result-image-label">HSV Analysis</div>
                  <img src={result.hsv_image_base64} alt="HSV" className="result-img-compact" />
                  <div className="result-img-caption" style={{ color: invalidMsg.accent }}>
                    {result.leaf_state === "NONE" ? "No leaf region isolated" : "Yellow/brown tissue detected"}
                  </div>
                </div>
              ) : <div className="result-col" />}

              <div className="result-col" />
            </div>
          );
        }

        // ── Normal result: GREEN leaf with valid NIR readings ─────────────────
        const tier = getVnirTier(result.status);
        const meta = TIER_META[tier];
        const isCal = tier === "calibrating";

        return (
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
        );
      })()}

      <HistorySection key={historyKey} plantId={plant.id} onDeleted={() => setHistoryKey(k => k + 1)} />
    </div>
  );
}
