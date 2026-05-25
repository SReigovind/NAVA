import React, { useEffect, useState } from "react";
import { apiFetch } from "../../lib/api.js";

const AUTO_NOTES_SEPARATOR = "--- NAVA Auto-notes ---";

/** Split raw notes field into { manual, auto } portions. */
export function splitNotes(raw = "") {
  if (!raw.includes(AUTO_NOTES_SEPARATOR)) return { manual: raw.trim(), auto: "" };
  const [before, ...rest] = raw.split(AUTO_NOTES_SEPARATOR);
  return { manual: before.trim(), auto: rest.join(AUTO_NOTES_SEPARATOR).trim() };
}

/**
 * AutoNotesIcon — a small 🤖 icon that lives directly in a notes-section title.
 * Always rendered (never returns null). Opens a read-only modal popup.
 *
 * Props:
 *   content  – the auto-generated text to display (can be empty string)
 *   label    – short description shown in the modal title ("crop" or "field")
 */
export function AutoNotesIcon({ content = "", label = "", onDeleteLine }) {
  const [open, setOpen] = useState(false);
  const lines = content ? content.split("\n").filter(l => l.trim()) : [];

  return (
    <>
      <button
        className="auto-notes-icon-btn"
        onClick={() => setOpen(true)}
        title="View what NAVA knows automatically about this section"
        aria-label="What NAVA knows"
      >
        🤖
      </button>

      {open && (
        <div className="auto-notes-overlay" onClick={() => setOpen(false)}>
          <div className="auto-notes-modal" onClick={e => e.stopPropagation()}>
            <div className="auto-notes-modal-header">
              <span>🤖 What NAVA knows — <em>{label}</em></span>
              <button className="auto-notes-close" onClick={() => setOpen(false)}>✕</button>
            </div>
            <p className="auto-notes-modal-sub">
              Automatically generated context used by NAVA during your conversations. Read-only.
            </p>
            {lines.length > 0 ? (
              <ul className="auto-notes-list">
                {lines.map((l, i) => (
                  <li key={i} className="auto-notes-item" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
                    <span>{l}</span>
                    {onDeleteLine && (
                      <button className="btn btn-ghost btn-sm" style={{ padding: "0 4px", color: "var(--danger-color, #ef4444)" }} onClick={() => onDeleteLine(i)} title="Delete this note">
                        ✕
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="auto-notes-empty">
                No automatic context recorded yet. NAVA will start adding relevant notes here as you have conversations about this {label}.
              </p>
            )}
          </div>
        </div>
      )}
    </>
  );
}

/** @deprecated Use AutoNotesIcon instead */
export function AutoNotesCard({ rawNotes = "", label = "" }) {
  const { auto } = splitNotes(rawNotes);
  return <AutoNotesIcon content={auto} label={label} />;
}

function StatCard({ icon, label, value, sub, subColor }) {
  return (
    <div className="ov-stat">
      <div className="ov-stat-icon">{icon}</div>
      <div className="ov-stat-body">
        <div className="ov-stat-value">{value ?? "—"}</div>
        <div className="ov-stat-label">{label}</div>
        {sub && <div className="ov-stat-sub" style={subColor ? { color: subColor } : {}}>{sub}</div>}
      </div>
    </div>
  );
}

function cleanLabel(label) {
  if (!label) return "";
  return label.replace(/^[a-z]+_/, "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function dotClass(status, type) {
  if (type === "diag") {
    if (!status) return { cls: "dot-gray",  tip: "Disease Detection: No scan yet" };
    const clean = cleanLabel(status);
    if (status.toLowerCase().includes("healthy")) return { cls: "dot-green", tip: `Disease Detection: All Clear (${clean})` };
    return { cls: "dot-red", tip: `Disease Detection: Concern Detected (${clean})` };
  }
  // vnir
  if (!status) return { cls: "dot-gray", tip: "VNIR Monitoring: No scan yet" };
  const l = status.toLowerCase();
  const clean = cleanLabel(l);
  if (l.includes("calibrat"))                       return { cls: "dot-blue",  tip: `VNIR Monitoring: Establishing Baseline` };
  if (l.includes("ok") || l.includes("healthy"))    return { cls: "dot-green", tip: `VNIR Monitoring: Healthy Level` };
  if (l.includes("warning") || l.includes("stress") || l.includes("critical")) return { cls: "dot-red", tip: `VNIR Monitoring: Stress Detected` };
  return { cls: "dot-gray", tip: `VNIR Monitoring: ${clean}` };
}

function PlantHealthRow({ plant, events }) {
  const pid = Number(plant.id);
  const diagEvents = events.filter(e => Number(e.plant_id) === pid && e.event_type === "diagnose");
  const vnirEvents = events.filter(e => Number(e.plant_id) === pid && e.event_type === "vnir");
  const latestDiagLabel = diagEvents[0]?.payload?.class_label || null;
  const latestVnirStatus = vnirEvents[0]?.payload?.status || null;

  const diagDot = dotClass(latestDiagLabel, "diag");
  const vnirDot = dotClass(latestVnirStatus, "vnir");

  const formatLabel = (raw) => raw ? raw.replace(/^[a-z]+_/, "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : null;

  return (
    <div className="ov-plant-row">
      <div className="ov-plant-info">
        <span className="ov-plant-name">{plant.name}</span>
        {plant.description && <span className="ov-plant-desc">{plant.description}</span>}
      </div>
      <div className="ov-plant-status" style={{ gap: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }} title={diagDot.tip}>
          <div className={`health-dot ${diagDot.cls}`} />
          <span className="text-xs text-muted" style={{ fontWeight: 500 }}>Disease Detection</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }} title={vnirDot.tip}>
          <div className={`health-dot ${vnirDot.cls}`} />
          <span className="text-xs text-muted" style={{ fontWeight: 500 }}>VNIR Monitoring</span>
        </div>
      </div>
    </div>
  );
}



export default function OverviewPanel({ fieldId, cropId, crop, field, onNavigate, onRefresh }) {
  const [plants, setPlants] = useState([]);
  const [events, setEvents] = useState([]);
  // Only initialise editable notes from the MANUAL portion (above the auto-notes separator)
  const [notes, setNotes] = useState(() => splitNotes(crop?.notes || "").manual);
  const [editingNotes, setEditingNotes] = useState(false);
  const [savingNotes, setSavingNotes] = useState(false);
  const [loading, setLoading] = useState(true);

  const formatLocalDate = (dateStr) => {
    if (!dateStr) return "";
    const d = new Date(dateStr.replace(" ", "T") + "Z");
    if (isNaN(d)) return dateStr.slice(0, 10);
    const pad = (n) => n.toString().padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
  };

  const load = async () => {
    setLoading(true);
    try {
      const [pData, eData] = await Promise.all([
        apiFetch(`/api/plants?crop_id=${cropId}`),
        apiFetch(`/api/events?crop_id=${cropId}&limit=100`),
      ]);
      setPlants(pData.plants || []);
      setEvents(eData.events || []);
    } catch { }
    finally { setLoading(false); }
  };

  useEffect(() => {
    load();
    setNotes(splitNotes(crop?.notes || "").manual);
  }, [cropId, crop]);

  const saveNotes = async () => {
    setSavingNotes(true);
    try {
      const { auto } = splitNotes(crop?.notes || "");
      const combined = auto ? `${notes.trim()}\n\n${AUTO_NOTES_SEPARATOR}\n${auto}`.trim() : notes.trim();
      await apiFetch("/api/crop-context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_id: Number(cropId), notes: combined }),
      });
      setEditingNotes(false);
      if (onRefresh) onRefresh();
    } catch { }
    finally { setSavingNotes(false); }
  };

  const diagCount = events.filter(e => e.event_type === "diagnose").length;
  const vnirCount = events.filter(e => e.event_type === "vnir").length;
  const diseaseEvents = events.filter(e =>
    e.event_type === "diagnose" && !e.payload?.class_label?.toLowerCase().includes("healthy")
  );
  const recentEvents = [...events].slice(0, 8);

  if (loading) return <div className="page-center" style={{ minHeight: 200 }}><div className="spinner" /></div>;

  return (
    <div className="ov-layout" style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
      {/* Stats row */}
      <div className="ov-stats">
        <StatCard icon="🌱" label="Plants Tracked" value={plants.length} />
        <StatCard icon="🔬" label="Disease Scans" value={diagCount}
          sub={diseaseEvents.length > 0 ? `⚠️ ${diseaseEvents.length} concern(s)` : diagCount > 0 ? "All clear" : "No scans yet"}
          subColor={diseaseEvents.length > 0 ? "#f87171" : undefined} />
        <StatCard icon="📡" label="VNIR Scans" value={vnirCount} sub="Stress monitoring" />
        <StatCard icon="🌾" label="Growth Stage" value={crop?.stage || "Unknown"} sub={crop?.season || ""} />
      </div>

      <div className="ov-main" style={{ flex: 1, minHeight: 0, alignItems: "stretch" }}>
        {/* Plant health */}
        <div className="ov-card" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          <div className="ov-card-header">
            <div>
              <h3 className="ov-card-title">Plant Health</h3>
            </div>
            <button className="btn btn-primary btn-sm" onClick={() => onNavigate("diagnose")}>+ Run Detection</button>
          </div>
          {plants.length === 0 ? (
            <div className="empty-state" style={{ padding: "20px 0" }}>
              <div className="icon" style={{ fontSize: "2rem" }}>🌱</div>
              <p>No plants tracked yet. Start from Disease Detection or Stress Monitor.</p>
            </div>
          ) : (
            <div className="ov-plant-list custom-scrollbar" style={{ flex: 1, minHeight: 0, overflowY: "auto", paddingRight: "8px" }}>
              {plants.map(p => <PlantHealthRow key={p.id} plant={p} events={events} />)}
            </div>
          )}
        </div>

        <div className="ov-side" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
          {/* Crop notes (manual, editable) */}
          <div className="ov-card">
            <div className="ov-card-header">
              <h3 className="ov-card-title">
                Crop Notes
                <AutoNotesIcon 
                  content={splitNotes(crop?.notes || "").auto} 
                  label="crop notes" 
                  onDeleteLine={async (index) => {
                    const parts = splitNotes(crop?.notes || "");
                    const lines = parts.auto.split("\n").filter(l => l.trim());
                    lines.splice(index, 1);
                    const newAuto = lines.join("\n");
                    const combined = newAuto ? `${parts.manual}\n\n${AUTO_NOTES_SEPARATOR}\n${newAuto}`.trim() : parts.manual;
                    await apiFetch("/api/crop-context", {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({ crop_id: Number(cropId), notes: combined }),
                    });
                    if (onRefresh) onRefresh();
                  }}
                />
              </h3>
              {!editingNotes && (
                <button className="btn btn-ghost btn-sm" onClick={() => setEditingNotes(true)}>
                  {notes ? "✏️ Edit" : "+ Add"}
                </button>
              )}
            </div>
            {editingNotes ? (
              <div className="stack stack-sm">
                <textarea className="textarea" value={notes}
                  onChange={e => setNotes(e.target.value)}
                  placeholder="Symptoms, treatments, observations…" rows={5} />
                <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                  <button className="btn btn-ghost btn-sm" onClick={() => { setEditingNotes(false); setNotes(splitNotes(crop?.notes || "").manual); }}>Cancel</button>
                  <button className="btn btn-primary btn-sm" onClick={saveNotes} disabled={savingNotes}>
                    {savingNotes ? "Saving…" : "Save"}
                  </button>
                </div>
              </div>
            ) : notes ? (
              <p className="text-sm" style={{ color: "var(--text-secondary)", whiteSpace: "pre-wrap", lineHeight: 1.7 }}>{notes}</p>
            ) : (
              <p className="text-sm text-muted">No notes yet. Add observations so NAVA gives better advice.</p>
            )}
          </div>

          {/* Recent activity */}
          <div className="ov-card" style={{ display: "flex", flexDirection: "column" }}>
            <div className="ov-card-header">
              <h3 className="ov-card-title">Recent Activity</h3>
              <button className="btn btn-ghost btn-sm" onClick={load} title="Refresh">↻</button>
            </div>
            {recentEvents.length === 0 ? (
              <p className="text-sm text-muted">No activity yet.</p>
            ) : (
              <div className="ov-activity-list custom-scrollbar"
                style={{ maxHeight: "220px", overflowY: "auto", paddingRight: "8px" }}>
                {recentEvents.map((e, i) => {
                  const isDiag = e.event_type === "diagnose";
                  const label = isDiag ? e.payload?.class_label?.replace(/_/g, " ") : e.payload?.status;
                  const l = label?.toLowerCase() || "";
                  
                  let dotCls = "dot-gray";
                  if (isDiag) {
                    if (l.includes("healthy")) dotCls = "dot-green";
                    else if (l) dotCls = "dot-red";
                  } else {
                    if (l.includes("calibrat")) dotCls = "dot-blue";
                    else if (l.includes("ok") || l.includes("healthy")) dotCls = "dot-green";
                    else if (l.includes("warning") || l.includes("stress") || l.includes("critical")) dotCls = "dot-red";
                  }

                  return (
                    <div key={i} className="ov-activity-row">
                      <div className="ov-activity-body">
                        <span className="ov-activity-label">{isDiag ? "Disease Detection" : "VNIR Monitoring"}</span>
                        <span className="ov-activity-meta">{label || "Scan"} · {e.payload?.plant_name} · {formatLocalDate(e.created_at)}</span>
                      </div>
                      <div className={`health-dot ${dotCls}`} style={{ width: 8, height: 8 }} title={label} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
