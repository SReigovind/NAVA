import React, { useEffect, useState } from "react";
import { apiFetch } from "../../lib/api.js";

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



export default function OverviewPanel({ fieldId, cropId, crop, field, onNavigate }) {
  const [plants, setPlants] = useState([]);
  const [events, setEvents] = useState([]);
  const [notes, setNotes] = useState(crop?.notes || "");
  const [editingNotes, setEditingNotes] = useState(false);
  const [savingNotes, setSavingNotes] = useState(false);
  const [loading, setLoading] = useState(true);

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
    setNotes(crop?.notes || "");
  }, [cropId, crop]);

  const saveNotes = async () => {
    setSavingNotes(true);
    try {
      await apiFetch("/api/crop-context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_id: Number(cropId), notes }),
      });
      setEditingNotes(false);
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
    <div className="ov-layout">
      {/* Stats row */}
      <div className="ov-stats">
        <StatCard icon="🌱" label="Plants Tracked" value={plants.length} />
        <StatCard icon="🔬" label="Disease Scans" value={diagCount}
          sub={diseaseEvents.length > 0 ? `⚠️ ${diseaseEvents.length} concern(s)` : diagCount > 0 ? "All clear" : "No scans yet"}
          subColor={diseaseEvents.length > 0 ? "#f87171" : undefined} />
        <StatCard icon="📡" label="VNIR Scans" value={vnirCount} sub="Stress monitoring" />
        <StatCard icon="🌾" label="Growth Stage" value={crop?.stage || "Unknown"} sub={crop?.season || ""} />
      </div>

      <div className="ov-main">
        {/* Plant health */}
        <div className="ov-card">
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
            <div className="ov-plant-list">
              {plants.map(p => <PlantHealthRow key={p.id} plant={p} events={events} />)}
            </div>
          )}
        </div>

        <div className="ov-side">
          {/* Crop notes (manual only) */}
          <div className="ov-card">
            <div className="ov-card-header">
              <h3 className="ov-card-title">Crop Notes</h3>
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
                  <button className="btn btn-ghost btn-sm" onClick={() => { setEditingNotes(false); setNotes(crop?.notes || ""); }}>Cancel</button>
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
          <div className="ov-card">
            <div className="ov-card-header">
              <h3 className="ov-card-title">Recent Activity</h3>
              <button className="btn btn-ghost btn-sm" onClick={load} title="Refresh">↻</button>
            </div>
            {recentEvents.length === 0 ? (
              <p className="text-sm text-muted">No activity yet.</p>
            ) : (
              <div className="ov-activity-list">
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
                        <span className="ov-activity-meta">{label || "Scan"} · {e.payload?.plant_name} · {(e.created_at || "").slice(0, 10)}</span>
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
