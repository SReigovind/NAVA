import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { apiFetch } from "../lib/api.js";
import { useAuth } from "../components/AuthProvider.jsx";
import ChatPanel from "../components/crop/ChatPanel.jsx";
import DiagnosePanel from "../components/crop/DiagnosePanel.jsx";
import MonitorPanel from "../components/crop/MonitorPanel.jsx";
import OverviewPanel from "../components/crop/OverviewPanel.jsx";

const CROP_STAGES = ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity", "Harvested"];

const NAV_ITEMS = [
  { id: "overview",  icon: "🏡", label: "Overview" },
  { id: "chat",      icon: "💬", label: "Ask NAVA" },
  { id: "diagnose",  icon: "🔬", label: "Disease Detection" },
  { id: "monitor",   icon: "📡", label: "Stress Monitor" },
];

export default function CropDetail() {
  const { fieldId, cropId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuth();

  const [field, setField] = useState(null);
  const [crop, setCrop] = useState(null);
  const [activeTool, setActiveTool] = useState("overview");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Edit crop meta
  const [editing, setEditing] = useState(false);
  const [editForm, setEditForm] = useState({ name: "", variety: "", season: "", stage: "" });
  const [savingEdit, setSavingEdit] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const [fieldData, cropData] = await Promise.all([
        apiFetch("/api/fields"),
        apiFetch(`/api/crops?field_id=${fieldId}`),
      ]);
      const f = (fieldData.fields || []).find(x => String(x.id) === String(fieldId));
      const c = (cropData.crops || []).find(x => String(x.id) === String(cropId));
      setField(f || null);
      setCrop(c || null);
      setEditForm({ name: c?.name || "", variety: c?.variety || "", season: c?.season || "", stage: c?.stage || "" });
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, [fieldId, cropId]);

  const saveCropEdit = async () => {
    setSavingEdit(true);
    try {
      const data = await apiFetch("/api/crops", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_id: Number(cropId), ...editForm }),
      });
      setCrop(data);
      setEditing(false);
    } catch (err) { setError(err.message); }
    finally { setSavingEdit(false); }
  };

  if (loading) return <div className="page-center"><div className="spinner" /></div>;
  if (!field || !crop) return <div className="card" style={{ margin: 24 }}>{error || "Crop not found."}</div>;

  const stageColor = { Seedling: "#34d399", Vegetative: "#10b981", Flowering: "#f59e0b", Fruiting: "#f97316", Maturity: "#8b5cf6", Harvested: "#6b7280" };

  return (
    <div className="crop-workspace">
      {/* ── Sidebar ─────────────────────────────── */}
      <aside className={`crop-sidebar ${sidebarOpen ? "open" : "collapsed"}`}>
        <div className="sidebar-top">
          <button className="sidebar-collapse-btn" onClick={() => setSidebarOpen(v => !v)} title={sidebarOpen ? "Collapse" : "Expand"}>
            {sidebarOpen ? "◀" : "▶"}
          </button>
          {sidebarOpen && (
            <button className="btn btn-ghost btn-sm" onClick={() => navigate(`/fields/${field.id}`)}>
              ← {field.name}
            </button>
          )}
        </div>

        {sidebarOpen && (
          <div className="sidebar-crop-info">
            <div className="sidebar-crop-name">{crop.name}</div>
            <div className="sidebar-crop-meta">
              {crop.stage && (
                <span className="stage-pill" style={{ background: `${stageColor[crop.stage] || "#6b7280"}22`, color: stageColor[crop.stage] || "#6b7280", borderColor: `${stageColor[crop.stage] || "#6b7280"}44` }}>
                  {crop.stage}
                </span>
              )}
              {crop.variety && <span className="text-xs text-muted">🧬 {crop.variety}</span>}
            </div>
          </div>
        )}

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <button
              key={item.id}
              className={`sidebar-nav-item ${activeTool === item.id ? "active" : ""}`}
              onClick={() => setActiveTool(item.id)}
              title={!sidebarOpen ? item.label : ""}
            >
              <span className="sidebar-nav-icon">{item.icon}</span>
              {sidebarOpen && <span className="sidebar-nav-label">{item.label}</span>}
            </button>
          ))}
        </nav>

        {sidebarOpen && (
          <div className="sidebar-footer">
            <button className="btn btn-ghost btn-sm" onClick={() => setEditing(true)} title="Edit crop details">
              ✏️ Edit Crop
            </button>
          </div>
        )}
      </aside>

      {/* ── Main content ─────────────────────────── */}
      <main className="crop-main">
        {/* Tool header */}
        <div className="crop-main-header">
          <h2 className="tool-title">
            {NAV_ITEMS.find(n => n.id === activeTool)?.icon} {NAV_ITEMS.find(n => n.id === activeTool)?.label}
          </h2>
          {activeTool === "overview" && (
            <div className="row row-gap text-sm text-muted" style={{ flexWrap: "wrap" }}>
              {crop.season && <span>🗓 {crop.season}</span>}
              {field.location && <span>📍 {field.location}</span>}
              {field.area && <span>📐 {field.area}</span>}
            </div>
          )}
        </div>

        {error && <div className="notice notice-danger" style={{ marginBottom: 16 }}>{error}</div>}

        <div className="crop-tool-body">
          {activeTool === "overview" && (
            <OverviewPanel fieldId={fieldId} cropId={cropId} crop={crop} field={field} onNavigate={setActiveTool} />
          )}
          {activeTool === "chat" && (
            <ChatPanel fieldId={fieldId} cropId={cropId} userId={user?.id} />
          )}
          {activeTool === "diagnose" && (
            <DiagnosePanel fieldId={fieldId} cropId={cropId} />
          )}
          {activeTool === "monitor" && (
            <MonitorPanel fieldId={fieldId} cropId={cropId} />
          )}
        </div>
      </main>

      {/* ── Edit crop modal ──────────────────────── */}
      {editing && (
        <div className="modal-overlay" onClick={() => setEditing(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h2 style={{ marginBottom: 20 }}>Edit Crop</h2>
            <div className="stack stack-md">
              <div className="grid-2">
                <label className="label">Crop Name
                  <input className="input" value={editForm.name} onChange={e => setEditForm({ ...editForm, name: e.target.value })} />
                </label>
                <label className="label">Variety
                  <input className="input" value={editForm.variety} onChange={e => setEditForm({ ...editForm, variety: e.target.value })} />
                </label>
              </div>
              <div className="grid-2">
                <label className="label">Season
                  <input className="input" value={editForm.season} onChange={e => setEditForm({ ...editForm, season: e.target.value })} />
                </label>
                <label className="label">Growth Stage
                  <select className="select" value={editForm.stage} onChange={e => setEditForm({ ...editForm, stage: e.target.value })}>
                    <option value="">Select…</option>
                    {CROP_STAGES.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
              </div>
              <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                <button className="btn btn-ghost" onClick={() => { setEditing(false); setEditForm({ name: crop.name, variety: crop.variety || "", season: crop.season || "", stage: crop.stage || "" }); }}>Cancel</button>
                <button className="btn btn-primary" onClick={saveCropEdit} disabled={savingEdit}>{savingEdit ? "Saving…" : "Save Changes"}</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
