import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../lib/api.js";

const SOIL_TYPES = [
  "Alluvial", "Black / Regur", "Red", "Laterite", "Desert / Arid",
  "Mountain / Forest", "Saline / Alkaline", "Peaty / Marshy", "Clay", "Sandy",
  "Loamy", "Silt", "Chalky", "Other",
];

const formatTimeAgo = (dateStr) => {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr.replace(" ", "T") + "Z").getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hrs ago`;
  return `${Math.floor(hours / 24)} days ago`;
};

const cleanLabel = (text) => {
  if (!text) return "";
  return text.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
};

export default function Fields() {
  const [fields, setFields] = useState([]);
  const [events, setEvents] = useState([]);
  const [cropsByField, setCropsByField] = useState({});
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editField, setEditField] = useState(null);
  const [form, setForm] = useState({ name: "", location: "", area: "", soil_type: "" });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const [activeTabId, setActiveTabId] = useState(null);

  const loadDashboard = async () => {
    try {
      setLoading(true);
      const data = await apiFetch("/api/fields");
      const fList = data.fields || [];
      setFields(fList);

      if (fList.length > 0) {
        setActiveTabId(prev => prev || fList[0].id);
      }

      const cropsMap = {};
      await Promise.all(fList.map(async (f) => {
        try {
          const cData = await apiFetch(`/api/crops?field_id=${f.id}`);
          cropsMap[f.id] = cData.crops || [];
        } catch (e) { }
      }));
      setCropsByField(cropsMap);

      try {
        const eData = await apiFetch("/api/events?limit=100");
        setEvents(eData.events || []);
      } catch (e) { }

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadDashboard(); }, []);

  const openEdit = (field) => {
    setForm({ name: field.name, location: field.location || "", area: field.area || "", soil_type: field.soil_type || "" });
    setEditField(field);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    try {
      if (editField) {
        await apiFetch("/api/fields", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ field_id: editField.id, ...form }),
        });
        setEditField(null);
      } else {
        await apiFetch("/api/fields", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
        });
        setShowCreate(false);
      }
      setForm({ name: "", location: "", area: "", soil_type: "" });
      loadDashboard();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const closeModal = () => { setShowCreate(false); setEditField(null); setForm({ name: "", location: "", area: "", soil_type: "" }); };

  if (loading) return <div className="page-center"><div className="spinner" /></div>;

  const showModal = showCreate || editField;

  // Compute Stats
  const totalFields = fields.length;
  const totalCrops = Object.values(cropsByField).reduce((acc, crops) => acc + crops.length, 0);
  const scanEvents = events.filter(e => e.event_type === "diagnose" || e.event_type === "vnir");
  const totalScans = scanEvents.length;

  const isConcern = (e) => {
    if (e.event_type === "diagnose") {
      const l = (e.payload?.class_label || "").toLowerCase();
      return l && !l.includes("healthy");
    }
    if (e.event_type === "vnir") {
      const s = (e.payload?.status || "").toLowerCase();
      return s && !s.includes("healthy") && !s.includes("ok") && !s.includes("calibrat");
    }
    return false;
  };

  const activeConcerns = scanEvents.filter(isConcern).length;

  const getFieldConcerns = (fieldId) => {
    return scanEvents.filter(e => e.field_id === fieldId && isConcern(e)).length;
  };

  const getCropName = (cropId) => {
    for (const cropList of Object.values(cropsByField)) {
      const crop = cropList.find(c => c.id === cropId);
      if (crop) return crop.name;
    }
    return "Unknown Crop";
  };

  return (
    <div className="stack stack-md" style={{ paddingTop: 4 }}>
      {/* Header */}
      <div className="row row-between">
        <div>
          <h1 style={{ fontSize: "2rem", marginBottom: 0 }}>Welcome back</h1>
        </div>
        <button id="btn-add-field" className="btn btn-primary" onClick={() => setShowCreate(true)}>
          + New Field
        </button>
      </div>

      {error && <div className="notice notice-danger">{error}</div>}

      {/* Main 2-Column Layout */}
      <div className="grid-2" style={{ alignItems: "stretch" }}>

        {/* Left Column: Summaries + Your Fields */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          {/* Summaries 2x2 */}
          <div className="grid-2" style={{ gap: "16px" }}>
            <div className="card" style={{ padding: "12px 16px" }}>
              <div className="text-sm text-muted mb-xs">Total Fields</div>
              <div style={{ fontSize: "1.75rem", fontWeight: 700 }}>{totalFields}</div>
            </div>
            <div className="card" style={{ padding: "12px 16px" }}>
              <div className="text-sm text-muted mb-xs">Active Crops</div>
              <div style={{ fontSize: "1.75rem", fontWeight: 700 }}>{totalCrops}</div>
            </div>
            <div className="card" style={{ padding: "12px 16px" }}>
              <div className="text-sm text-muted mb-xs">Total Scans</div>
              <div style={{ fontSize: "1.75rem", fontWeight: 700 }}>{totalScans}</div>
            </div>
            <div className="card" style={{ padding: "12px 16px" }}>
              <div className="text-sm text-muted mb-xs">Concerns</div>
              <div style={{ fontSize: "1.75rem", fontWeight: 700, color: activeConcerns > 0 ? "var(--red-400)" : "inherit" }}>
                {activeConcerns}
              </div>
            </div>
          </div>

          {/* Your Fields */}
          <div className="card" style={{ padding: 0, display: "flex", flexDirection: "column", height: "290px" }}>
            <div className="row row-between" style={{ padding: "12px 16px", borderBottom: "1px solid var(--border-default)" }}>
              <h2 style={{ fontSize: "1.125rem", color: "var(--text-secondary)", margin: 0 }}>YOUR FIELDS</h2>
            </div>
            <div className="custom-scrollbar" style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
              {fields.length === 0 ? (
                <div className="empty-state" style={{ margin: 0 }}>
                  <p className="text-muted">No fields yet.</p>
                </div>
              ) : (
                <div className="grid-2">
                  {fields.map((field) => {
                    const cropCount = (cropsByField[field.id] || []).length;
                    const concerns = getFieldConcerns(field.id);
                    return (
                      <div key={field.id} className="card card-interactive" style={{ padding: "12px 16px", display: "flex", flexDirection: "column" }} onClick={() => navigate(`/fields/${field.id}`)}>
                        <div className="row row-between mb-xs">
                          <h3 style={{ fontSize: "1.125rem", margin: 0 }}>{field.name}</h3>
                          <button
                            className="btn btn-ghost btn-sm"
                            style={{ padding: 0, height: "auto" }}
                            onClick={(e) => { e.stopPropagation(); openEdit(field); }}
                            title="Edit field"
                          >✏️</button>
                        </div>
                        <div className="stack stack-xs mb-sm">
                          <span className="text-xs text-muted">{cropCount} crop{cropCount !== 1 && "s"} · {field.soil_type || "Unknown Soil"}</span>
                        </div>
                        <div className="row" style={{ alignItems: "center", gap: 8, marginTop: "auto" }}>
                          {concerns > 0 ? (
                            <><span className="dot dot-red" style={{ width: 8, height: 8 }}></span><span className="text-xs" style={{ color: "var(--red-400)", fontWeight: 500 }}>{concerns} concern{concerns !== 1 ? "s" : ""}</span></>
                          ) : (
                            <><span className="dot dot-green" style={{ width: 8, height: 8 }}></span><span className="text-xs" style={{ color: "var(--green-400)", fontWeight: 500 }}>All clear</span></>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Recent Activity */}
        <div style={{ position: "relative", minHeight: 0 }}>
          <div className="card" style={{ padding: 0, display: "flex", flexDirection: "column", position: "absolute", inset: 0, margin: 0 }}>
            <div className="row row-between" style={{ padding: "8px 12px", borderBottom: "1px solid var(--border-default)" }}>
              <h2 style={{ fontSize: "1.125rem", color: "var(--text-secondary)", margin: 0 }}>RECENT ACTIVITY</h2>
            </div>

            {fields.length > 0 && (
              <div className="custom-scrollbar" style={{ padding: "8px 8px", display: "flex", overflowX: "auto", gap: "8px", borderBottom: "1px solid var(--border-default)" }}>
                {fields.map(f => (
                  <button
                    key={f.id}
                    className={`btn ${activeTabId === f.id ? "btn-primary" : "btn-ghost"} btn-sm`}
                    onClick={() => setActiveTabId(f.id)}
                    style={{ whiteSpace: "nowrap" }}
                  >
                    {f.name}
                  </button>
                ))}
              </div>
            )}

            <div className="custom-scrollbar" style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
              {fields.length === 0 || scanEvents.length === 0 ? (
                <div className="empty-state" style={{ margin: 0 }}>
                  <p className="text-muted">No recent activity.</p>
                </div>
              ) : (
                <div className="stack stack-xl">
                  {(() => {
                    const fieldEvents = scanEvents.filter(e => e.field_id === activeTabId).slice(0, 3);
                    if (fieldEvents.length === 0) return <p className="text-muted text-sm" style={{ textAlign: "center", padding: "20px 0" }}>No activity for this field yet.</p>;

                    return (
                      <div className="stack stack-md">
                        {fieldEvents.map((e, idx) => {
                          const isDiag = e.event_type === "diagnose";
                          const label = isDiag ? e.payload?.class_label : e.payload?.status;
                          const l = (label || "").toLowerCase();

                          let dotClass = "dot-gray";
                          if (isDiag) {
                            if (l.includes("healthy")) dotClass = "dot-green";
                            else if (l) dotClass = "dot-red";
                          } else {
                            if (l.includes("calibrat")) dotClass = "dot-blue";
                            else if (l.includes("ok") || l.includes("healthy")) dotClass = "dot-green";
                            else if (l.includes("warning") || l.includes("stress") || l.includes("critical")) dotClass = "dot-red";
                          }
                          const icon = isDiag ? "🔬" : "📡";

                          return (
                            <div key={e.id} style={{ borderBottom: idx < fieldEvents.length - 1 ? "1px solid var(--border-default)" : "none", paddingBottom: idx < fieldEvents.length - 1 ? "16px" : "0" }}>
                              <div className="row" style={{ gap: 12, alignItems: "flex-start" }}>
                                <span style={{ fontSize: "1.25rem", marginTop: "-2px" }}>{icon}</span>
                                <div className="stack" style={{ gap: 6, flex: 1 }}>
                                  <div className="row row-between" style={{ alignItems: "center" }}>
                                    <div className="row" style={{ gap: 6, alignItems: "center" }}>
                                      <span className={`dot ${dotClass}`} style={{ width: 8, height: 8 }}></span>
                                      <span style={{ fontWeight: 600, fontSize: "0.95rem" }}>{cleanLabel(label)}</span>
                                    </div>
                                    <span className="text-xs text-muted">{formatTimeAgo(e.created_at)}</span>
                                  </div>

                                  <div className="text-sm" style={{ color: "var(--text-primary)" }}>
                                    <span className="text-muted">Crop:</span> {getCropName(e.crop_id)}
                                  </div>
                                  <div className="text-sm" style={{ color: "var(--text-primary)" }}>
                                    <span className="text-muted">Plant:</span> {e.payload?.plant_name || "Unknown"}
                                  </div>
                                  <div className="text-sm" style={{ color: "var(--text-primary)" }}>
                                    <span className="text-muted">Type:</span> {isDiag ? "Disease Scan" : "VNIR Reading"}
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>{editField ? "Edit Field" : "Create New Field"}</h2>
            <form onSubmit={handleSubmit} className="stack stack-md">
              <label className="label">
                Field Name *
                <input id="field-name" className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="e.g. North Paddock" required />
              </label>
              <label className="label">
                Location
                <input id="field-location" className="input" value={form.location} onChange={(e) => setForm({ ...form, location: e.target.value })} placeholder="e.g. Wayanad, Kerala" />
              </label>
              <div className="grid-2">
                <label className="label">
                  Area / Size
                  <input className="input" value={form.area} onChange={(e) => setForm({ ...form, area: e.target.value })} placeholder="e.g. 2 acres" />
                </label>
                <label className="label">
                  Soil Type
                  <select className="select" value={form.soil_type} onChange={(e) => setForm({ ...form, soil_type: e.target.value })}>
                    <option value="">Select soil type...</option>
                    {SOIL_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
              </div>
              <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                <button type="button" className="btn btn-ghost" onClick={closeModal}>Cancel</button>
                <button id="field-submit" type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? "Saving..." : editField ? "Save Changes" : "Create Field"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
