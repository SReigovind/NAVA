import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { apiFetch } from "../lib/api.js";

const CROP_STAGES = ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity", "Harvested"];
const SOIL_TYPES = [
  "Alluvial", "Black / Regur", "Red", "Laterite", "Desert / Arid",
  "Mountain / Forest", "Saline / Alkaline", "Peaty / Marshy", "Clay", "Sandy",
  "Loamy", "Silt", "Chalky", "Other",
];

export default function FieldDetail() {
  const { fieldId } = useParams();
  const navigate = useNavigate();
  const [field, setField] = useState(null);
  const [crops, setCrops] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editCrop, setEditCrop] = useState(null);
  const [form, setForm] = useState({ name: "", variety: "", season: "", stage: "" });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const [editingField, setEditingField] = useState(false);
  const [fieldForm, setFieldForm] = useState({ name: "", location: "", area: "", soil_type: "" });
  const [savingField, setSavingField] = useState(false);

  // Manual notes only (field_notes) — shared_context is auto-generated, hidden from UI
  const [editingCtx, setEditingCtx] = useState(false);
  const [ctxValue, setCtxValue] = useState("");
  const [savingCtx, setSavingCtx] = useState(false);

  const loadData = async () => {
    try {
      const [fieldData, cropData] = await Promise.all([
        apiFetch("/api/fields"),
        apiFetch(`/api/crops?field_id=${fieldId}`),
      ]);
      const f = (fieldData.fields || []).find((x) => String(x.id) === String(fieldId));
      setField(f || null);
      setCtxValue(f?.field_notes || "");   // ← manual notes only
      setCrops(cropData.crops || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, [fieldId]);

  const openEditCrop = (crop) => {
    setForm({ name: crop.name, variety: crop.variety || "", season: crop.season || "", stage: crop.stage || "" });
    setEditCrop(crop);
  };

  const handleCropSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      if (editCrop) {
        await apiFetch("/api/crops", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ crop_id: editCrop.id, ...form }),
        });
        setEditCrop(null);
      } else {
        await apiFetch("/api/crops", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...form, field_id: Number(fieldId) }),
        });
        setShowCreate(false);
      }
      setForm({ name: "", variety: "", season: "", stage: "" });
      loadData();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const deleteCrop = async (crop) => {
    if (!confirm(`Delete crop "${crop.name}" and all its plants and history? This cannot be undone.`)) return;
    try {
      await apiFetch(`/api/crops/${crop.id}`, { method: "DELETE" });
      loadData();
    } catch (err) { setError(err.message); }
  };

  const closeModal = () => { setShowCreate(false); setEditCrop(null); setForm({ name: "", variety: "", season: "", stage: "" }); setEditingField(false); };

  const handleFieldSubmit = async (e) => {
    e.preventDefault();
    setSavingField(true);
    try {
      await apiFetch("/api/fields", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ field_id: field.id, ...fieldForm }),
      });
      setEditingField(false);
      loadData();
    } catch (err) { setError(err.message); }
    finally { setSavingField(false); }
  };

  const saveFieldNotes = async () => {
    setSavingCtx(true);
    try {
      await apiFetch("/api/field-notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ field_id: Number(fieldId), notes: ctxValue }),
      });
      setField({ ...field, field_notes: ctxValue });
      setEditingCtx(false);
    } catch (err) { setError(err.message); }
    finally { setSavingCtx(false); }
  };

  if (loading) return <div className="page-center"><div className="spinner" /></div>;
  if (!field) return <div className="card">Field not found.</div>;

  const showModal = showCreate || editCrop;

  return (
    <div className="stack stack-lg">
      <div className="row row-gap">
        <button className="btn btn-ghost btn-sm" onClick={() => navigate("/fields")}>← Back to Fields</button>
      </div>

      {/* Main 2-Column Bento Layout */}
      <div className="grid-2" style={{ alignItems: "stretch", minHeight: "450px" }}>
        
        {/* Left Column: Context & Notes */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Field header */}
          <div className="card">
            <div className="row row-between">
              <div>
                <div className="badge badge-green mb-md">Field</div>
                <div className="row row-gap" style={{ alignItems: "center" }}>
                  <h2 style={{ margin: 0 }}>{field.name}</h2>
                  <button className="btn btn-ghost btn-sm" style={{ padding: "4px" }} onClick={() => {
                    setFieldForm({ name: field.name, location: field.location || "", area: field.area || "", soil_type: field.soil_type || "" });
                    setEditingField(true);
                  }} title="Edit Field">✏️</button>
                </div>
                <div className="row row-gap mt-sm text-sm text-muted" style={{ flexWrap: "wrap" }}>
                  {field.location && <span>📍 {field.location}</span>}
                  {field.area && <span>• 📐 {field.area}</span>}
                  {field.soil_type && <span>• 🪨 {field.soil_type}</span>}
                </div>
              </div>
            </div>
          </div>

          {/* Manual field notes */}
          <div className="card" style={{ background: "var(--bg-glass)", display: "flex", flexDirection: "column", flex: 1 }}>
            <div className="row row-between mb-md">
              <strong className="text-sm">Field Notes</strong>
              {!editingCtx && (
                <button className="btn btn-ghost btn-sm" onClick={() => setEditingCtx(true)}>
                  {ctxValue ? "✏️ Edit" : "+ Add Notes"}
                </button>
              )}
            </div>
            {editingCtx ? (
              <div className="stack stack-sm" style={{ flex: 1, display: "flex", flexDirection: "column" }}>
                <textarea className="textarea custom-scrollbar" value={ctxValue} onChange={(e) => setCtxValue(e.target.value)}
                  placeholder="Observations, soil history, irrigation notes, pest pressure…" style={{ flex: 1, minHeight: "120px", resize: "none" }} />
                <div className="row row-gap mt-sm" style={{ justifyContent: "flex-end" }}>
                  <button className="btn btn-ghost btn-sm" onClick={() => { setEditingCtx(false); setCtxValue(field.field_notes || ""); }}>Cancel</button>
                  <button className="btn btn-primary btn-sm" onClick={saveFieldNotes} disabled={savingCtx}>
                    {savingCtx ? "Saving…" : "Save Notes"}
                  </button>
                </div>
              </div>
            ) : ctxValue ? (
              <div className="custom-scrollbar" style={{ flex: 1, overflowY: "auto", paddingRight: "8px" }}>
                <p className="text-sm text-muted" style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, margin: 0 }}>{ctxValue}</p>
              </div>
            ) : (
              <p className="text-sm text-muted" style={{ margin: "auto 0" }}>No notes yet. Add field-level observations to help NAVA give better advice.</p>
            )}
          </div>
        </div>

        {/* Right Column: Crops */}
        <div style={{ position: "relative", minHeight: 0 }}>
          <div className="card" style={{ padding: 0, display: "flex", flexDirection: "column", position: "absolute", inset: 0, margin: 0 }}>
            <div className="row row-between" style={{ padding: "16px", borderBottom: "1px solid var(--border-default)" }}>
              <h2 style={{ fontSize: "1.125rem", color: "var(--text-secondary)", margin: 0 }}>CROPS IN THIS FIELD</h2>
              <button id="btn-add-crop" className="btn btn-primary btn-sm" onClick={() => setShowCreate(true)}>+ Add Crop</button>
            </div>
            
            <div className="custom-scrollbar" style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
              {error && <div className="notice notice-danger mb-md">{error}</div>}
              
              {crops.length === 0 ? (
                <div className="empty-state" style={{ height: "100%", justifyContent: "center" }}>
                  <div className="icon">🌱</div>
                  <h3>No crops yet</h3>
                  <p>Add a crop to start diagnostics, monitoring, and chat.</p>
                  <button className="btn btn-primary mt-sm" onClick={() => setShowCreate(true)}>Add First Crop</button>
                </div>
              ) : (
                <div className="grid-2">
                  {crops.map((crop) => (
                    <div key={crop.id} className="card card-interactive card-accent" style={{ position: "relative", padding: "16px" }}>
                      <div onClick={() => navigate(`/fields/${fieldId}/crops/${crop.id}`)} style={{ cursor: "pointer" }}>
                        <h3 style={{ margin: "0 0 8px 0" }}>{crop.name}</h3>
                        <div className="stack stack-xs text-sm text-muted">
                          {crop.variety && <span>🧬 {crop.variety}</span>}
                          {crop.stage && <span className={`badge ${crop.stage === "Harvested" ? "badge-neutral" : "badge-green"}`} style={{ alignSelf: "flex-start", marginTop: "4px" }}>{crop.stage}</span>}
                          {crop.season && <span>🗓 {crop.season}</span>}
                        </div>
                      </div>
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ position: "absolute", top: 12, right: 44, padding: "4px" }}
                        onClick={(e) => { e.stopPropagation(); openEditCrop(crop); }}
                        title="Edit crop"
                      >✏️</button>
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ position: "absolute", top: 12, right: 12, padding: "4px" }}
                        onClick={(e) => { e.stopPropagation(); deleteCrop(crop); }}
                        title="Delete crop"
                      >🗑️</button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Create / Edit crop modal */}
      {showModal && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>{editCrop ? "Edit Crop" : "Add New Crop"}</h2>
            <form onSubmit={handleCropSubmit} className="stack stack-md">
              <label className="label">
                Crop Name *
                <input id="crop-name" className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="e.g. Rice" required />
              </label>
              <label className="label">
                Variety
                <input className="input" value={form.variety} onChange={(e) => setForm({ ...form, variety: e.target.value })} placeholder="e.g. Jyothi" />
              </label>
              <div className="grid-2">
                <label className="label">
                  Season
                  <input className="input" value={form.season} onChange={(e) => setForm({ ...form, season: e.target.value })} placeholder="e.g. Kharif 2026" />
                </label>
                <label className="label">
                  Growth Stage
                  <select className="select" value={form.stage} onChange={(e) => setForm({ ...form, stage: e.target.value })}>
                    <option value="">Select stage...</option>
                    {CROP_STAGES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
              </div>
              <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                <button type="button" className="btn btn-ghost" onClick={closeModal}>Cancel</button>
                <button id="crop-submit" type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? "Saving..." : editCrop ? "Save Changes" : "Add Crop"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit field modal */}
      {editingField && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Edit Field</h2>
            <form onSubmit={handleFieldSubmit} className="stack stack-md">
              <label className="label">
                Field Name *
                <input className="input" value={fieldForm.name} onChange={(e) => setFieldForm({ ...fieldForm, name: e.target.value })} placeholder="e.g. North Paddock" required />
              </label>
              <label className="label">
                Location
                <input className="input" value={fieldForm.location} onChange={(e) => setFieldForm({ ...fieldForm, location: e.target.value })} placeholder="e.g. Wayanad, Kerala" />
              </label>
              <div className="grid-2">
                <label className="label">
                  Area / Size
                  <input className="input" value={fieldForm.area} onChange={(e) => setFieldForm({ ...fieldForm, area: e.target.value })} placeholder="e.g. 2 acres" />
                </label>
                <label className="label">
                  Soil Type
                  <select className="select" value={fieldForm.soil_type} onChange={(e) => setFieldForm({ ...fieldForm, soil_type: e.target.value })}>
                    <option value="">Select soil type...</option>
                    {SOIL_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
              </div>
              <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                <button type="button" className="btn btn-ghost" onClick={closeModal}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={savingField}>
                  {savingField ? "Saving..." : "Save Changes"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
