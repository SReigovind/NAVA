import React, { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { apiFetch } from "../lib/api.js";

const CROP_STAGES = ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity", "Harvested"];

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

  const closeModal = () => { setShowCreate(false); setEditCrop(null); setForm({ name: "", variety: "", season: "", stage: "" }); };

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

      {/* Field header */}
      <div className="card">
        <div className="row row-between">
          <div>
            <div className="badge badge-green mb-md">Field</div>
            <h2>{field.name}</h2>
            <div className="row row-gap mt-sm text-sm text-muted" style={{ flexWrap: "wrap" }}>
              {field.location && <span>📍 {field.location}</span>}
              {field.area && <span>• 📐 {field.area}</span>}
              {field.soil_type && <span>• 🪨 {field.soil_type}</span>}
            </div>
          </div>
        </div>

        {/* Manual field notes — auto-generated context is hidden, used only in backend */}
        <div className="card mt-md" style={{ background: "var(--bg-glass)" }}>
          <div className="row row-between mb-md">
            <strong className="text-sm">Field Notes</strong>
            {!editingCtx && (
              <button className="btn btn-ghost btn-sm" onClick={() => setEditingCtx(true)}>
                {ctxValue ? "✏️ Edit" : "+ Add Notes"}
              </button>
            )}
          </div>
          {editingCtx ? (
            <div className="stack stack-sm">
              <textarea className="textarea" value={ctxValue} onChange={(e) => setCtxValue(e.target.value)}
                placeholder="Observations, soil history, irrigation notes, pest pressure…" rows={5} />
              <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                <button className="btn btn-ghost btn-sm" onClick={() => { setEditingCtx(false); setCtxValue(field.field_notes || ""); }}>Cancel</button>
                <button className="btn btn-primary btn-sm" onClick={saveFieldNotes} disabled={savingCtx}>
                  {savingCtx ? "Saving…" : "Save Notes"}
                </button>
              </div>
            </div>
          ) : ctxValue ? (
            <p className="text-sm text-muted" style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>{ctxValue}</p>
          ) : (
            <p className="text-sm text-muted">No notes yet. Add field-level observations to help NAVA give better advice.</p>
          )}
        </div>

      </div>

      {/* Crops */}
      <div className="row row-between">
        <h3>Crops</h3>
        <button id="btn-add-crop" className="btn btn-primary btn-sm" onClick={() => setShowCreate(true)}>+ Add Crop</button>
      </div>

      {error && <div className="notice notice-danger">{error}</div>}

      {crops.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="icon">🌱</div>
            <h3>No crops yet</h3>
            <p>Add a crop to start diagnostics, monitoring, and chat.</p>
            <button className="btn btn-primary" onClick={() => setShowCreate(true)}>Add First Crop</button>
          </div>
        </div>
      ) : (
        <div className="grid-3">
          {crops.map((crop) => (
            <div key={crop.id} className="card card-interactive card-accent" style={{ position: "relative" }}>
              <div onClick={() => navigate(`/fields/${fieldId}/crops/${crop.id}`)} style={{ cursor: "pointer" }}>
                <h3>{crop.name}</h3>
                <div className="stack stack-xs mt-sm text-sm text-muted">
                  {crop.variety && <span>🧬 {crop.variety}</span>}
                  {crop.stage && <span className={`badge ${crop.stage === "Harvested" ? "badge-neutral" : "badge-green"}`}>{crop.stage}</span>}
                  {crop.season && <span>🗓 {crop.season}</span>}
                </div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                style={{ position: "absolute", top: 12, right: 48 }}
                onClick={(e) => { e.stopPropagation(); openEditCrop(crop); }}
                title="Edit crop"
              >✏️</button>
              <button
                className="btn btn-ghost btn-sm"
                style={{ position: "absolute", top: 12, right: 12 }}
                onClick={(e) => { e.stopPropagation(); deleteCrop(crop); }}
                title="Delete crop"
              >🗑️</button>
            </div>
          ))}
        </div>
      )}

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
    </div>
  );
}
