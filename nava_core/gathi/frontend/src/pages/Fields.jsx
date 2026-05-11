import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../lib/api.js";

const SOIL_TYPES = [
  "Alluvial", "Black / Regur", "Red", "Laterite", "Desert / Arid",
  "Mountain / Forest", "Saline / Alkaline", "Peaty / Marshy", "Clay", "Sandy",
  "Loamy", "Silt", "Chalky", "Other",
];

export default function Fields() {
  const [fields, setFields] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editField, setEditField] = useState(null);
  const [form, setForm] = useState({ name: "", location: "", area: "", soil_type: "" });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const loadFields = async () => {
    try {
      const data = await apiFetch("/api/fields");
      setFields(data.fields || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadFields(); }, []);

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
      loadFields();
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const closeModal = () => { setShowCreate(false); setEditField(null); setForm({ name: "", location: "", area: "", soil_type: "" }); };

  if (loading) return <div className="page-center"><div className="spinner" /></div>;

  const showModal = showCreate || editField;

  return (
    <div className="stack stack-lg">
      <div className="row row-between">
        <div>
          <h1>Your Fields</h1>
          <p className="text-sm text-muted mt-sm">Manage your agricultural spaces</p>
        </div>
        <button id="btn-add-field" className="btn btn-primary" onClick={() => setShowCreate(true)}>
          + New Field
        </button>
      </div>

      {error && <div className="notice notice-danger">{error}</div>}

      {fields.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="icon">🌾</div>
            <h3>No fields yet</h3>
            <p>Create your first field to start managing crops, running diagnostics, and chatting with NAVA.</p>
            <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
              Create Your First Field
            </button>
          </div>
        </div>
      ) : (
        <div className="grid-3">
          {fields.map((field) => (
            <div key={field.id} className="card card-interactive" style={{ position: "relative" }}>
              <div onClick={() => navigate(`/fields/${field.id}`)} style={{ cursor: "pointer" }}>
                <div className="badge badge-green mb-md">Field</div>
                <h3>{field.name}</h3>
                <div className="stack stack-xs mt-sm">
                  {field.location && <span className="text-sm text-muted">📍 {field.location}</span>}
                  {field.area && <span className="text-sm text-muted">📐 {field.area}</span>}
                  {field.soil_type && <span className="text-sm text-muted">🪨 {field.soil_type}</span>}
                </div>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                style={{ position: "absolute", top: 16, right: 16 }}
                onClick={(e) => { e.stopPropagation(); openEdit(field); }}
                title="Edit field"
              >✏️</button>
            </div>
          ))}
        </div>
      )}

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
