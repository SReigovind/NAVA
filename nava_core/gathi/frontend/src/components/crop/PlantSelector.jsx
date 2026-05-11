import React, { useEffect, useState } from "react";
import { apiFetch } from "../../lib/api.js";

export default function PlantSelector({ cropId, onSelect }) {
  const [plants, setPlants] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    try {
      const data = await apiFetch(`/api/plants?crop_id=${cropId}`);
      setPlants(data.plants || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [cropId]);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setSaving(true);
    setError("");
    try {
      const plant = await apiFetch("/api/plants", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ crop_id: Number(cropId), name: newName.trim(), description: newDesc.trim() || null }),
      });
      setPlants((p) => [...p, plant]);
      setNewName(""); setNewDesc("");
      setCreating(false);
      onSelect(plant);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (plant, e) => {
    e.stopPropagation();
    if (!confirm(`Delete plant "${plant.name}" and all its history?`)) return;
    try {
      await apiFetch(`/api/plants/${plant.id}`, { method: "DELETE" });
      setPlants((p) => p.filter((x) => x.id !== plant.id));
    } catch (e) {
      setError(e.message);
    }
  };

  if (loading) return <div className="page-center" style={{ minHeight: 120 }}><div className="spinner" /></div>;

  return (
    <div className="plant-selector">
      <div className="row row-between mb-md">
        <h3 className="text-sm" style={{ color: "var(--green-400)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Select Plant
        </h3>
        <button className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>+ Add Plant</button>
      </div>

      {error && <div className="notice notice-danger mb-md">{error}</div>}

      {plants.length === 0 ? (
        <div className="empty-state" style={{ padding: "24px" }}>
          <div className="icon">🌱</div>
          <p className="text-sm text-muted">No plants tracked yet. Add your first plant to begin analysis.</p>
        </div>
      ) : (
        <div className="plant-grid">
          {plants.map((p) => (
            <div key={p.id} className="plant-card-wrapper" style={{ position: "relative" }}>
              <button className="plant-card" id={`plant-${p.id}`} onClick={() => onSelect(p)}>
                <div className="plant-card-icon">🪴</div>
                <div className="plant-card-name">{p.name}</div>
                {p.description && <div className="plant-card-desc">{p.description}</div>}
              </button>
              <button
                className="plant-delete-btn"
                onClick={(e) => handleDelete(p, e)}
                title={`Delete ${p.name}`}
              >×</button>
            </div>
          ))}
        </div>
      )}

      {creating && (
        <div className="modal-overlay" onClick={() => setCreating(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <h3 className="mb-md">Add New Plant</h3>
            <form onSubmit={handleCreate} className="stack stack-md">
              <label className="label">
                Plant Name *
                <input className="input" value={newName} onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Plant-A, Row-1, North-Bed" autoFocus required />
              </label>
              <label className="label">
                Description (optional)
                <input className="input" value={newDesc} onChange={(e) => setNewDesc(e.target.value)}
                  placeholder="e.g. Near north fence, tallest in row" />
              </label>
              <div className="row row-gap" style={{ justifyContent: "flex-end" }}>
                <button type="button" className="btn btn-ghost" onClick={() => setCreating(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={saving || !newName.trim()}>
                  {saving ? "Adding..." : "Add Plant"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
