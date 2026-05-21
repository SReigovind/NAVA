import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../components/AuthProvider.jsx";
import { apiFetch } from "../lib/api.js";

export default function Profile() {
  const { user } = useAuth();
  const navigate = useNavigate();
  
  // Profile Form
  const [nameForm, setNameForm] = useState(user?.name || "");
  const [savingName, setSavingName] = useState(false);
  const [nameSuccess, setNameSuccess] = useState(false);
  
  // Password Form
  const [passwordForm, setPasswordForm] = useState({ current: "", newPass: "", confirmPass: "" });
  const [savingPassword, setSavingPassword] = useState(false);
  const [passError, setPassError] = useState("");
  const [passSuccess, setPassSuccess] = useState(false);

  // Danger Zone
  const [deleteStep, setDeleteStep] = useState(0);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    if (user) setNameForm(user.name);
  }, [user]);

  const handleUpdateName = async (e) => {
    e.preventDefault();
    setSavingName(true);
    setNameSuccess(false);
    try {
      await apiFetch("/api/auth/me", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: nameForm })
      });
      setNameSuccess(true);
      setTimeout(() => window.location.reload(), 1000);
    } catch (e) {
      console.error(e);
    } finally {
      setSavingName(false);
    }
  };

  const handleUpdatePassword = async (e) => {
    e.preventDefault();
    setPassError("");
    setPassSuccess(false);

    if (passwordForm.newPass !== passwordForm.confirmPass) {
      setPassError("New passwords do not match.");
      return;
    }
    if (passwordForm.current === passwordForm.newPass) {
      setPassError("New password must be different from current password.");
      return;
    }

    setSavingPassword(true);
    try {
      await apiFetch("/api/auth/password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_password: passwordForm.current, new_password: passwordForm.newPass })
      });
      setPassSuccess(true);
      setPasswordForm({ current: "", newPass: "", confirmPass: "" });
    } catch (e) {
      setPassError(e.message || "Failed to update password.");
    } finally {
      setSavingPassword(false);
    }
  };

  const handleDeleteAccount = async () => {
    setDeleting(true);
    try {
      await apiFetch("/api/auth/me", { method: "DELETE" });
      localStorage.removeItem("nava_token");
      window.location.href = "/";
    } catch (e) {
      console.error(e);
      setDeleting(false);
    }
  };

  if (!user) return null;

  return (
    <div style={{ flex: 1, overflowY: "auto", background: "var(--bg-primary)" }}>
      <div style={{ maxWidth: 640, margin: "0 auto", padding: "32px 16px" }}>
        
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
          <h1 style={{ fontSize: "1.5rem", fontWeight: 600, margin: 0, color: "var(--text-primary)" }}>Settings</h1>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate("/fields")}>Back to Dashboard</button>
        </div>

        <div className="stack" style={{ gap: "24px" }}>
          
          {/* Profile Section */}
          <div className="card shadow-sm" style={{ padding: 0, overflow: "hidden", background: "var(--bg-secondary)" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border-default)" }}>
              <h2 style={{ fontSize: "1.1rem", margin: 0, fontWeight: 600 }}>Profile Details</h2>
            </div>
            <form onSubmit={handleUpdateName}>
              <div className="stack stack-sm" style={{ padding: "20px" }}>
                <label className="label" style={{ margin: 0 }}>
                  <span style={{ display: "block", marginBottom: "4px", fontSize: "0.875rem", fontWeight: 500 }}>Email Address</span>
                  <input className="input" type="email" value={user.email} disabled style={{ opacity: 0.7, cursor: "not-allowed", padding: "8px 12px" }} />
                </label>
                <label className="label" style={{ margin: 0 }}>
                  <span style={{ display: "block", marginBottom: "4px", fontSize: "0.875rem", fontWeight: 500 }}>Username</span>
                  <input className="input" type="text" value={nameForm} onChange={e => setNameForm(e.target.value)} required minLength={1} style={{ padding: "8px 12px" }} />
                </label>
              </div>
              
              <div style={{ background: "var(--bg-card)", borderTop: "1px solid var(--border-default)", padding: "12px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="text-sm">
                  {nameSuccess ? <span style={{ color: "var(--green-500)", fontWeight: 500 }}>Updated successfully!</span> : <span style={{ color: "var(--text-muted)" }}>Manage your display name.</span>}
                </div>
                <button className="btn btn-primary btn-sm" type="submit" disabled={savingName || nameForm === user.name}>
                  {savingName ? "Saving..." : "Save"}
                </button>
              </div>
            </form>
          </div>

          {/* Security Section */}
          <div className="card shadow-sm" style={{ padding: 0, overflow: "hidden", background: "var(--bg-secondary)" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--border-default)" }}>
              <h2 style={{ fontSize: "1.1rem", margin: 0, fontWeight: 600 }}>Password & Security</h2>
            </div>
            <form onSubmit={handleUpdatePassword}>
              <div className="stack stack-sm" style={{ padding: "20px" }}>
                <label className="label" style={{ margin: 0 }}>
                  <span style={{ display: "block", marginBottom: "4px", fontSize: "0.875rem", fontWeight: 500 }}>Current Password</span>
                  <input className="input" type="password" required value={passwordForm.current} onChange={e => setPasswordForm({...passwordForm, current: e.target.value})} style={{ padding: "8px 12px" }} />
                </label>
                <label className="label" style={{ margin: 0 }}>
                  <span style={{ display: "block", marginBottom: "4px", fontSize: "0.875rem", fontWeight: 500 }}>New Password</span>
                  <input className="input" type="password" required minLength={8} value={passwordForm.newPass} onChange={e => setPasswordForm({...passwordForm, newPass: e.target.value})} style={{ padding: "8px 12px" }} />
                </label>
                <label className="label" style={{ margin: 0 }}>
                  <span style={{ display: "block", marginBottom: "4px", fontSize: "0.875rem", fontWeight: 500 }}>Confirm New Password</span>
                  <input className="input" type="password" required minLength={8} value={passwordForm.confirmPass} onChange={e => setPasswordForm({...passwordForm, confirmPass: e.target.value})} style={{ padding: "8px 12px" }} />
                </label>
                
                {passError && (
                  <div style={{ padding: "8px 12px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: "4px", color: "var(--red-500)", fontSize: "0.875rem", marginTop: "8px" }}>
                    {passError}
                  </div>
                )}
                {passSuccess && (
                  <div style={{ padding: "8px 12px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.2)", borderRadius: "4px", color: "var(--green-500)", fontSize: "0.875rem", marginTop: "8px" }}>
                    Password updated successfully.
                  </div>
                )}
              </div>
              
              <div style={{ background: "var(--bg-card)", borderTop: "1px solid var(--border-default)", padding: "12px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="text-sm" style={{ color: "var(--text-muted)" }}>
                  Must be at least 8 characters.
                </div>
                <button className="btn btn-primary btn-sm" type="submit" disabled={savingPassword || !passwordForm.current || !passwordForm.newPass || !passwordForm.confirmPass}>
                  {savingPassword ? "Updating..." : "Update Password"}
                </button>
              </div>
            </form>
          </div>

          {/* Danger Zone */}
          <div className="card shadow-sm" style={{ padding: 0, overflow: "hidden", border: "1px solid rgba(239, 68, 68, 0.3)" }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid rgba(239, 68, 68, 0.2)", background: "rgba(239, 68, 68, 0.05)" }}>
              <h2 style={{ fontSize: "1.1rem", margin: 0, fontWeight: 600, color: "var(--red-500)" }}>Delete Account</h2>
            </div>
            
            <div style={{ padding: "20px" }}>
              <p style={{ margin: "0 0 16px 0", fontSize: "0.875rem", color: "var(--text-muted)" }}>
                Permanently remove your account and all associated data. This cannot be undone.
              </p>

              {deleteStep === 0 && (
                <button className="btn btn-sm" style={{ background: "rgba(239, 68, 68, 0.1)", color: "var(--red-500)", border: "1px solid rgba(239, 68, 68, 0.2)" }} onClick={() => setDeleteStep(1)}>
                  Delete Account
                </button>
              )}

              {deleteStep === 1 && (
                <div style={{ padding: "12px", background: "rgba(239, 68, 68, 0.05)", borderRadius: "6px", border: "1px dashed rgba(239, 68, 68, 0.3)" }}>
                  <div style={{ fontSize: "0.875rem", color: "var(--red-400)", fontWeight: 500, marginBottom: "8px" }}>Are you absolutely sure?</div>
                  <div className="row row-gap">
                    <button className="btn btn-sm" style={{ background: "var(--red-600)", color: "white", border: "none" }} onClick={() => setDeleteStep(2)}>
                      Yes, proceed
                    </button>
                    <button className="btn btn-ghost btn-sm" onClick={() => setDeleteStep(0)}>Cancel</button>
                  </div>
                </div>
              )}

              {deleteStep === 2 && (
                <div style={{ padding: "12px", background: "rgba(239, 68, 68, 0.1)", borderRadius: "6px", border: "1px solid var(--red-600)" }}>
                  <div style={{ fontSize: "0.875rem", color: "var(--red-500)", fontWeight: 500, marginBottom: "8px" }}>Final confirmation: Data will be lost forever.</div>
                  <div className="row row-gap">
                    <button className="btn btn-sm" style={{ background: "var(--red-600)", color: "white", border: "none" }} onClick={handleDeleteAccount} disabled={deleting}>
                      {deleting ? "Deleting..." : "Permanently Delete"}
                    </button>
                    <button className="btn btn-ghost btn-sm" onClick={() => setDeleteStep(0)} disabled={deleting}>Cancel</button>
                  </div>
                </div>
              )}
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}
