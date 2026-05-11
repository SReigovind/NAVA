import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { apiFetch } from "../lib/api.js";
import { useAuth } from "../components/AuthProvider.jsx";

export default function Auth() {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      const url = mode === "login" ? "/api/auth/login" : "/api/auth/register";
      const body = mode === "login" ? { email, password } : { name, email, password };
      const data = await apiFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      login(data.token, data.user);
      navigate("/fields");
    } catch (err) {
      setError(err.message || "Authentication failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="card auth-card">
        <div style={{ textAlign: "center", marginBottom: "24px" }}>
          <img src="/api/logo" alt="NAVA" style={{ width: 56, height: 56, borderRadius: "14px", marginBottom: "12px", boxShadow: "0 4px 16px rgba(16, 185, 129, 0.2)" }} />
          <div className="app-logo" style={{ fontSize: "1.5rem", marginBottom: "4px" }}>
            <span style={{ color: "var(--green-400)" }}>N</span>AVA
          </div>
          <p className="text-sm text-muted">Sign in to your digital agronomist</p>
        </div>

        <div className="auth-toggle">
          <button className={`tab ${mode === "login" ? "active" : ""}`} onClick={() => setMode("login")}>
            Sign In
          </button>
          <button className={`tab ${mode === "register" ? "active" : ""}`} onClick={() => setMode("register")}>
            Create Account
          </button>
        </div>

        {error && <div className="notice notice-danger mb-md">{error}</div>}

        <form onSubmit={handleSubmit} className="stack stack-md">
          {mode === "register" && (
            <label className="label">
              Full Name
              <input
                id="auth-name"
                className="input"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                required
              />
            </label>
          )}
          <label className="label">
            Email
            <input
              id="auth-email"
              className="input"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />
          </label>
          <label className="label">
            Password
            <input
              id="auth-password"
              className="input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "register" ? "Min 8 characters" : "••••••••"}
              minLength={8}
              required
            />
          </label>
          <button id="auth-submit" className="btn btn-primary w-full" type="submit" disabled={busy}>
            {busy ? "Please wait..." : mode === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>
      </div>
    </div>
  );
}
