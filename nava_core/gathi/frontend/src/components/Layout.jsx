import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "./AuthProvider.jsx";

export default function Layout({ children, noPadding = false }) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [theme, setTheme] = useState(() => localStorage.getItem("nava_theme") || "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("nava_theme", theme);
  }, [theme]);

  const toggle = () => setTheme(t => t === "dark" ? "light" : "dark");

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="row row-gap" onClick={() => navigate("/fields")} style={{ cursor: "pointer" }}>
          <img src="/api/logo" alt="NAVA" style={{ width: 30, height: 30, borderRadius: "8px" }} />
          <span className="app-logo"><span>N</span>AVA</span>
        </div>
        <div className="row row-gap">
          <span className="text-sm text-muted">{user?.name}</span>
          <button id="theme-toggle" className="btn btn-ghost btn-sm theme-toggle" onClick={toggle}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => { logout(); navigate("/"); }}>Logout</button>
        </div>
      </header>
      {noPadding
        ? <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>{children}</div>
        : <main className="app-main">{children}</main>
      }
    </div>
  );
}
