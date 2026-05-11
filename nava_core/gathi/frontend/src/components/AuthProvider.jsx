import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { apiFetch } from "../lib/api.js";
import { clearStoredUser, clearToken, getStoredUser, getToken, setStoredUser, setToken } from "../lib/auth.js";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(getStoredUser());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = getToken();
    if (!token) { setUser(null); setLoading(false); return; }
    apiFetch("/api/auth/me")
      .then(data => { setUser(data); setStoredUser(data); })
      .catch(() => { clearToken(); clearStoredUser(); setUser(null); })
      .finally(() => setLoading(false));
  }, []);

  const login = (token, data) => { setToken(token); setStoredUser(data); setUser(data); };
  const logout = async () => {
    try { await apiFetch("/api/auth/logout", { method: "POST" }); } catch {}
    clearToken(); clearStoredUser(); setUser(null);
  };

  const value = useMemo(() => ({ user, login, logout, loading }), [user, loading]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be within AuthProvider");
  return ctx;
};
