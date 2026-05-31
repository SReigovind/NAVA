import React, { useEffect, useState } from "react";
import { apiFetch } from "../../lib/api.js";

/**
 * Compact weather strip for the OverviewPanel.
 * - Reads from GET /api/weather?field_id=X  (DB-backed, zero latency after login refresh)
 * - ↻ button calls POST /api/weather/refresh?field_id=X  (force-fetch, updates DB)
 * - Shows when data was last updated
 * - Silently hides if field has no location, no coordinates yet, or fetch fails
 */
export default function WeatherStrip({ fieldId }) {
  const [weather, setWeather] = useState(null);
  const [status, setStatus]   = useState("loading"); // "loading" | "ok" | "hidden"
  const [refreshing, setRefreshing] = useState(false);

  const load = (isRefresh = false) => {
    if (!fieldId) { setStatus("hidden"); return; }
    if (!isRefresh) setStatus("loading");

    const endpoint = isRefresh
      ? `/api/weather/refresh?field_id=${fieldId}`
      : `/api/weather?field_id=${fieldId}`;
    const method = isRefresh ? "POST" : "GET";

    apiFetch(endpoint, { method })
      .then((data) => {
        if (data.error) { setStatus("hidden"); return; }
        setWeather(data);
        setStatus("ok");
      })
      .catch(() => setStatus("hidden"))
      .finally(() => { if (isRefresh) setRefreshing(false); });
  };

  useEffect(() => {
    let cancelled = false;
    if (!fieldId) { setStatus("hidden"); return; }
    setStatus("loading");

    apiFetch(`/api/weather?field_id=${fieldId}`)
      .then((data) => {
        if (cancelled) return;
        if (data.error) { setStatus("hidden"); return; }
        setWeather(data);
        setStatus("ok");
      })
      .catch(() => { if (!cancelled) setStatus("hidden"); });

    return () => { cancelled = true; };
  }, [fieldId]);

  const handleRefresh = () => {
    if (refreshing) return;
    setRefreshing(true);
    load(true);
  };

  if (status !== "ok" || !weather) return null;

  const fmt = (v, unit) => v !== null && v !== undefined ? `${v}${unit}` : "—";

  // Weather icon heuristic
  const icon = (() => {
    const t = weather.temp ?? 0;
    const p = weather.precipitation ?? 0;
    if (p > 2)   return "🌧️";
    if (p > 0.2) return "🌦️";
    if (t > 35)  return "☀️";
    if (t > 28)  return "🌤️";
    if (t < 18)  return "🌥️";
    return "⛅";
  })();

  // Format updated_at as a relative "X min ago" / "X h ago" string
  const updatedLabel = (() => {
    if (!weather.updated_at) return null;
    try {
      const diff = Math.floor((Date.now() - new Date(weather.updated_at).getTime()) / 1000);
      if (diff < 60)   return "just now";
      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
      return `${Math.floor(diff / 86400)}d ago`;
    } catch { return null; }
  })();

  return (
    <div className="weather-strip" title={`Weather near ${weather.location}`}>
      <span className="weather-strip-icon">{icon}</span>
      <span className="weather-strip-location">{weather.location}</span>
      <span className="weather-strip-divider" />
      <WeatherPill emoji="🌡️" value={fmt(weather.temp, "°C")}       label="Temp" />
      <WeatherPill emoji="💧" value={fmt(weather.humidity, "%")}     label="Humidity" />
      <WeatherPill emoji="🌧"  value={fmt(weather.precipitation, " mm")} label="Rain" />
      <WeatherPill emoji="💨" value={fmt(weather.wind_speed, " km/h")} label="Wind" />
      {updatedLabel && (
        <span className="weather-strip-updated" title={weather.updated_at}>
          {updatedLabel}
        </span>
      )}
      <button
        className="weather-strip-refresh"
        onClick={handleRefresh}
        disabled={refreshing}
        title="Refresh weather"
        aria-label="Refresh weather"
      >
        {refreshing ? "⏳" : "↻"}
      </button>
    </div>
  );
}

function WeatherPill({ emoji, value, label }) {
  return (
    <div className="weather-pill" title={label}>
      <span className="weather-pill-emoji">{emoji}</span>
      <span className="weather-pill-value">{value}</span>
    </div>
  );
}
