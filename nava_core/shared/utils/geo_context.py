"""Geo-weather utilities for NAVA chat context injection.

Two responsibilities:
  1. resolve_coordinates(location_str) — parse or geocode a location string
     to (lat, lon). Parsed decimal coordinates are used directly; free-text
     locations (e.g. "Wayanad, Kerala") are geocoded via Nominatim (OSM).
     Result is stored in the DB so Nominatim is called at most once per field.

  2. get_weather_context(lat, lon) — fetch current weather from Open-Meteo
     (free, no API key, 10k requests/day). Cached for 60 minutes per ~11km
     grid cell (0.1° rounding).

No new pip dependencies — uses stdlib urllib and json only.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from nava_core.shared.utils.logging import get_logger

log = get_logger("shared.geo_context")

# ── Coordinate resolution ────────────────────────────────────────────────────

_DECIMAL_RE = re.compile(
    r"^([+-]?\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)$"
)


def resolve_coordinates(location_str: str) -> tuple[float, float] | None:
    """Parse or geocode a location string to (lat, lon).

    Strategy:
      1. Detect decimal coordinates directly, e.g. "10.85, 76.27"
      2. Geocode via Nominatim (OpenStreetMap), e.g. "Wayanad, Kerala"

    Returns (lat, lon) or None if resolution fails.
    Network timeout: 5 seconds. Never raises.
    """
    s = location_str.strip()
    if not s:
        return None

    # Strategy 1: parse "lat, lon" decimal format
    log.info("[GEO] Attempting decimal coordinate parse for: %r", s)
    m = _DECIMAL_RE.match(s)
    if m:
        try:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                log.info("[GEO] ✓ Parsed decimal coordinates directly: lat=%.6f, lon=%.6f", lat, lon)
                return lat, lon
            else:
                log.info("[GEO] Decimal parse matched but values out of range (lat=%.4f, lon=%.4f) — falling through to Nominatim", lat, lon)
        except ValueError:
            log.debug("[GEO] Decimal parse failed (ValueError) — falling through to Nominatim")

    # Strategy 2: Nominatim geocode
    encoded = urllib.parse.quote_plus(s)
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
    log.info("[GEO] No decimal match — querying Nominatim for: %r", s)
    log.info("[GEO] Nominatim URL: %s", nominatim_url)

    try:
        req = urllib.request.Request(nominatim_url, headers={"User-Agent": "NAVA-AG/2.0 (academic)"})
        with urllib.request.urlopen(req, timeout=5) as r:
            raw_bytes = r.read()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        log.info("[GEO] Nominatim raw response (%d bytes): %s", len(raw_bytes), raw_text[:500])

        data = json.loads(raw_text)
        if data:
            result = data[0]
            log.info("[GEO] Nominatim top result: display_name=%r, lat=%r, lon=%r",
                     result.get("display_name", ""), result.get("lat"), result.get("lon"))
            lat = float(result["lat"])
            lon = float(result["lon"])
            log.info("[GEO] ✓ Geocoded %r → lat=%.6f, lon=%.6f", s, lat, lon)
            return lat, lon
        else:
            log.warning("[GEO] Nominatim returned empty results for: %r", s)
    except Exception as exc:
        log.warning("[GEO] Nominatim request failed for %r: %s", s, exc)

    log.warning("[GEO] Could not resolve coordinates for: %r", s)
    return None


# ── Weather fetch (pure, no in-process cache — DB is source of truth) ────────


def get_weather_context(lat: float, lon: float) -> dict | None:
    """Fetch current weather from Open-Meteo (free, no key required).

    No in-process caching — the DB is the persistent cache.
    Returns dict{temp, humidity, precipitation, wind_speed} or None on error.
    """

    params = (
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m"
        ",precipitation,wind_speed_10m&timezone=auto"
    )
    open_meteo_url = f"https://api.open-meteo.com/v1/forecast?{params}"
    log.info("[WEATHER] Fetching from Open-Meteo for lat=%.4f, lon=%.4f", lat, lon)
    log.info("[WEATHER] Open-Meteo URL: %s", open_meteo_url)

    try:
        with urllib.request.urlopen(open_meteo_url, timeout=5) as r:
            raw_bytes = r.read()
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        log.info("[WEATHER] Open-Meteo raw response (%d bytes): %s", len(raw_bytes), raw_text[:800])

        raw = json.loads(raw_text)
        cur = raw.get("current", {})
        log.info("[WEATHER] Parsed current block: %s", cur)

        result: dict = {
            "temp":          cur.get("temperature_2m"),
            "humidity":      cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"),
            "wind_speed":    cur.get("wind_speed_10m"),
        }
        log.info(
            "[WEATHER] ✓ Fetched for lat=%.4f, lon=%.4f: %.1f°C, %s%% RH, %.1f mm rain, %.1f km/h wind",
            lat, lon,
            result["temp"] or 0,
            result["humidity"] or 0,
            result["precipitation"] or 0,
            result["wind_speed"] or 0,
        )
        return result
    except Exception as exc:
        log.warning("[WEATHER] Open-Meteo fetch failed for lat=%.4f, lon=%.4f: %s", lat, lon, exc)
        return None


# ── Login-triggered batch refresh ───────────────────────────────────────────


def refresh_user_weather(field_store) -> None:
    """Fetch fresh weather for all fields that have lat/lon stored.

    Called as a BackgroundTask on login. Sleeps 1 second between each
    Open-Meteo call so the load is spread across ~N seconds for N fields.
    All failures are silent — the DB retains the previous value and timestamp.
    """
    import time
    fields = field_store.list_fields()
    log.info("[WEATHER-REFRESH] Starting login weather refresh for %d fields", len(fields))
    updated = 0
    for field in fields:
        lat = field.get("lat")
        lon = field.get("lon")
        if lat is None or lon is None:
            log.info(
                "[WEATHER-REFRESH] Field %d (%s): no coordinates — skipping",
                field["id"], field.get("name", ""),
            )
            continue
        try:
            wx = get_weather_context(lat, lon)
            if wx:
                field_store.update_field_weather(
                    field["id"],
                    wx["temp"], wx["humidity"], wx["precipitation"], wx["wind_speed"],
                )
                updated += 1
                log.info(
                    "[WEATHER-REFRESH] ✓ Field %d (%s): %.1f°C, %s%% RH",
                    field["id"], field.get("name", ""), wx["temp"] or 0, wx["humidity"] or 0,
                )
            else:
                log.warning(
                    "[WEATHER-REFRESH] Field %d (%s): API returned nothing — retaining previous value",
                    field["id"], field.get("name", ""),
                )
        except Exception as exc:
            log.warning(
                "[WEATHER-REFRESH] Field %d (%s): failed — %s",
                field["id"], field.get("name", ""), exc,
            )
        time.sleep(1)
    log.info("[WEATHER-REFRESH] Done — updated %d/%d fields", updated, len(fields))


# ── Legacy entry point (used by weather.py router for single-field refresh) ─────


def get_field_weather_context(
    location_str: str,
    cached_lat: float | None = None,
    cached_lon: float | None = None,
) -> tuple[dict | None, float | None, float | None]:
    """Resolve coordinates (if not cached) and fetch weather.

    Used by the /api/weather endpoint when lat/lon aren't yet stored in DB.
    Returns (weather_dict | None, lat | None, lon | None).
    """
    lat, lon = cached_lat, cached_lon

    if lat is not None and lon is not None:
        log.info("[GEO] Using DB-cached coordinates for %r: lat=%.6f, lon=%.6f", location_str, lat, lon)
    else:
        log.info("[GEO] No cached coordinates for %r — starting resolution", location_str)
        if not location_str or not location_str.strip():
            log.info("[GEO] Empty location string — skipping geo resolution")
            return None, None, None
        coords = resolve_coordinates(location_str)
        if coords is None:
            log.warning("[GEO] Resolution failed for %r — no weather will be injected", location_str)
            return None, None, None
        lat, lon = coords
        log.info("[GEO] Resolution complete: lat=%.6f, lon=%.6f", lat, lon)

    weather = get_weather_context(lat, lon)
    return weather, lat, lon
