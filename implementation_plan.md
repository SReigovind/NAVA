# NAVA — Implementation Plan
## Session: 30 May 2026

> This plan covers the two remaining features before project submission.
> Items are ordered by implementation effort (smallest first).

---

## Feature 1 — Kerala Season Dropdown (frontend only, ~15 min)

Replace the free-text season `<input>` with a structured `<select>` dropdown
using Kerala's three-season calendar. Backend and schema are unchanged.

### Season options

| Value | Months |
|-------|--------|
| Summer / Hot Season | March – May |
| Monsoon Season | June – November |
| Winter / Cool Season | December – February |

### Change 1 — `FieldDetail.jsx` (crop create / edit modal)

**Find:**
```jsx
<input className="input" value={form.season}
    onChange={(e) => setForm({ ...form, season: e.target.value })}
    placeholder="e.g. Kharif 2026" />
```

**Replace with:**
```jsx
<select className="select" value={form.season}
    onChange={(e) => setForm({ ...form, season: e.target.value })}>
    <option value="">— Select season —</option>
    <option value="Summer / Hot Season">Summer / Hot Season (Mar–May)</option>
    <option value="Monsoon Season">Monsoon Season (Jun–Nov)</option>
    <option value="Winter / Cool Season">Winter / Cool Season (Dec–Feb)</option>
</select>
```

### Change 2 — `CropDetail.jsx` (inline edit form)

**Find:**
```jsx
<input className="input" value={editForm.season}
    onChange={e => setEditForm({ ...editForm, season: e.target.value })} />
```

**Replace with:**
```jsx
<select className="select" value={editForm.season}
    onChange={e => setEditForm({ ...editForm, season: e.target.value })}>
    <option value="">— Select season —</option>
    <option value="Summer / Hot Season">Summer / Hot Season (Mar–May)</option>
    <option value="Monsoon Season">Monsoon Season (Jun–Nov)</option>
    <option value="Winter / Cool Season">Winter / Cool Season (Dec–Feb)</option>
</select>
```

**No backend changes. No new dependencies.**

---

## Feature 2 — Geo-Weather Context (~2 hours)

Auto-resolve the field's location to lat/lon, fetch live weather from
Open-Meteo (free, no API key), and inject a `CURRENT WEATHER CONDITIONS`
block into each chat system prompt.

### Architecture

```
POST /api/chat
    → ChatService._build_context_message()
        → read field.lat, field.lon from DB
        → if NULL: resolve_coordinates(field.location)  ← Nominatim, once
                   → store.set_field_coordinates(field_id, lat, lon)
        → get_weather_context(lat, lon)    ← Open-Meteo, 60-min cache
        → append "=== CURRENT WEATHER CONDITIONS ===" to system prompt
```

---

### Step 1 — DB migration: add lat/lon to fields

In `FieldStore._migrate_schema()`:

```python
field_cols = {row[1] for row in conn.execute("PRAGMA table_info(fields)")}
if "lat" not in field_cols:
    conn.execute("ALTER TABLE fields ADD COLUMN lat REAL DEFAULT NULL")
if "lon" not in field_cols:
    conn.execute("ALTER TABLE fields ADD COLUMN lon REAL DEFAULT NULL")
```

New `FieldStore` method:

```python
def set_field_coordinates(self, field_id: int, lat: float, lon: float) -> None:
    with self._connect() as conn:
        conn.execute("UPDATE fields SET lat=?, lon=? WHERE id=?", (lat, lon, field_id))
        conn.commit()
```

Update `get_field()` / `list_fields()` SELECT to include `lat, lon` in returned dicts.

---

### Step 2 — New file: `nava_core/shared/utils/geo_context.py`

Uses **stdlib only** — no new pip installs.

```python
"""Geo-weather utilities for NAVA chat context injection."""
from __future__ import annotations
import re, time, json, urllib.request, urllib.parse
from datetime import datetime, timezone

def resolve_coordinates(location_str: str) -> tuple[float, float] | None:
    """
    Strategy 1: parse decimal coords  "10.85, 76.27"
    Strategy 2: Nominatim geocode     "Wayanad, Kerala"
    Returns (lat, lon) or None.
    """
    s = location_str.strip()
    m = re.match(r"([+-]?\d+\.?\d*)[^\d.+-]+([+-]?\d+\.?\d*)", s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    try:
        encoded = urllib.parse.quote_plus(s)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
        req = urllib.request.Request(url, headers={"User-Agent": "NAVA-AG/2.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


_weather_cache: dict = {}
_CACHE_TTL = 3600  # 60 minutes

def get_weather_context(lat: float, lon: float) -> dict | None:
    """
    Open-Meteo API (free, no key, 10k req/day).
    Cached 60 min per 0.1° grid cell.
    Returns dict{temp, humidity, precipitation, wind_speed} or None.
    """
    key = (round(lat, 1), round(lon, 1))
    now = time.time()
    if key in _weather_cache and now - _weather_cache[key][0] < _CACHE_TTL:
        return _weather_cache[key][1]
    try:
        params = (
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m"
            f",precipitation,wind_speed_10m&timezone=auto"
        )
        url = f"https://api.open-meteo.com/v1/forecast?{params}"
        with urllib.request.urlopen(url, timeout=5) as r:
            raw = json.loads(r.read())
        cur = raw["current"]
        result = {
            "temp":          cur.get("temperature_2m"),
            "humidity":      cur.get("relative_humidity_2m"),
            "precipitation": cur.get("precipitation"),
            "wind_speed":    cur.get("wind_speed_10m"),
        }
        _weather_cache[key] = (now, result)
        return result
    except Exception:
        return None


def get_field_weather_context(
    location_str: str,
    cached_lat: float | None = None,
    cached_lon: float | None = None,
) -> tuple[dict | None, float | None, float | None]:
    """Top-level entry. Returns (weather | None, lat | None, lon | None)."""
    lat, lon = cached_lat, cached_lon
    if lat is None or lon is None:
        coords = resolve_coordinates(location_str)
        if coords is None:
            return None, None, None
        lat, lon = coords
    return get_weather_context(lat, lon), lat, lon
```

---

### Step 3 — ChatService: inject weather into system prompt

In `ChatService._build_context_message()`, after existing field/crop context block:

```python
from nava_core.shared.utils.geo_context import get_field_weather_context
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

location   = (field or {}).get("location", "") or ""
cached_lat = (field or {}).get("lat")
cached_lon = (field or {}).get("lon")

if location:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(get_field_weather_context, location, cached_lat, cached_lon)
        try:
            wx, new_lat, new_lon = fut.result(timeout=5)
        except FutureTimeout:
            wx, new_lat, new_lon = None, None, None

    # Persist freshly geocoded coordinates
    if new_lat and new_lon and (cached_lat is None or cached_lon is None):
        if self.field_store and field:
            self.field_store.set_field_coordinates(field["id"], new_lat, new_lon)

    if wx:
        context_parts.append(
            f"\n=== CURRENT WEATHER CONDITIONS ===\n"
            f"Temperature: {wx['temp']}°C\n"
            f"Humidity: {wx['humidity']}%\n"
            f"Recent precipitation: {wx['precipitation']} mm\n"
            f"Wind speed: {wx['wind_speed']} km/h"
        )
```

> Import at function level to avoid circular imports.

---

### Implementation order

```
Step 1  Season dropdown (15 min — frontend only)
        Edit FieldDetail.jsx + CropDetail.jsx
        Test: create a crop → dropdown renders, value saves correctly

Step 2  DB migration (15 min — backend)
        _migrate_schema(), set_field_coordinates(), update SELECT queries
        Test: restart server → migration runs cleanly on existing DB

Step 3  geo_context.py (30 min)
        Create nava_core/shared/utils/geo_context.py
        Quick test: resolve_coordinates("Wayanad, Kerala") → (~11.6, ~76.0)

Step 4  ChatService injection (30 min)
        Add weather block to _build_context_message()
        Test: chat on crop whose field has a location → weather block visible
              in server logs / ask NAVA "what's the weather at my farm?"
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Season input | Dropdown (3 Kerala options) | Prevents typos; consistent LLM context |
| Weather API | Open-Meteo | Free, no key, no billing, GDPR-compliant |
| Geocoding | Nominatim (OSM) | Free, no key; result stored in DB |
| lat/lon storage | Two new REAL columns in `fields` | Geocoding runs once per field |
| Weather cache | 60-min in-memory per 0.1° cell | No redundant API calls within a session |
| Network timeout | 5 s in ThreadPoolExecutor | Weather fetch never delays chat response |
| Failure mode | Silent skip | Chat works normally if network is down |
| New pip installs | None | Entirely stdlib urllib + json |

---

## Future Work — Multilingual Support (deferred)

When ready, the recommended approach:

- **API:** DeepL Free tier (500k chars/month, no payment, official Python SDK, Malayalam code `"ML"`)
- **Input pipeline:** Translate ML → EN before RAG routing and LLM call
- **Output:** Instruct LLM via system prompt to respond in Malayalam (Llama-3 70B handles this natively)
- **Frontend:** EN / ML toggle pill in ChatPanel header
- **DB:** `preferred_lang TEXT DEFAULT 'en'` column in `chat_context`
- **Detection:** `langdetect` library (offline, ~3 MB) to auto-detect language of incoming message
