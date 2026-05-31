# Weather and Geocoding

> **Subfolder:** `technical/`
> **Cross-references:** [08_database_design.md](08_database_design.md) | [10_api_and_auth_design.md](10_api_and_auth_design.md) | [code/13_shared_config_utils_frontend.md](../code/13_shared_config_utils_frontend.md) | [code/04_gathi_routers_auth_and_fields.md](../code/04_gathi_routers_auth_and_fields.md)

---

## Why Weather Context Matters

When a farmer asks NAVA "Should I spray fungicide today?", the answer depends critically on the weather. Fungicides applied before or during rain are washed off. High humidity increases fungal infection risk and argues for spraying. High wind speed makes spraying impractical. Temperature affects both plant physiology and chemical effectiveness.

Without weather context, NAVA can only give generic timing advice. With it, NAVA can give advice calibrated to the actual current conditions at the farmer's field.

---

## Two External Services

NAVA integrates with two entirely free, API-key-free services:

### Nominatim (OpenStreetMap Geocoding)
Nominatim is the geocoding service powering OpenStreetMap. Given a place name or address string, it returns the latitude and longitude. NAVA uses it to resolve the user-entered field location ("Wayanad, Kerala") to coordinates.

**API call:**
```
GET https://nominatim.openstreetmap.org/search?q=Wayanad+Kerala&format=json&limit=1
```

Response contains `lat` and `lon` as string-formatted floats.

**Why Nominatim instead of Google Maps Geocoding API?**
- Nominatim is completely free, no API key required, no usage limits for reasonable traffic
- OpenStreetMap's coverage in rural South Asia is excellent for district and village-level place names
- No dependency on a commercial API that could change pricing or terms

**Limitations:**
- Very specific sub-village location names may not be found
- Nominatim's rate limiting requires respecting their usage policy: no more than 1 request per second

### Open-Meteo (Weather API)
Open-Meteo provides free weather data (historical and current) for any latitude/longitude, with no API key. NAVA uses it to fetch the current weather at a field's coordinates.

**API call:**
```
GET https://api.open-meteo.com/v1/forecast?
    latitude={lat}&longitude={lon}&
    current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m
```

Response: a JSON object with a `current` block containing `temperature_2m` (°C), `relative_humidity_2m` (%), `precipitation` (mm), and `wind_speed_10m` (km/h).

**Why Open-Meteo instead of OpenWeatherMap?**
- Open-Meteo requires no API key and has no free tier limitations (beyond 10,000 requests/day, well beyond NAVA's expected usage)
- The API returns SI units (°C, mm, km/h) without requiring unit parameter specification
- Excellent global coverage using ERA5 reanalysis data for historical weather and ECMWF models for current/forecast

---

## The Geocoding and Weather Lifecycle

Weather data flows through three stages in NAVA:

### Stage 1: Geocoding (One-Time, on Field Creation)

When a user creates or edits a field with a location string, a background task fires:

```python
background_tasks.add_task(_geocode_and_fetch_weather, field_id, location, field_store)
```

`_geocode_and_fetch_weather()` calls Nominatim to resolve the location to (lat, lon), then immediately calls Open-Meteo to fetch initial weather. Both results are stored in the field's DB columns. This is a background task — the API returns the field data to the user immediately, while geocoding happens asynchronously. The weather data will appear within 2–3 seconds.

**Why store lat/lon in the DB?** Once resolved, coordinates never need to be re-looked up for the same location. Nominatim's rate limits make repeated geocoding calls wasteful. Storing (lat, lon) means all subsequent weather fetches use the coordinates directly, skipping geocoding entirely.

**Why not geocode every weather refresh?** Geocoding converts a place name to coordinates. The coordinates of "Wayanad, Kerala" don't change. Re-geocoding wastes a Nominatim call and risks hitting their rate limits with high-traffic usage.

### Stage 2: Login Refresh

When a user logs in, a background task refreshes weather for all their fields:

```python
background_tasks.add_task(_refresh_weather_on_login, user, user_store, ...)
```

`refresh_user_weather()` iterates over all the user's fields. For each field with stored (lat, lon), it calls Open-Meteo to get fresh weather data and updates the weather columns in the DB. A 1-second `asyncio.sleep()` delay between fields is observed to avoid overwhelming Open-Meteo if a user has many fields.

**Why refresh at login?** Login is a natural synchronisation point: the user is about to start working with their farm data, so fresh weather is maximally useful. It is also a moment when the user doesn't mind a few seconds of background activity.

**Why 1-second delay between fields?** Open-Meteo's terms of service request reasonable usage. With 1 second between calls, a user with 5 fields takes 5 seconds to refresh — imperceptible since this runs in the background while the user navigates to their dashboard.

### Stage 3: Manual Refresh

The `POST /api/weather/refresh?field_id={id}` endpoint immediately fetches fresh weather for a specific field (using its stored lat/lon) and returns the updated values. This is triggered by the ↻ refresh button in the WeatherStrip component.

**Why a manual refresh at all, if login refreshes automatically?** A user who has been logged in for several hours (watching their crops over a day) might want to see updated weather without logging out and back in. The manual refresh gives them control.

---

## Background Task Architecture

Both the geocoding-on-create and login weather refresh use FastAPI's `BackgroundTasks`:

```python
@router.post("/api/fields")
async def create_field(
    ...,
    background_tasks: BackgroundTasks,
):
    field = store.create_field(...)
    background_tasks.add_task(_geocode_and_fetch_weather, field.id, field.location, store)
    return field
```

FastAPI's `BackgroundTasks` executes the task after the HTTP response has been sent. The user sees their field created immediately; the geocoding happens in the background.

**Why not `asyncio.create_task()`?**
`create_task()` creates a coroutine that runs on the event loop — which is shared with all request handlers. If the geocoding HTTP call blocks (even briefly), it blocks all other requests. FastAPI's `BackgroundTasks` uses a thread pool for blocking I/O (HTTP calls), preventing event loop blocking.

**Why not Celery or a task queue?**
Celery requires a Redis or RabbitMQ broker — additional infrastructure. For NAVA's single-server deployment, adding a message broker would add operational complexity without benefit. FastAPI's built-in background tasks handle the requirement without external dependencies.

---

## Error Handling Philosophy

Both Nominatim and Open-Meteo calls are wrapped in try/except. If either service is unavailable or returns an error:
- The geocoding or weather fetch silently fails
- The field record is updated with whatever data was successfully retrieved (e.g., lat/lon even if weather failed)
- The user is not notified — they simply see "Weather unavailable" or a stale timestamp in the WeatherStrip

This is the "let it fail silently" philosophy for non-critical background data. Weather context is valuable for chat advice but not essential for core functionality (disease detection, VNIR monitoring, farm management). A temporary API outage should not degrade the primary user experience.

---

## Logging for Manual Debugging

`geo_context.py` includes detailed logging at each step of the geocoding and weather fetch process:

```
[geo_context] Resolving location: "Wayanad, Kerala"
[geo_context] Nominatim response: lat=11.6854, lon=76.0423
[geo_context] Fetching weather for lat=11.6854, lon=76.0423
[geo_context] Weather response: temp=24.2°C, humidity=78%, precip=1.2mm, wind=8.3km/h
[geo_context] Stored weather for field_id=3
```

These logs go to the application logger (configurable via `logging.py`) and appear in the server terminal output. They were added to allow manual verification of the full geocoding and weather pipeline without requiring a dedicated test suite.
