# Shared Utilities: `geo_context.py`, `settings.py`, `paths.py`, `logging.py`, `schemas`

> **Subfolder:** `code/`
> **Cross-references:** [technical/09_weather_and_geocoding.md](../technical/09_weather_and_geocoding.md) | [technical/10_api_and_auth_design.md](../technical/10_api_and_auth_design.md) | [02_gathi_main_and_startup.md](02_gathi_main_and_startup.md)

**Source files:**
- [`geo_context.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/shared/utils/geo_context.py)
- [`settings.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/shared/config/settings.py)
- [`paths.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/shared/utils/paths.py)
- [`logging.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/shared/utils/logging.py)

---

## `geo_context.py` — Geocoding and Weather

This file has three public functions, each with a distinct responsibility:

### `resolve_coordinates(location_str)` — Two-Strategy Geocoding

```python
_DECIMAL_RE = re.compile(r"^([+-]?\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)$")

def resolve_coordinates(location_str: str) -> tuple[float, float] | None:
    m = _DECIMAL_RE.match(s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon   # Direct parse — no network call

    # Fallback: Nominatim geocoding
    encoded = urllib.parse.quote_plus(s)
    nominatim_url = f"https://nominatim.openstreetmap.org/search?q={encoded}&format=json&limit=1"
    req = urllib.request.Request(nominatim_url, headers={"User-Agent": "NAVA-AG/2.0 (academic)"})
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None
```

**Strategy 1 — Decimal parsing:**
The regex `^([+-]?\d+\.?\d*)\s*[,\s]\s*([+-]?\d+\.?\d*)$` matches strings like `"10.85, 76.27"` or `"10.85 76.27"`. If the string matches and the values are in valid ranges (lat: -90 to 90, lon: -180 to 180), the function returns immediately without making any network call.

This matters for performance: users who enter exact decimal coordinates (common among technically-inclined farmers or those copy-pasting from Google Maps) get instant results.

**Strategy 2 — Nominatim (OpenStreetMap):**
For free-text locations like `"Wayanad, Kerala"`, the string is URL-encoded and sent to Nominatim. The `User-Agent` header is required by Nominatim's Terms of Service — bulk requests without a User-Agent are blocked. The header identifies NAVA as an academic project.

**`urllib.request` instead of `requests`:** The entire `geo_context.py` uses only Python stdlib (`urllib`, `json`). The module docstring states this explicitly. This avoids adding a production dependency for what is a simple GET request.

**"Never raises":** All Nominatim calls are wrapped in `try/except`. The function returns `None` on any failure. The caller checks for `None` and either queues a retry (background task) or surfaces a "no_coordinates" error to the frontend.

### `get_weather_context(lat, lon)` — Open-Meteo Fetch

```python
def get_weather_context(lat: float, lon: float) -> dict | None:
    params = (
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m"
        ",precipitation,wind_speed_10m&timezone=auto"
    )
    open_meteo_url = f"https://api.open-meteo.com/v1/forecast?{params}"
    with urllib.request.urlopen(open_meteo_url, timeout=5) as r:
        raw = json.loads(r.read())
    cur = raw.get("current", {})
    return {
        "temp":          cur.get("temperature_2m"),
        "humidity":      cur.get("relative_humidity_2m"),
        "precipitation": cur.get("precipitation"),
        "wind_speed":    cur.get("wind_speed_10m"),
    }
```

**Open-Meteo API parameters:**
- `current=temperature_2m,...` — requests the "current weather" endpoint (most recent observation)
- `timezone=auto` — Open-Meteo auto-detects the timezone from the coordinates, returning the data in the local timezone

**No in-process caching:** The docstring explicitly states "No in-process caching — the DB is the persistent cache." The `update_field_weather()` call in `FieldStore` sets `weather_updated_at`, and the `GET /api/weather` endpoint serves from the DB. There is no in-memory cache in `geo_context.py` itself.

### `refresh_user_weather(field_store)` — Login Batch Refresh

```python
def refresh_user_weather(field_store) -> None:
    fields = field_store.list_fields()
    for field in fields:
        lat, lon = field.get("lat"), field.get("lon")
        if lat is None or lon is None:
            continue   # no coordinates yet — skip
        try:
            wx = get_weather_context(lat, lon)
            if wx:
                field_store.update_field_weather(field["id"], ...)
        except Exception as exc:
            log.warning("Field %d failed — %s", field["id"], exc)
        time.sleep(1)  # 1-second delay between fields
```

Called from `auth.py`'s login endpoint as a `BackgroundTasks` task. The `time.sleep(1)` spreads the Open-Meteo calls over N seconds for N fields — preventing burst requests that could hit the free tier's rate limit (10,000 requests/day).

**Fields without coordinates are skipped.** Coordinates are populated when a field is created (via the `_geocode_and_fetch_weather` background task in `fields.py`). If a user creates a field without a location, or if geocoding failed, the weather refresh simply skips it.

### `get_field_weather_context()` — Legacy Single-Field Entry Point

```python
def get_field_weather_context(location_str, cached_lat=None, cached_lon=None):
    if cached_lat is not None and cached_lon is not None:
        # Use DB-cached coordinates — no Nominatim call
        pass
    else:
        coords = resolve_coordinates(location_str)
        lat, lon = coords
    weather = get_weather_context(lat, lon)
    return weather, lat, lon
```

Used by the `GET /api/weather` endpoint's fallback path (when the field has a location string but no cached coordinates in the DB). Returns a tuple `(weather_dict, lat, lon)` so the caller can simultaneously get the weather and store the coordinates in the DB.

---

## `settings.py` — Application Configuration

### The `Settings` Dataclass

```python
@dataclass(frozen=True)
class Settings:
    efficientnet_model_path: Path
    efficientnet_labels_path: Path
    torch_device: str
    confidence_threshold: float
    vnir_model_path: Path
    vnir_stress_threshold_pct: float
    hf_api_key: str
    hf_model: str
    hf_router_url: str
    ...
    yukthi_enabled: bool
    yukthi_chroma_dir: Path
    yukthi_embed_model: str
```

All settings are declared with types in the `Settings` dataclass. `frozen=True` means the settings object is immutable after construction — no code can accidentally mutate a setting at runtime.

### `@lru_cache` Singleton

```python
@lru_cache
def get_settings() -> Settings:
    m = models_dir()
    lg = logs_dir()
    return Settings(
        efficientnet_model_path=_path_env("NAVA_EFFICIENTNET_PATH", m / "EfficientNet-B0.pth"),
        ...
    )
```

`@lru_cache` on a zero-argument function creates a singleton: the first call constructs and caches the `Settings` object; all subsequent calls return the cached instance. Environment variables are read only once at startup.

**Helper functions for env parsing:**
```python
def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default  # never raise on bad env var — use default
```

All parsing functions (`_float_env`, `_int_env`, `_path_env`) silently fall back to defaults on parse errors. A malformed environment variable produces a warning in logs but doesn't crash startup.

### HuggingFace Token Forwarding

```python
_hf_key = os.getenv("HF_API_KEY", "")
if _hf_key:
    os.environ.setdefault("HF_TOKEN", _hf_key)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf_key)
```

This module-level code runs at import time. It forwards `HF_API_KEY` to `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` — the environment variables that `sentence-transformers` and `huggingface_hub` use for authenticated model downloads. Without this, users would need to set three separate env variables for the same key.

`os.environ.setdefault()` only sets the variable if it's not already set — so if the user has already configured `HF_TOKEN` separately, it's preserved.

---

## `paths.py` — Path Resolution

```python
def project_root() -> Path:
    # Go up from shared/utils/ to the project root
    return Path(__file__).parent.parent.parent.parent

def models_dir() -> Path:
    return project_root() / "models"

def logs_dir() -> Path:
    return project_root() / "logs"
```

All path references in NAVA go through these helpers, which resolve paths relative to the project root. This makes the project relocatable — moving the directory doesn't break any hardcoded paths.

---

## `logging.py` — Structured Logging

```python
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    return logger
```

All modules create their logger with `log = get_logger("module.submodule")`. The `if not logger.handlers` guard prevents duplicate handlers if the same logger is requested multiple times (which happens when modules are reloaded during development).

The format `timestamp | LEVEL | name | message` makes log output easy to filter by module name (e.g., `grep "yukthi.retriever"` shows only RAG retrieval logs).

---

## `shared/schemas/` — Pydantic API Models

The `schemas/` directory contains the Pydantic models used for API request/response validation. Key models:

**`DiagnoseResponse`:**
```python
class DiagnoseResponse(BaseModel):
    class_label: str
    class_index: int
    confidence: float
    reliability: str
    original_image_base64: Optional[str] = None
    gradcam_image_base64: Optional[str] = None
```
The two image fields are optional — they are omitted when `reliability == "UNRELIABLE"`.

**`VNIRResponse`:**
All `VNIRStats` fields serialised to the frontend, including `vs_baseline`, `vs_rolling`, `vs_prev_checkpoint`, etc. `Optional[float]` is used for fields that are `None` during calibration — Pydantic serialises `None` as JSON `null`, and the frontend renders "—" for null values.

**`ChatRequest` / `ChatResponse`:**
`ChatRequest` accepts `message`, `session_id` (optional), `field_id` (optional), `crop_id` (optional). `ChatResponse` includes `rag_used`, `rag_chunk_count`, and `rag_chunks` — the RAG metadata that powers the citation tooltips in the frontend.
