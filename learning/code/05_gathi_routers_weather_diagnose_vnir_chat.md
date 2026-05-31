# Gathi Routers: `weather.py`, `diagnose.py`, `vnir.py`, `chat.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/02_disease_detection_pipeline.md](../technical/02_disease_detection_pipeline.md) | [technical/04_vnir_stress_monitoring.md](../technical/04_vnir_stress_monitoring.md) | [technical/06_llm_and_prompt_engineering.md](../technical/06_llm_and_prompt_engineering.md) | [04_gathi_routers_auth_and_fields.md](04_gathi_routers_auth_and_fields.md)

**Source files:**
- [`weather.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/routers/weather.py)
- [`diagnose.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/routers/diagnose.py)
- [`vnir.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/routers/vnir.py)
- [`chat.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/routers/chat.py)

---

## `weather.py` — Weather Router

Two endpoints: one for reading (with a lazy-fetch fallback), one for manual refresh.

### `GET /api/weather?field_id={id}`

```python
@router.get("/weather")
def get_weather(field_id: int, user: UserRecord = Depends(require_user)) -> dict:
    store = field_store_for_user(user)
    field = store.get_field(field_id)

    location = (field.get("location") or "").strip()
    if not location:
        return {"error": "no_location"}

    if field.get("weather_updated_at") is not None:
        return {
            "temp":          field["weather_temp"],
            "humidity":      field["weather_humidity"],
            "precipitation": field["weather_precipitation"],
            "wind_speed":    field["weather_wind_speed"],
            "updated_at":    field["weather_updated_at"],
            "location":      location,
        }

    # No DB value yet — try to fetch now
    ...
```

**The two-path structure:**
1. **Happy path (DB cached):** `weather_updated_at` is not NULL → return DB values immediately. No network call. This is the path taken on virtually every request after login.
2. **Fallback path (no DB value):** The field has coordinates but no weather yet (edge case: user opens the WeatherStrip before the login background task has completed). A synchronous Open-Meteo call is made, the result is stored, and returned.

**Error codes (not HTTP errors — deliberate):**
- `{"error": "no_location"}` — field has no location string
- `{"error": "no_coordinates"}` — location exists but Nominatim hasn't resolved it yet
- `{"error": "unavailable"}` — Open-Meteo call failed

**Why return `{"error": ...}` instead of HTTP 4xx?** These are not client errors — the field exists, the request is valid. They are informational states about the weather data lifecycle. The frontend WeatherStrip renders a muted "Weather unavailable" placeholder for any error response without logging a console error.

### `POST /api/weather/refresh?field_id={id}`

```python
@router.post("/weather/refresh")
def refresh_weather(field_id: int, user: UserRecord = Depends(require_user)) -> dict:
    ...
    wx = get_weather_context(lat, lon)  # synchronous Open-Meteo call
    store.update_field_weather(field_id, wx["temp"], ...)
    refreshed = store.get_field(field_id)
    return {
        "temp": wx["temp"],
        ...,
        "updated_at": refreshed.get("weather_updated_at"),  # exact DB timestamp
    }
```

Manual refresh endpoint called by the ↻ button in WeatherStrip. Makes a synchronous Open-Meteo call (typically 200–500ms), updates the DB, and returns the fresh values.

**Why re-read `updated_at` from DB?** `update_field_weather()` sets the timestamp using `CURRENT_TIMESTAMP` in SQL. Rather than computing the timestamp in Python and risk a millisecond discrepancy between what's returned and what's stored, the endpoint re-reads the field from the DB after the update to get the exact stored timestamp.

---

## `diagnose.py` — Disease Detection Router

Single endpoint: `POST /api/diagnose`.

```python
@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    image: UploadFile = File(...),
    plant_id: int = Form(...),
    crop_id: int | None = Form(None),
    field_id: int | None = Form(None),
    user: UserRecord = Depends(require_user),
) -> DiagnoseResponse:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")

    store = field_store_for_user(user)
    plant = store.get_plant(plant_id)

    pil_image = load_image_from_bytes(data)
    predictor = get_predictor()

    result, cam_image = predictor.predict_with_cam(pil_image)

    event_payload = {
        "plant_name": plant["name"],
        "class_label": result.class_label,
        "confidence": result.confidence,
        "reliability": result.reliability,
    }
    store.add_event(event_type="diagnose", field_id=field_id, crop_id=..., plant_id=plant_id, payload=event_payload)
    _refresh_field_context(store, effective_field_id)

    if result.reliability == "UNRELIABLE":
        return DiagnoseResponse(..., no image fields)

    return DiagnoseResponse(..., original_image_base64=..., gradcam_image_base64=...)
```

**`multipart/form-data` instead of JSON:**
The request must carry both an image file (`UploadFile`) and numeric IDs. This requires multipart form encoding rather than JSON. `File(...)` and `Form(...)` are FastAPI's multipart handling types.

**`async def` for file reading:**
`await image.read()` is the async file read. Because this involves I/O (reading the uploaded file from the request body), the endpoint is `async def` so it doesn't block the event loop.

**Single forward pass (not two):**
The comment in the code is important: earlier versions called `predict()` first (fast, no grad) to check reliability, then `predict_with_cam()` only if reliable. That meant two full forward passes for every reliable diagnosis. The current version calls only `predict_with_cam()`, which always computes CAM. If the result is UNRELIABLE, the `cam_image` is simply discarded. This is one forward pass in all cases.

**Why always compute CAM even if UNRELIABLE?** The cost of running Grad-CAM (even when discarding the result) is small compared to the simplicity gained by having a single code path. The previous two-pass design also had a subtle bug: the first pass and second pass could theoretically give different results (due to non-determinism in PyTorch), creating inconsistency between the reliability verdict and the CAM image.

---

## `vnir.py` — VNIR Stress Monitoring Router

Two endpoints: `POST /api/vnir-upload` (the main scan) and `POST /api/vnir-clear` (clear history).

### `POST /api/vnir-upload`

```python
history_ratios = store.get_vnir_ratios(plant_id)
stats, hsv_image, vnir_image = pipeline.process_image(pil_image, plant["name"], history_ratios)

if stats.leaf_state == "GREEN":
    store.add_vnir_reading(plant_id, stats.ratio, ...)

store.add_event(event_type="vnir", ..., payload={
    "status": stats.status,
    "leaf_state": stats.leaf_state,
    "ratio": stats.ratio,
    ...
})
```

**The `history_ratios` parameter:**
The VNIR pipeline needs the plant's scan history to compute the baseline mean, rolling mean, and comparison statistics. The history is fetched from `get_vnir_ratios(plant_id)` before calling `pipeline.process_image()`. This means the VNIRPipeline itself is stateless — it doesn't know about the database. All state is fetched before the call and passed in explicitly. This design makes the VNIRPipeline easy to test in isolation.

**The `leaf_state == "GREEN"` guard for history persistence:**
Critical design decision: only GREEN scans (valid leaf detected, NIR estimated) are stored in the `vnir_history` timeseries table. YELLOW_BROWN and No Leaf Detected scans are still written as events (for the UI's scan history log) but do not contribute to the statistical baseline.

**Why?** If a failed scan (no leaf detected) added ratio=0.0 to the history, all subsequent baseline calculations would be corrupted — the mean would be pulled toward zero, and every subsequent healthy scan would appear to be far above the (wrong) baseline. This would generate false CRITICAL alerts.

The event record for non-GREEN scans preserves the diagnostic information (the farmer can see that the scan failed) without corrupting the statistical model.

**Response fields include five comparison values:**
- `ratio` — current NIR/Green ratio
- `baseline` — mean of first 5 valid scans
- `rolling_avg` — rolling mean of last 5
- `vs_baseline` — % change from baseline
- `vs_rolling` — % change from rolling mean

These values populate the detailed statistics card in the MonitorPanel.

---

## `chat.py` — Chat Router

The simplest router: 4 endpoints, each delegating directly to `ChatService`.

### `POST /api/chat` — The Main Chat Endpoint

```python
@router.post("", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest, bg_tasks: BackgroundTasks,
         user: UserRecord = Depends(require_user)) -> ChatResponse:
    service = chat_service_for_user(user, request)
    result = service.chat(body.message, body.session_id, field_id=body.field_id, crop_id=body.crop_id)
    bg_tasks.add_task(service._summarize_if_needed, result.session_id)
    return ChatResponse(...)
```

**Note `def` not `async def`:**
Although the LLM API call within `service.chat()` is an HTTP request, `ChatService.chat()` makes it synchronously using the `requests` library (not `aiohttp`). This makes the route handler `def` (synchronous). FastAPI runs `def` handlers in a thread pool automatically — they do not block the event loop.

**Why `requests` instead of `aiohttp`?** The `ChatService` code predates the decision to make it async. The `requests` library is simpler and well-understood. For a small-scale deployment, the thread pool model is adequate. Switching to `aiohttp` for true async LLM calls is noted in futureWork.

**`_summarize_if_needed` as a background task:**
After the chat response is sent, `_summarize_if_needed()` checks whether the session has accumulated enough unsummarised messages to trigger a compression. If so, it calls the 8B model to generate a summary, stores it, and updates the `last_summarized_id` pointer. This prevents summarisation latency from being felt by the user — the user sees their response immediately; the summarisation happens in the background.

### Other Chat Endpoints

**`POST /api/chat/clear`** — Clears all messages and summaries for a session (used by the "Start fresh" button in ChatPanel).

**`POST /api/chat/history`** — Returns the last N messages for a session (used by the frontend to display conversation history on page load).

**`POST /api/chat/summary`** — Returns the formatted summary display for a session (used by the memory indicator in ChatPanel, showing the user that NAVA has summarised the conversation).
