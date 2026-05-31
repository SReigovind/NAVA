# Gathi Routers: `auth.py` and `fields.py`

> **Subfolder:** `code/`
> **Cross-references:** [03_gathi_deps.md](03_gathi_deps.md) | [technical/09_weather_and_geocoding.md](../technical/09_weather_and_geocoding.md) | [technical/08_database_design.md](../technical/08_database_design.md) | [12_shared_storage.md](12_shared_storage.md)

**Source files:**
- [`auth.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/routers/auth.py)
- [`fields.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/routers/fields.py)

---

## `auth.py` — Authentication Router

All routes under `/api/auth/`. Handles user registration, login, session management, profile management, and account deletion.

### Helper Functions

**`_to_user_response(user: UserRecord) -> UserResponse`**
A small conversion helper that maps the internal `UserRecord` dataclass to the API's `UserResponse` Pydantic model. This prevents leaking internal fields (like `password_hash` or `db_path`) to the API response — `UserResponse` only includes safe public fields.

**`_preload_models()`**
```python
def _preload_models():
    from nava_core.gathi.api.deps import get_predictor, get_vnir_pipeline
    try:
        get_predictor()
        get_vnir_pipeline()
    except Exception as e:
        log.error("Failed to preload models: %s", e)
```
Called as a background task on login and register. Triggers the `@lru_cache` construction of the EfficientNet predictor and VNIR pipeline if they haven't been loaded yet. Since startup.py already loads them in background threads, this is usually a no-op — but it guarantees they're loaded from the auth event path as a second safety net.

**`_models_loaded() -> bool`**
```python
def _models_loaded() -> bool:
    return (
        get_predictor.cache_info().currsize > 0
        and get_vnir_pipeline.cache_info().currsize > 0
    )
```
`lru_cache` objects expose `cache_info()` which includes `currsize` — the number of cached entries. If both are > 0, the models are loaded. This guard prevents spawning a redundant background thread on every `/api/auth/me` call (which may be polled by the frontend as a keep-alive).

**`_refresh_user_weather(user_id: int) -> None`**
```python
def _refresh_user_weather(user_id: int) -> None:
    store = get_user_store()
    user = store.get_user(user_id)
    fstore = FieldStore(Path(user.db_path))
    refresh_user_weather(fstore)
```
Called as a background task at login. Opens the user's FieldStore and calls `refresh_user_weather()` (from `geo_context.py`), which iterates all fields with stored (lat, lon) and fetches fresh weather from Open-Meteo with a 1-second delay between calls.

### Endpoints

**`POST /api/auth/register`**
```python
@router.post("/register", response_model=AuthResponse)
def register(request: AuthRegisterRequest, bg_tasks: BackgroundTasks) -> AuthResponse:
    store = get_user_store()
    try:
        user = store.create_user(request.name, request.email, request.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = store.create_session(user.id)
    bg_tasks.add_task(_preload_models)
    return AuthResponse(token=token, user=_to_user_response(user))
```

`sqlite3.IntegrityError` is caught explicitly because `users.email` has a `UNIQUE` constraint. If a duplicate email is submitted, SQLite raises `IntegrityError` — which we translate to a user-friendly 400.

`ValueError` from `create_user()` covers domain validation (e.g., empty name, invalid email format).

On success: creates a session token, fires the `_preload_models` background task, returns the token and user profile. The frontend stores the token in `localStorage`.

**`POST /api/auth/login`**
```python
@router.post("/login", response_model=AuthResponse)
def login(request: AuthLoginRequest, bg_tasks: BackgroundTasks) -> AuthResponse:
    user = store.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = store.create_session(user.id)
    bg_tasks.add_task(_preload_models)
    bg_tasks.add_task(_refresh_user_weather, user.id)
    return AuthResponse(token=token, user=_to_user_response(user))
```

Two background tasks fire at login:
1. `_preload_models` — ensures ML models are ready for the first scan
2. `_refresh_user_weather` — fetches fresh weather for all this user's fields

Both are fire-and-forget. The login response returns immediately; these tasks complete in the background (typically 2–10 seconds after login).

**`POST /api/auth/logout`**
```python
@router.post("/logout")
def logout(authorization: str | None = Header(None), user: UserRecord = Depends(require_user)) -> dict:
    token = _extract_token(authorization)
    if token:
        get_user_store().delete_session(token)
    return {"status": "logged_out"}
```

Deletes the session token from the database. Any subsequent request using that token will receive a 401. The `require_user` dependency validates the token first — you can't log out without a valid token (which prevents CSRF-style forced logout attacks).

**`GET /api/auth/me`**
Returns the current user's profile. Used by the frontend to verify session validity and display the user's name. The `_models_loaded()` guard prevents redundant background task spawning on frequent `/me` calls.

**`PUT /api/auth/me`** — Update display name.

**`PUT /api/auth/password`** — Change password with current password verification.

**`DELETE /api/auth/me`** — Delete account (deletes the user's DB file and removes the user row).

---

## `fields.py` — Fields, Crops, Plants, Events Router

All routes under `/api/`. This is the largest router: 20 endpoints managing the full farm data hierarchy.

### The `_refresh_field_context()` Helper

```python
def _refresh_field_context(store, field_id: int) -> None:
    ctx = store.auto_generate_field_context(field_id)
    store.update_field_context(field_id, ctx)
```

Called after every mutation (create/update/delete crop, create/delete plant, delete event). Regenerates the `fields.shared_context` text that is injected into LLM prompts. This ensures the chat assistant always sees an up-to-date summary of the farm.

This is synchronous (not a background task) because `shared_context` is needed immediately for subsequent reads. A race condition where a user opens the crop chat before the context is updated would produce stale advice.

### The `_geocode_and_fetch_weather()` Background Task

```python
def _geocode_and_fetch_weather(db_path: str, field_id: int) -> None:
    store = FieldStore(Path(db_path))
    field = store.get_field(field_id)
    location = (field.get("location") or "").strip()

    lat = field.get("lat")
    lon = field.get("lon")
    if lat is None or lon is None:
        coords = resolve_coordinates(location)  # Nominatim call
        lat, lon = coords
        store.set_field_coordinates(field_id, lat, lon)

    wx = get_weather_context(lat, lon)  # Open-Meteo call
    store.update_field_weather(field_id, wx["temp"], wx["humidity"], ...)
```

Why does this function accept `db_path: str` instead of a `FieldStore`? Because it runs as a `BackgroundTasks` task — after the HTTP response has been sent. The `FieldStore` from the request handler is no longer valid after the response (it was constructed for that request's lifetime). A fresh `FieldStore` is constructed inside the background task.

**Coordinate caching logic:**
The background task first checks if (lat, lon) already exist in the DB. If they do (from a previous geocode run), the Nominatim call is skipped — only the weather fetch runs. This prevents hitting Nominatim on every field edit, even if only the field name was changed (not the location).

**Location changed → coordinates invalidated:**
In the `update_field` endpoint:
```python
if request.location is not None:
    old = store.get_field(request.field_id)
    if old and (request.location or "").strip() != (old.get("location") or "").strip():
        store.set_field_coordinates(request.field_id, None, None)  # invalidate
```
If the user edits the location string, the stored (lat, lon) are set to NULL. The next `_geocode_and_fetch_weather` call will re-geocode the new location.

### Field CRUD Endpoints

**`GET /api/fields`** — List all fields for the authenticated user. Response includes weather columns (`weather_temp`, `weather_humidity`, etc.).

**`POST /api/fields`** — Create field. Fires `_geocode_and_fetch_weather` as a background task if a location is provided.

**`PUT /api/fields`** — Update field. Invalidates coordinates if location changed. Fires `_geocode_and_fetch_weather` as a background task.

**`DELETE /api/fields/{field_id}`** — Cascading delete. Calls `store.delete_field(field_id)`, which:
1. Deletes VNIR history for all plants in all crops in this field
2. Deletes events for all plants in all crops in this field
3. Deletes all plants in all crops
4. Deletes all crops
5. Deletes the field itself

The delete is ordered child-first to avoid foreign key violations.

### Crop and Plant CRUD

All crop endpoints (`/api/crops`) follow the same pattern:
1. Validate the operation (check the entity exists)
2. Perform the mutation via `FieldStore`
3. Call `_refresh_field_context()` to update the LLM context
4. Return the updated entity

**`DELETE /api/plants/{plant_id}/history`** — Special endpoint to clear a plant's scan history (events and VNIR records) without deleting the plant itself. Useful for starting fresh after a disease treatment. The optional `event_type` query parameter allows selective clearing (delete only `diagnose` events, or only `vnir` records).

### Events Endpoints

**`GET /api/events`** — Flexible event query: filter by field_id, crop_id, plant_id, with a configurable limit. Returns the full event list for the fields dashboard's recent activity section.

**`DELETE /api/events/{event_id}`** — Delete a single event. Re-generates the field context after deletion so the chat assistant no longer sees the deleted scan in its context.

### Context Endpoints

**`GET /api/field-context`** and **`POST /api/field-context`** — Read or write the `shared_context` field manually. The `GET /api/field-context/refresh` endpoint triggers a re-generation of the auto-context from the current DB state.

**`POST /api/field-notes`** — Save manual notes the farmer has typed in the FieldDetail view. Stored separately in `fields.field_notes` (not `shared_context` — manual notes are shown in the UI, auto-context is injected silently into LLM prompts).
