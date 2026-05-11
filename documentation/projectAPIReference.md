# NAVA NAVA — API Reference

> Complete reference for every HTTP endpoint exposed by the FastAPI backend.
> Base URL: `http://localhost:8000` (dev) or the deployed host.
> All protected routes require `Authorization: Bearer <token>` header.

---

## Authentication (`/api/auth`)

### `POST /api/auth/register`
Register a new user account.

**Body**
```json
{ "name": "Sreegovind", "email": "sg@farm.in", "password": "secret" }
```

**Response** `200`
```json
{ "token": "<session_token>", "user_id": 1, "name": "Sreegovind" }
```

---

### `POST /api/auth/login`
Log in with email and password.

**Body** `{ "email": "...", "password": "..." }`  
**Response** Same as register.

---

### `POST /api/auth/logout`
Invalidate the current session token. 🔒

**Response** `{ "status": "logged_out" }`

---

### `GET /api/auth/me`
Return the currently authenticated user. 🔒

**Response** `{ "id": 1, "name": "...", "email": "...", "location": null, "farm_type": null }`

---

## Fields (`/api/fields`) 🔒

### `GET /api/fields`
List all fields belonging to the authenticated user.

**Response** `{ "fields": [ FieldResponse, ... ] }`

---

### `POST /api/fields`
Create a new field.

**Body** `{ "name": "North Plot", "location": "...", "area": "2 acres", "soil_type": "Loamy" }`  
**Response** `FieldResponse`

---

### `PUT /api/fields`
Update field metadata. Auto-regenerates `shared_context`.

**Body** `{ "field_id": 1, "name": "...", "location": "...", "area": "...", "soil_type": "..." }`  
**Response** `FieldResponse`

---

### `POST /api/field-notes`
Save manually written field notes (user-visible; separate from auto-generated context).

**Body** `{ "field_id": 1, "notes": "Irrigation scheduled for Friday." }`  
**Response** `{ "field_id": 1, "notes": "..." }`

---

### `POST /api/field-context`
Overwrite the auto-generated shared context (internal use / refresh).

**Body** `{ "field_id": 1, "shared_context": "..." }`  
**Response** `{ "field_id": 1, "shared_context": "..." }`

---

### `GET /api/field-context/refresh`
Re-generate and store `shared_context` from current crops and field metadata.

**Query** `?field_id=1`  
**Response** `{ "field_id": 1, "shared_context": "..." }`

---

## Crops (`/api/crops`) 🔒

### `GET /api/crops?field_id=1`
List crops in a field.

### `POST /api/crops`
Create a crop. Body includes `field_id`, `name`, `variety?`, `season?`, `stage?`, `notes?`.

### `PUT /api/crops`
Update crop. Body includes `crop_id` plus optional fields.

### `DELETE /api/crops/{crop_id}`
Delete a crop and cascade-delete its plants, events, and VNIR history.

### `POST /api/crop-context`
Save manual crop notes (used by NAVA in chat context).

**Body** `{ "crop_id": 1, "notes": "Showing yellow lower leaves." }`

---

## Plants (`/api/plants`) 🔒

### `GET /api/plants?crop_id=1`
List plants in a crop.

### `POST /api/plants`
Create a plant. Body: `{ "crop_id": 1, "name": "Row-1", "description": "..." }`.

### `DELETE /api/plants/{plant_id}`
Delete a plant and all its events and VNIR history.

### `DELETE /api/plants/{plant_id}/history`
Clear event history for a plant. Optional query param `?event_type=diagnose` or `?event_type=vnir` to clear only one type.

---

## Events (`/api/events`) 🔒

### `GET /api/events`
List events with optional filters.

**Query params** `field_id`, `crop_id`, `plant_id`, `limit` (default 50)  
**Response** `{ "events": [ EventResponse, ... ] }`

`EventResponse` includes: `id`, `event_type`, `field_id`, `crop_id`, `plant_id`, `payload` (dict), `created_at`.

---

### `DELETE /api/events/{event_id}`
Delete a single event by ID (used by history section per-item delete).

**Response** `{ "status": "deleted", "event_id": 42 }`

---

## Disease Detection (`/api/diagnose`) 🔒

### `POST /api/diagnose`
Run disease detection on a leaf image.

**Body** multipart/form-data:
- `image` — image file
- `plant_id` — integer
- `crop_id` — integer (optional)
- `field_id` — integer (optional)

**Response** `DiagnoseResponse`

```json
{
  "class_label": "tomato_late_blight",
  "class_index": 12,
  "confidence": 0.9989,
  "reliability": "RELIABLE",
  "original_image_base64": "data:image/png;base64,...",
  "gradcam_image_base64": "data:image/png;base64,..."
}
```

> `original_image_base64` and `gradcam_image_base64` are **full data URIs**.
> On unreliable predictions both image fields are `null`.

---

## VNIR Monitoring (`/api/vnir-*`) 🔒

### `POST /api/vnir-upload`
Run VNIR stress analysis on a plant image.

**Body** multipart/form-data: `image`, `plant_id`, `crop_id?`, `field_id?`

**Response** `VNIRResponse`

```json
{
  "plant_id": "3",
  "leaf_state": "GREEN",
  "status": "Calibrating (3/5)",
  "avg_green": 142.3,
  "avg_vnir": 198.7,
  "ratio": 0.7163,
  "baseline": null,
  "rolling_avg": null,
  "prev_checkpoint_avg": null,
  "global_avg": 0.7163,
  "vs_baseline": null,
  "vs_global": 0.0,
  "vs_rolling": null,
  "vs_prev_checkpoint": null,
  "hsv_image_base64": "data:image/png;base64,...",
  "vnir_image_base64": "data:image/png;base64,..."
}
```

---

### `POST /api/vnir-clear`
Clear all VNIR ratio history for a plant (resets calibration).

**Body** form: `plant_id`

---

## Chat (`/api/chat`) 🔒

### `POST /api/chat`
Send a message and receive an NAVA reply.

**Body**
```json
{
  "message": "What disease was found on my tomato plants?",
  "session_id": "abc123",
  "field_id": 1,
  "crop_id": 2
}
```

**Response** `{ "session_id": "abc123", "reply": "...", "error": null }`

The service injects `get_rich_crop_context(crop_id)` as a system message before the LLM call.
This includes field metadata, all crops in the field (with sibling health snapshots),
current crop details, and full plant event history with priority rules.

---

### `POST /api/chat/history`
Retrieve chat messages for a session (for UI restoration on refresh).

**Body** `{ "session_id": "abc123", "limit": 50 }`  
**Response** `{ "session_id": "...", "messages": [ { "role": "user", "content": "...", "created_at": "..." }, ... ] }`

---

### `POST /api/chat/summary`
Retrieve the memory summaries for display.

**Body** `{ "session_id": "abc123" }`  
**Response** `{ "session_id": "...", "summary": "..." }`

---

### `POST /api/chat/clear`
Delete a session and all its messages and summaries.

**Body** `{ "session_id": "abc123" }`  
**Response** `{ "session_id": "...", "status": "cleared" }`

---

## Static & Utility

| Route | Description |
|---|---|
| `GET /api/logo` | Returns `NAVA-Logo.png` as `image/png` |
| `GET /` and all non-`/api` routes | Serves `dist/index.html` (SPA fallback) |
| `GET /assets/*` | Serves Vite-built JS/CSS bundles |
