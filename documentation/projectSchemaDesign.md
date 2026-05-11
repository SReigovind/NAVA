# NAVA NAVA — Schema Design

> This document describes every database table, Pydantic schema, and API contract.
> It is the single source of truth for data shapes across the backend.

---

## Database Overview

There are **two SQLite databases** per deployment:

| Database | Path | Scope |
|---|---|---|
| `users.db` | `logs/users/users.db` | Shared — user accounts and sessions |
| `user_data.db` | `logs/users/user_{id}/user_data.db` | Per-user — all domain data |
| `mozhi_sessions.db` | `logs/mozhi_sessions.db` | Shared — chat messages and summaries |

---

## `users.db` — UserStore

### Table: `users`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `name` | TEXT NOT NULL | Display name |
| `email` | TEXT UNIQUE NOT NULL | Login identifier |
| `hashed_password` | TEXT NOT NULL | PBKDF2-SHA256, 100k iterations |
| `location` | TEXT | Optional onboarding field |
| `farm_type` | TEXT | Optional onboarding field |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

### Table: `sessions`

| Column | Type | Notes |
|---|---|---|
| `token` | TEXT PK | `secrets.token_urlsafe(32)` |
| `user_id` | INTEGER FK | References `users.id` |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

---

## `user_data.db` — FieldStore

All tables below live in the per-user database.

### Table: `fields`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `name` | TEXT NOT NULL | |
| `location` | TEXT | |
| `area` | TEXT | Human-readable e.g. "2 acres" |
| `soil_type` | TEXT | Selected from dropdown |
| `shared_context` | TEXT | **Auto-generated** — assembled from crops+events; never shown in UI |
| `field_notes` | TEXT | **Manual** — user-written notes; shown and editable in UI |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

> **Key design decision:** `shared_context` and `field_notes` are intentionally separate.
> `shared_context` is regenerated on every crop change and fed silently to the LLM.
> `field_notes` is only ever written by the user and displayed as-is.

### Table: `crops`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `field_id` | INTEGER FK → `fields.id` | |
| `name` | TEXT NOT NULL | |
| `variety` | TEXT | |
| `season` | TEXT | e.g. "Kharif 2026" |
| `stage` | TEXT | Enum: Seedling, Vegetative, Flowering, Fruiting, Maturity, Harvested |
| `notes` | TEXT | **Manual crop notes** — user-written; shown in Overview and used in context |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

### Table: `plants`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `crop_id` | INTEGER FK → `crops.id` | |
| `name` | TEXT NOT NULL | User-assigned identifier e.g. "Row-1" |
| `description` | TEXT | Optional notes about this specific plant |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

Plants are shared between Disease Detection and VNIR Monitoring within the same crop.

### Table: `events`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `event_type` | TEXT | `"diagnose"` or `"vnir"` |
| `field_id` | INTEGER | Denormalized for fast field-level queries |
| `crop_id` | INTEGER | Denormalized for crop-level context building |
| `plant_id` | INTEGER FK → `plants.id` | Primary scoping key |
| `payload` | TEXT | JSON blob — see payload schemas below |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

#### Event payload — `diagnose`

```json
{
  "plant_name": "Row-1",
  "class_label": "tomato_late_blight",
  "class_index": 12,
  "confidence": 0.9989,
  "reliability": "RELIABLE"
}
```

#### Event payload — `vnir`

```json
{
  "plant_name": "Row-1",
  "status": "OK",
  "leaf_state": "GREEN",
  "ratio": 0.7423,
  "vs_baseline": -1.2,
  "vs_global": 0.8,
  "vs_rolling": -0.5,
  "vs_prev_checkpoint": -1.0
}
```

### Table: `vnir_history`

Stores raw ratio readings for statistical computation (baseline, rolling average, etc.).

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `plant_id` | INTEGER FK → `plants.id` | |
| `ratio` | REAL | VNIR ratio value |
| `avg_green` | REAL | |
| `avg_vnir` | REAL | |
| `status` | TEXT | Pipeline-assigned status at time of scan |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

---

## `mozhi_sessions.db` — SessionStore

### Table: `chat_messages`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | Used to track last summarized position |
| `session_id` | TEXT | Client-managed UUID string |
| `role` | TEXT | `"user"` or `"assistant"` |
| `content` | TEXT | |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

### Table: `chat_summaries`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `session_id` | TEXT | |
| `level` | INTEGER | 1 = recent summary batch, 2 = long-term rollup |
| `content` | TEXT | Bullet-format memory |
| `created_at` | TEXT | `CURRENT_TIMESTAMP` |

### Table: `chat_state`

| Column | Type | Notes |
|---|---|---|
| `session_id` | TEXT PK | |
| `last_summarized_id` | INTEGER | Highest message `id` already included in a summary |

### Table: `session_context`

| Column | Type | Notes |
|---|---|---|
| `session_id` | TEXT PK | |
| `field_id` | INTEGER | Bound field (nullable) |
| `crop_id` | INTEGER | Bound crop (nullable) |

---

## Pydantic Schemas

All schemas live in `nava_core/shared/schemas/` and are exported from `__init__.py`.

### Auth

| Schema | Fields |
|---|---|
| `AuthRegisterRequest` | name, email, password, location?, farm_type? |
| `AuthLoginRequest` | email, password |
| `AuthResponse` | token, user_id, name |
| `UserResponse` | id, name, email, location?, farm_type? |
| `OnboardingRequest` | location, farm_type |

### Fields / Crops / Plants

| Schema | Key Fields |
|---|---|
| `FieldCreateRequest` | name, location?, area?, soil_type?, shared_context? |
| `FieldUpdateRequest` | field_id, name?, location?, area?, soil_type? |
| `FieldResponse` | id, name, location, area, soil_type, shared_context, **field_notes**, created_at |
| `FieldContextRequest` | field_id, shared_context (auto-context update) |
| `FieldContextResponse` | field_id, shared_context |
| `CropCreateRequest` | field_id, name, variety?, season?, stage?, notes? |
| `CropUpdateRequest` | crop_id, name?, variety?, season?, stage?, notes? |
| `CropResponse` | id, field_id, name, variety, season, stage, notes, created_at |
| `CropContextRequest` | crop_id, notes |
| `PlantCreateRequest` | crop_id, name, description? |
| `PlantResponse` | id, crop_id, name, description, created_at |

### Events

| Schema | Fields |
|---|---|
| `EventResponse` | id, event_type, field_id, crop_id, **plant_id**, payload (dict), created_at |
| `EventListResponse` | events: list[EventResponse] |

> `plant_id` was added in a schema fix — it was previously omitted causing the frontend
> to receive `undefined` and fail plant-level event filtering.

### Detection & VNIR

| Schema | Key Fields |
|---|---|
| `DiagnoseResponse` | class_label, class_index, confidence, reliability, original_image_base64?, gradcam_image_base64? |
| `VNIRResponse` | plant_id, leaf_state, status, avg_green, avg_vnir, ratio, baseline, rolling_avg, prev_checkpoint_avg, global_avg, vs_baseline, vs_global, vs_rolling, vs_prev_checkpoint, hsv_image_base64?, vnir_image_base64? |

> Image fields contain **full data URIs** (`data:image/png;base64,...`).
> The frontend must use them as `src={value}` directly — do not prepend the prefix again.

### Chat

| Schema | Fields |
|---|---|
| `ChatRequest` | message, session_id?, field_id?, crop_id? |
| `ChatResponse` | session_id, reply, error? |
| `ChatHistoryRequest` | session_id, limit? |
| `ChatHistoryResponse` | session_id, messages: list[{role, content, created_at}] |
| `ChatSummaryRequest` | session_id |
| `ChatSummaryResponse` | session_id, summary? |
| `ChatClearRequest` | session_id |
| `ChatClearResponse` | session_id, status |

---

## Schema Migration Strategy

All database schema changes are handled by `FieldStore._migrate_schema()` which runs on
every startup. The pattern:

```python
cols = {row[1] for row in conn.execute("PRAGMA table_info(table_name)").fetchall()}
if "new_column" not in cols:
    conn.execute("ALTER TABLE table_name ADD COLUMN new_column TEXT")
```

This makes deploys safe — existing databases are upgraded in place without data loss.
