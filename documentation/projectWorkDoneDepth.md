# NAVA NAVA — Detailed Research Diary

> Deep-dive implementation notes, design decisions, failure log, and rationale for every
> significant change made during the NAVA build.

---

# Date — 11 May 2026 (Session 1) · Foundation

## Plan for the Day
- Create a clean-room reimplementation inside `nava/` without touching `nava_core/`.
- Wire real model files, implement per-user storage, stand up auth, fields, crops, chat,
  diagnose, and VNIR as a single runnable service.
- Provide a React SPA served directly from FastAPI.

---

## Work Completed (Detailed)

### 1. Project Scaffold

Created `nava/` with its own package (`nava-core`) and `pyproject.toml`.
Module root is `nava_core/` which mirrors the `nava_core/` naming convention but keeps
all nava code isolated.

```
nava/
  nava_core/
    mizhi/          ← detection + VNIR
    mozhi/          ← chat service + session store
    shared/         ← config, schemas, storage, utils
    gathi/
      api/          ← FastAPI routers, deps
      frontend/     ← Vite + React SPA
  models/           ← EfficientNet .pth + Thanal .onnx (copied from parent)
  documentation/
  run.py
```

The entrypoint `run.py` launches Uvicorn with `--reload` watching `nava_core/`.

### 2. Per-User Storage Architecture

`FieldStore` is instantiated once per user via `field_store_for_user(user)` in `deps.py`.
Each user gets their own SQLite database at `logs/users/user_{id}/user_data.db`.

Tables:
- `fields` — name, location, area, soil_type, shared_context (auto), field_notes (manual)
- `crops` — name, variety, season, stage, notes (manual crop notes)
- `plants` — name, description, linked to a crop
- `events` — event_type, field_id, crop_id, plant_id, payload (JSON), created_at
- `vnir_history` — plant_id, ratio, avg_green, avg_vnir, status

Schema migrations are handled incrementally via `_migrate_schema()` using `PRAGMA table_info`
to `ALTER TABLE` safely without losing existing data.

### 3. Auth System

`UserStore` lives at `logs/users/users.db` (shared across users).
- PBKDF2-SHA256 password hashing (100,000 iterations).
- Session tokens generated with `secrets.token_urlsafe(32)`.
- `require_user` FastAPI dependency reads token from `Authorization: Bearer` header.

Routes:
- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET  /api/auth/me`

### 4. Disease Detection Pipeline (Mizhi)

`nava_core/mizhi/detection/` wraps the EfficientNet-B0 checkpoint.
- `EfficientNetPredictor.predict(image)` → `PredictionResult` (class_label, confidence,
  reliability, class_index).
- `predict_with_cam(image)` → `(PredictionResult, cam_pil_image)` — runs Grad-CAM only
  when reliability == "RELIABLE" to avoid misleading heatmaps.
- `image_to_base64(pil_image)` returns a full data URI string (`data:image/png;base64,...`).
- Results are persisted as `diagnose` events in the DB.

Route: `POST /api/diagnose` — accepts multipart form with `image`, `plant_id`, `crop_id`,
`field_id`. Returns `DiagnoseResponse`.

### 5. VNIR Pipeline (Mizhi/VNIR)

`nava_core/mizhi/vnir/` wraps the Thanal ONNX model.
- `VNIRPipeline.process_image(pil_image, plant_name, history_ratios)` → `(VNIRStats, hsv_pil, vnir_pil)`.
- History ratios fed from `FieldStore.get_vnir_ratios(plant_id)` enable baseline, rolling
  average, and checkpoint comparisons.
- VNIR readings are also stored in `vnir_history` for the ratio trend history.
- Results persisted as `vnir` events in the DB.

Route: `POST /api/vnir-upload` — same multipart form pattern. Returns `VNIRResponse`.

### 6. Chat Service (Mozhi)

`ChatService` orchestrates:
1. **Context injection** — `_build_context_message(field_id, crop_id)` calls
   `get_rich_crop_context(crop_id)` for crop-level chats (full plant history, sibling crops,
   priority rules for LLM) or assembles minimal field metadata for field-level sessions.
2. **HF Router client** — sends `messages` list to Hugging Face Inference API.
3. **Summarization** — after every `summary_batch` messages, a summary bullet is generated
   and stored; after `summary_rollup` level-1 summaries, a level-2 rollup is created.
4. **Memory injection** — level-2 + recent level-1 summaries are prepended as system messages.

`SessionStore` at `logs/mozhi_sessions.db` holds `chat_messages`, `chat_summaries`,
`chat_state` (last summarized ID), and `session_context` (field_id / crop_id binding).

### 7. React SPA

Built with Vite. Served from FastAPI via a catch-all HTML fallback route.
Pages: `Landing`, `Auth`, `Fields`, `FieldDetail`, `CropDetail`.
Components: `Layout`, `AuthProvider`, `PlantSelector`, `ChatPanel`, `DiagnosePanel`,
`MonitorPanel`, `OverviewPanel`.
`apiFetch` wrapper in `lib/api.js` handles auth token injection and JSON parsing.

---

## Failures, Issues, and Fixes

| Issue | Cause | Fix |
|---|---|---|
| `ERR_INVALID_URL` on image display | Frontend was prepending `data:image/png;base64,` to a value that already was a full data URI | Use `src={result.gradcam_image_base64}` directly |
| Overview "No scans" despite events existing | `EventResponse` Pydantic schema was missing `plant_id`; field stripped by serializer | Added `plant_id: Optional[int] = None` to `EventResponse` |
| SQLite `OperationalError` on startup | Adding columns to existing DBs failed with `duplicate column` | Wrapped `ALTER TABLE` in `PRAGMA table_info` check inside `_migrate_schema()` |
| Chat didn't know sibling crop disease status | `get_rich_crop_context` listed sibling names only, no health data | Added per-sibling plant loop querying latest `diagnose` and `vnir` events |
| Auto-generated context visible in UI | `shared_context` held both auto and manual text | Added separate `field_notes` column; UI only reads/writes `field_notes` |
| Chat markdown rendering as `*plain*` | Bubbles used `{item.content}` plain text | Added `renderMarkdown()` JSX function; applied to assistant messages |

---

## Design Decisions

- **Separate `field_notes` from `shared_context`** — `shared_context` is regenerated
  automatically whenever crops change; `field_notes` is user-controlled. Mixing them
  would overwrite user text on every crop update.
- **`plant_id` scoping for events** — both `diagnose` and `vnir` events are keyed by
  `plant_id`, not just `crop_id`, enabling per-plant history isolation.
- **VNIR lower priority in chat context** — explicit "PRIORITY RULES" block injected into
  the system context tells NAVA to treat VNIR as precautionary and disease detection as
  the primary clinical signal.
- **No emoji in result cards** — professional status indicators use colour-coded pills,
  severity bars, and confidence gauge tracks rather than emojis.
- **Sidebar collapsible at 56 px** — collapsed state shows only icon-width nav items;
  the crop workspace fills `calc(100vh - 57px)` to avoid double scrollbars.

---

# Date — 11 May 2026 (Session 2) · UI Polish & Professional Cards

## Plan
- Redesign Diagnose and VNIR result display to 1:1:1 layout.
- Fix dot colour logic in overview.
- Ensure cross-crop context works in chat.

## Work Completed (Detailed)

### 1:1:1 Result Row Layout

CSS grid of three equal-flex columns (`flex: 1`):
- **Diagnose**: `status col` | `original image` | `Grad-CAM`
- **VNIR**: `status + all metrics col` | `HSV image` | `VNIR stress map`

On mobile (<700 px) stacks vertically via media query.

### Disease Detection Status Card (Col 1)

```
┌──────────────────────────┐
│ ████ severity bar (4px)  │
│                          │
│  ● DISEASE DETECTED      │ ← pill tag
│                          │
│  Late Blight             │ ← cleaned label
│  Tomato                  │ ← crop sub-tag
│                          │
│  Model Confidence  99%   │
│  ████████████████░░      │ ← gauge track
│                          │
│  ● Reliable              │ ← chips
│  Action needed           │
└──────────────────────────┘
```

Label cleaning: `"tomato_late_blight"` → remove prefix → `"late_blight"` → title case → `"Late Blight"`.

### VNIR Status Card (Col 1)

Tier system: `ok` / `warning` / `critical` / `calibrating` → drives border colour, tag background.
Raw measurements displayed in a 3-column mini-grid (Ratio / Avg Green / Avg VNIR).
Delta rows (vs Baseline, Global, Rolling, Checkpoint) listed below with red highlight
when `|Δ| > 5%`.

### Overview Dot Legend

| Colour | Disease Detection | VNIR Monitoring |
|---|---|---|
| Green | Healthy | OK / Healthy |
| Red | Disease detected | Warning / Stress / Critical |
| Blue | — | Calibrating |
| Grey | No scan | No scan |

Each dot uses the native `title` attribute for tooltip:
`"Disease detected · Tomato Late Blight"` or `"Calibrating · Calibrating (3/5)"`.

Dots scale up on hover (`transform: scale(1.4)`) for discoverability.
