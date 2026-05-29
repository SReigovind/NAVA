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

# Date — 11-21 May 2026 (Session 2 & UI Finalization) · Architecture & Professional Polish

## Plan
- Refactor the dashboard layout to a flexible "Bento Box" responsive structure.
- Redesign Diagnose and VNIR result display to a strict 1:1:1 layout.
- Fix hover alert clipping and unify tooltip components.
- Introduce inline data editing (Field properties).
- Build a secure, modern Profile settings page from scratch.

## Work Completed (Detailed)

### 1. Flexible Viewport Layouts (Bento Box Architecture)

Previously, components used hardcoded pixel heights which caused clipping on smaller screens or awkward empty space on large ones.
- `OverviewPanel.jsx` and `CropDetail.jsx` were refactored using CSS Flexbox. By applying `flex: 1` and `flexDirection: column` combined with `minHeight: 0` on nested containers, the Plant Health and Recent Activity cards now dynamically stretch to fill the available vertical viewport space.
- Implemented persistent vertical scrollbars (`custom-scrollbar`) within the bounded flex containers for handling long lists without breaking the main layout.

### 2. Global Hover Tooltip System

Hover alerts inside the `Fields.jsx` grid previously clipped under other grid items due to `overflow: hidden`.
- Migrated all hover-based concern alerts to a root-level, fixed-position system using coordinate tracking (`rect.top`, `rect.left`).
- Unified single-concern and multi-concern tooltips into a singular "carousel-style" presentation, ensuring consistent sizing and structural predictability.
- Added a `150ms` grace period timeout to prevent flickering when transitioning the mouse between the target and the tooltip.

### 3. Sidebar Navigation & Field Editing

- **Sidebar Persistence**: The "Back to Field" and "Edit Crop" buttons in the `CropDetail` sidebar were refactored. When collapsed, they no longer disappear; instead, they shift seamlessly into icon-only items placed at the absolute top/bottom of the navigation stack.
- **Inline Field Edit**: Added an edit modal directly to `FieldDetail.jsx` allowing users to update field properties (Location, Area, Soil Type) via a `PUT` request without returning to the main menu.

### 4. 1:1:1 Result Row Layout

CSS grid of three equal-flex columns (`flex: 1`):
- **Diagnose**: `status col` | `original image` | `Grad-CAM`
- **VNIR**: `status + all metrics col` | `HSV image` | `VNIR stress map`
- On mobile (<700 px) stacks vertically via media query.

### 5. Profile & Security Subsystem

Completely rebuilt the profile architecture to standard SaaS specifications.
- **Backend**: Extended `UserStore.py` with `update_user`, `update_password` (with PBKDF2 hash verification), and `delete_user` capabilities. Added corresponding FastAPI routes (`PUT /api/auth/me`, `PUT /api/auth/password`, `DELETE /api/auth/me`).
- **Frontend (`Profile.jsx`)**: Implemented a highly compact, centralized configuration layout (max-width `640px`) devoid of extraneous emojis. Features include real-time form validation (verifying new password match, blocking reuse of current password) and a secure double-confirmation state machine for permanent account deletion.

### 6. Chat Summarizer Engine Refinements

**The Plan:** 
To prevent the LLM's context window from overflowing during long, persistent crop-monitoring sessions, we planned a hierarchical, autonomous summarizer engine. The goal was to silently chunk conversation history into Level 1 (short-term) summaries every `summary_batch` (12 messages), and then roll those up into Level 2 (long-term) summaries after hitting the `summary_rollup` threshold (5 summaries).

**What We Tried & The Failures Experienced:**
1. **Context Duplication & Runaway Summaries:**
   - *Failure:* Initially, the summarizer was fed the entire chat history on every trigger. This caused the model to summarize already-summarized text, leading to recursive hallucinations and massive token bloat.
   - *Fix:* We strictly isolated the summarization logic in `service.py` to only pull the exact `batch` of new messages using `fetch_messages_with_ids(session_id, after_id=last_id, limit=self.summary_batch)`. We updated `session_store.py` to explicitly track the `last_summarized_id` in the `chat_state` table.
2. **Formatting Instability:**
   - *Failure:* The summarizer model occasionally returned conversational filler (e.g., "Here is the summary of your chat:") which consumed context space when injected back into the system prompt.
   - *Fix:* Engineered a strict prompt architecture (`_build_summary_prompt`) forcing the LLM to output "ONLY bullet points" and strictly forbidding introductory conversational text. 
3. **Ghost Context on Deletion:**
   - *Failure:* When users deleted specific chat messages or cleared an entire session, the background summaries persisted in the SQLite `chat_summaries` table. This led to "ghost context" where the LLM remembered things the user had explicitly deleted.
   - *Fix:* Enforced tight cascading wipes in `SessionStore`. Added exact `delete_summaries(summary_ids)` methods and ensured that resetting a session correctly wipes the `chat_state`'s `last_summarized_id` tracker back to 0.
4. **Token Generation Limits:**
   - *Failure:* Level 2 rollups (summaries of summaries) were occasionally truncating because they hit the default generation limits.
   - *Fix:* Exposed `summary_max_new_tokens` explicitly in the configuration, setting a dedicated hard override of `200` tokens just for the background summarizer client execution (`self.client.send(..., max_new_tokens_override=self.summary_max_new_tokens)`).

**The Success:**
The dual-level (L1 / L2) memory injection system now perfectly stabilizes long-term context. `_summary_context()` reliably injects exactly one `Level 2 Long-term summary` and up to four `Level 1 Recent summaries` directly into the system prompt without user intervention, making the chatbot deeply context-aware over months of crop lifecycle without ever crashing due to token limits.
