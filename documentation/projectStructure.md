# NAVA NAVA — Project Structure

> This document describes every folder and file in the `nava/` tree and explains
> its purpose. Update this file whenever a new module or major file is added.

---

## Root (`nava/`)

```
nava/
├── nava_core/                  ← All application source code
├── documentation/            ← This documentation folder
├── logs/                     ← Runtime data (gitignored)
│   ├── users/                ← Per-user SQLite databases
│   │   └── user_{id}/
│   │       └── user_data.db  ← Fields, crops, plants, events, VNIR history
│   └── mozhi_sessions.db     ← Chat sessions, messages, summaries (shared)
├── models/                   ← Model weights (gitignored)
│   ├── best_model.pth        ← EfficientNet-B0 for disease detection
│   └── ThanalModel.onnx      ← Thanal ONNX model for VNIR stress inference
├── NAVA-Logo.png             ← Logo served at /api/logo
├── pyproject.toml            ← Package definition (nava-core, editable)
├── run.py                    ← Uvicorn entrypoint with hot-reload
└── .env                      ← Environment variables (not committed)
```

---

## Application Core (`nava_core/`)

### `nava_core/mizhi/` — Sensing & Analysis

| Path | Purpose |
|---|---|
| `detection/inference.py` | `EfficientNetPredictor` — loads `.pth`, runs forward pass, returns `PredictionResult` |
| `detection/gradcam.py` | Grad-CAM implementation; produces heatmap PIL images |
| `detection/labels.py` | Loads class label list from text file |
| `vnir/__init__.py` | Exports `validate_plant_id` helper |
| `vnir/inference.py` | ONNX runtime wrapper for Thanal model |
| `vnir/pipeline.py` | `VNIRPipeline` — HSV isolation → ONNX inference → `VNIRStats` + output images |
| `vnir/analyzer.py` | Baseline, rolling average, checkpoint, delta calculations |

### `nava_core/mozhi/` — Language & Memory

| Path | Purpose |
|---|---|
| `chat/client.py` | `ChatClient` — HF Inference API wrapper with timeout and model configuration |
| `chat/service.py` | `ChatService` — orchestrates context injection, LLM calls, summarization, rollup |
| `memory/session_store.py` | `SessionStore` — SQLite-backed messages, summaries, state, and session context |

### `nava_core/shared/` — Cross-Cutting Concerns

| Path | Purpose |
|---|---|
| `config/settings.py` | Pydantic `Settings` — reads `.env`; model paths, HF tokens, mozhi config |
| `schemas/__init__.py` | Exports all Pydantic schema types |
| `schemas/auth.py` | Auth request/response schemas |
| `schemas/fields.py` | Field, Crop, Plant, Context, Notes schemas |
| `schemas/events.py` | `EventResponse`, `EventListResponse` (includes `plant_id`) |
| `schemas/diagnose.py` | `DiagnoseResponse` (includes optional image base64 fields) |
| `schemas/vnir.py` | `VNIRResponse` (all metrics + image base64 fields) |
| `storage/user_store.py` | `UserStore` — user registration, login, session tokens (shared DB) |
| `storage/field_store.py` | `FieldStore` — all per-user domain data; see Schema Design doc |
| `utils/image.py` | `image_to_base64(pil)` → full `data:image/png;base64,...` URI |
| `utils/paths.py` | Path helpers (user DB path computation) |

### `nava_core/gathi/` — Delivery Layer

#### `nava_core/gathi/api/`

| Path | Purpose |
|---|---|
| `main.py` | FastAPI app factory — registers routers, mounts static dist, SPA fallback, logo |
| `deps.py` | FastAPI dependency functions: `require_user`, `field_store_for_user`, `get_predictor`, `get_vnir_pipeline`, `chat_service_for_user` |
| `routers/auth.py` | `/api/auth/*` — register, login, logout, me |
| `routers/fields.py` | `/api/fields`, `/api/crops`, `/api/plants`, `/api/events`, `/api/field-context`, `/api/field-notes`, `/api/crop-context` |
| `routers/diagnose.py` | `POST /api/diagnose` |
| `routers/vnir.py` | `POST /api/vnir-upload`, `POST /api/vnir-clear` |
| `routers/chat.py` | `POST /api/chat`, `/api/chat/history`, `/api/chat/summary`, `/api/chat/clear` |

#### `nava_core/gathi/frontend/` — React SPA

```
frontend/
├── src/
│   ├── App.jsx               ← Router; defines CropLayout (noPadding) wrapper
│   ├── main.jsx              ← React DOM root mount
│   ├── styles.css            ← Single global stylesheet with CSS custom properties
│   ├── lib/
│   │   └── api.js            ← apiFetch() with auth token injection
│   ├── components/
│   │   ├── AuthProvider.jsx  ← Context + hooks for auth state
│   │   ├── Layout.jsx        ← App shell (header, theme toggle, noPadding support)
│   │   └── crop/
│   │       ├── PlantSelector.jsx   ← Plant list + create form
│   │       ├── OverviewPanel.jsx   ← Stat cards, plant health dots, activity feed
│   │       ├── ChatPanel.jsx       ← Session rail, chat bubbles, markdown renderer
│   │       ├── DiagnosePanel.jsx   ← Disease detection UI + history section
│   │       └── MonitorPanel.jsx    ← VNIR monitoring UI + history section
│   └── pages/
│       ├── Landing.jsx       ← Public landing page
│       ├── Auth.jsx          ← Login/register form
│       ├── Fields.jsx        ← Field list + create modal
│       ├── FieldDetail.jsx   ← Field info, field notes editor, crop grid
│       └── CropDetail.jsx    ← Sidebar workspace (Overview/Chat/Diagnose/Monitor)
├── index.html
├── vite.config.js            ← Proxy /api/* to :8000 during dev; outDir=dist
└── package.json
```

---

## Build & Run

```bash
# 1. Install Python package (from nava/)
pip install -e .

# 2. Build frontend
cd nava_core/gathi/frontend
npm install
npm run build

# 3. Start server (from nava/)
python run.py
# → http://localhost:8000
```

During development, `npm run dev` proxies `/api/*` to the running FastAPI server.

---

## Environment Variables (`.env`)

| Variable | Description |
|---|---|
| `MIZHI_MODEL_PATH` | Absolute path to `best_model.pth` |
| `VNIR_MODEL_PATH` | Absolute path to `ThanalModel.onnx` |
| `MIZHI_LABELS_PATH` | Absolute path to label list text file |
| `HF_API_KEY` | Hugging Face Inference API token |
| `HF_CHAT_MODEL` | Model ID for main chat (e.g. `meta-llama/...`) |
| `HF_SUMMARY_MODEL` | Model ID for summarization |
| `USERS_DB_PATH` | Path to `logs/users/users.db` |
| `SECRET_KEY` | Secret for session token signing (unused if using raw tokens) |
