# Code Learning: Module Index

> **Subfolder:** `code/`
> This file is a navigation guide for the `code/` folder. Every file in this folder explains the implementation of a specific module or set of files in the NAVA codebase, with code snippets and design justifications.

---

## Reading Order

For a first read-through, follow this order — each file builds on context from the previous ones:

| File | What it Covers |
|------|---------------|
| [01_entry_points.md](01_entry_points.md) | `run.py`, `ingest.py` — how NAVA starts and how knowledge is ingested |
| [02_gathi_main_and_startup.md](02_gathi_main_and_startup.md) | `main.py`, `startup.py` — FastAPI app, lifespan, model preloading |
| [03_gathi_deps.md](03_gathi_deps.md) | `deps.py` — dependency injection, lru_cache singletons, auth guard |
| [04_gathi_routers_auth_and_fields.md](04_gathi_routers_auth_and_fields.md) | `auth.py`, `fields.py` — user registration/login, farm data CRUD, background tasks |
| [05_gathi_routers_weather_diagnose_vnir_chat.md](05_gathi_routers_weather_diagnose_vnir_chat.md) | `weather.py`, `diagnose.py`, `vnir.py`, `chat.py` — remaining API routers |
| [06_mizhi_detection.md](06_mizhi_detection.md) | `inference.py`, `gradcam.py`, `labels.py` — EfficientNet-B0, Grad-CAM, model loading |
| [07_mizhi_vnir.md](07_mizhi_vnir.md) | `pipeline.py`, `inference.py`, `analyzer.py` — HSV isolation, ONNX inference, statistics |
| [08_mozhi_client_and_service.md](08_mozhi_client_and_service.md) | `client.py`, `service.py` — LLM API client, context assembly, RAG integration, summarisation |
| [09_mozhi_session_store.md](09_mozhi_session_store.md) | `session_store.py` — chat message storage, summary L1/L2, context persistence |
| [10_yukthi_pipeline_and_store.md](10_yukthi_pipeline_and_store.md) | `pipeline.py`, `store.py`, `retriever.py`, `router.py` — ingestion, ChromaDB, hybrid retrieval, routing |
| [11_yukthi_retriever_router_keywords.md](11_yukthi_retriever_router_keywords.md) | `chunker.py`, `keywords.py` — document chunking strategies, LLM keyword extraction |
| [12_shared_storage.md](12_shared_storage.md) | `user_store.py`, `field_store.py` — user/session DB, farm data DB, migration, cascade delete |
| [13_shared_utils_and_config.md](13_shared_utils_and_config.md) | `geo_context.py`, `settings.py`, `paths.py`, `logging.py`, `schemas/` |

---

## Module Map

```
NAVA-AG/
├── run.py                        → 01_entry_points.md
├── ingest.py                     → 01_entry_points.md
│
└── nava_core/
    ├── gathi/api/
    │   ├── main.py               → 02_gathi_main_and_startup.md
    │   ├── startup.py            → 02_gathi_main_and_startup.md
    │   ├── deps.py               → 03_gathi_deps.md
    │   └── routers/
    │       ├── auth.py           → 04_gathi_routers_auth_and_fields.md
    │       ├── fields.py         → 04_gathi_routers_auth_and_fields.md
    │       ├── weather.py        → 05_gathi_routers_weather_diagnose_vnir_chat.md
    │       ├── diagnose.py       → 05_gathi_routers_weather_diagnose_vnir_chat.md
    │       ├── vnir.py           → 05_gathi_routers_weather_diagnose_vnir_chat.md
    │       └── chat.py           → 05_gathi_routers_weather_diagnose_vnir_chat.md
    │
    ├── mizhi/
    │   ├── detection/
    │   │   ├── inference.py      → 06_mizhi_detection.md
    │   │   ├── gradcam.py        → 06_mizhi_detection.md
    │   │   └── labels.py         → 06_mizhi_detection.md
    │   └── vnir/
    │       ├── pipeline.py       → 07_mizhi_vnir.md
    │       ├── inference.py      → 07_mizhi_vnir.md
    │       ├── analyzer.py       → 07_mizhi_vnir.md
    │       └── validation.py     → 07_mizhi_vnir.md
    │
    ├── mozhi/
    │   ├── chat/
    │   │   ├── client.py         → 08_mozhi_client_and_service.md
    │   │   └── service.py        → 08_mozhi_client_and_service.md
    │   └── memory/
    │       └── session_store.py  → 09_mozhi_session_store.md
    │
    ├── yukthi/
    │   ├── pipeline.py           → 10_yukthi_pipeline_and_store.md
    │   ├── store.py              → 10_yukthi_pipeline_and_store.md
    │   ├── retriever.py          → 10_yukthi_pipeline_and_store.md
    │   ├── router.py             → 10_yukthi_pipeline_and_store.md
    │   ├── chunker.py            → 11_yukthi_retriever_router_keywords.md
    │   └── keywords.py           → 11_yukthi_retriever_router_keywords.md
    │
    └── shared/
        ├── storage/
        │   ├── user_store.py     → 12_shared_storage.md
        │   └── field_store.py    → 12_shared_storage.md
        ├── config/
        │   └── settings.py       → 13_shared_utils_and_config.md
        ├── utils/
        │   ├── geo_context.py    → 13_shared_utils_and_config.md
        │   ├── paths.py          → 13_shared_utils_and_config.md
        │   ├── logging.py        → 13_shared_utils_and_config.md
        │   └── image.py          → 13_shared_utils_and_config.md
        └── schemas/
            └── ...               → 13_shared_utils_and_config.md
```

---

## Key Patterns Recurring Across the Codebase

| Pattern | Where Used | Why |
|---------|-----------|-----|
| `@lru_cache` singleton | `deps.py`, `settings.py` | Heavy objects (models, stores) constructed once, shared |
| `BackgroundTasks` | `auth.py`, `fields.py`, `chat.py` | Defer slow work (weather, model load, summarisation) after HTTP response |
| `try/except: pass` (best-effort) | weather refresh, RAG routing, auto-notes | Fail silently for non-critical background work |
| `frozen=True` dataclass | `UserRecord`, `ChatConfig`, `Settings` | Immutable value objects |
| Dynamic SQL with parameterised args | `field_store.update_field()` | Partial updates without overwriting existing values |
| Registry pattern | `chunker.CHUNKER_REGISTRY` | Add new file formats without changing orchestration code |
| `_ensure_state()` before every read | `session_store.py` | Lazy initialisation of session state rows |
| `id > last_id` sliding window | `session_store.fetch_messages()` | Exclude already-summarised messages from context |
| WAL mode | `field_store._connect()` | Allow concurrent read+write without blocking |
| Manual cascade delete | `field_store.delete_field()` | SQLite FKs disabled by default |
