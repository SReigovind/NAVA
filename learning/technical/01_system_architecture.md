# System Architecture

> **Subfolder:** `technical/`
> **Cross-references:** [non_technical/07_project_evolution.md](../non_technical/07_project_evolution.md) | [02_disease_detection_pipeline.md](02_disease_detection_pipeline.md) | [08_database_design.md](08_database_design.md) | [10_api_and_auth_design.md](10_api_and_auth_design.md) | [code/02_gathi_main_and_startup.md](../code/02_gathi_main_and_startup.md)

---

## The Five Modules

NAVA is structured as a monorepo containing five named modules, each with a specific responsibility:

| Module | Malayalam meaning | Responsibility |
|--------|------------------|----------------|
| **Gathi** (ഗതി) | Path, movement | API server + React frontend; the entry point for all interactions |
| **Mizhi** (മിഴി) | Eye, vision | Disease detection (EfficientNet-B0) + VNIR stress monitoring (Thanal) |
| **Mozhi** (മൊഴി) | Language, voice | LLM chat orchestration, session memory, RAG integration |
| **Yukthi** (യുക്തി) | Logic, reasoning | RAG pipeline: document ingestion, vector store, hybrid retrieval |
| **Shared** | — | Cross-cutting concerns: configuration, storage, utilities, geocoding |

These five modules are co-located in `nava_core/` under a single Python package. They are not separate services or microservices — they run in a single process. This is an intentional simplicity decision (see "Monorepo vs Microservices" below).

---

## How the Modules Communicate

Modules communicate through direct Python function and method calls, not network calls or message queues. This is possible because they run in the same process.

The communication graph:

```
Browser (React SPA)
    │
    │ HTTP REST (JSON / multipart form)
    ▼
Gathi (FastAPI)
    ├─── auth router ──────────────────► Shared (UserStore)
    ├─── fields router ────────────────► Shared (FieldStore)
    ├─── weather router ───────────────► Shared (FieldStore + geo_context)
    ├─── diagnose router ──────────────► Mizhi (EfficientNetB0Predictor + GradCamGenerator)
    ├─── vnir router ──────────────────► Mizhi (VNIRPipeline)
    └─── chat router ──────────────────► Mozhi (ChatService)
                                              ├─► Shared (FieldStore)  [farm context]
                                              └─► Yukthi (RAGRetriever, QueryRouter) [retrieval]
```

There are no circular dependencies. Yukthi does not call Mozhi. Mizhi does not call Mozhi. Shared does not call any other module (it is a dependency of all others, never the reverse).

---

## Why This Module Boundary?

The boundary was drawn around cognitive concern, not technical concern. Each module corresponds to one of the system's answerable questions:

- **Can I see what's wrong with this plant?** → Mizhi
- **What do experts say about this condition?** → Yukthi  
- **Let me talk about it.** → Mozhi
- **Where is my farm data?** → Shared (FieldStore)
- **How does the browser reach all of this?** → Gathi

This means a developer working on disease detection accuracy only touches `mizhi/`. A developer improving the RAG retrieval algorithm only touches `yukthi/`. Changes to the chat memory system only touch `mozhi/`. The blast radius of a change is limited to one module.

---

## Monorepo vs. Microservices

A natural question: why not split these into separate services that communicate over HTTP? Microservices architecture is a common choice for systems with multiple independent components.

**Arguments for microservices:**
- Independent deployability (update the VNIR model without restarting the chat service)
- Independent scaling (add more chat API replicas if chat load increases)
- Language independence (Yukthi could theoretically be implemented in Go)

**Arguments against, for NAVA specifically:**
- **Deployment complexity:** Each microservice requires its own Docker container, health check, service discovery, load balancer, and inter-service networking. For an MSc thesis project targeting deployment on a Raspberry Pi, this is an unreasonable operational burden.
- **Latency:** A chat request that assembles context from FieldStore, calls the RAG retriever, and then calls the LLM would require three inter-service HTTP calls in a microservices architecture. In a monolith, they are function calls — nanoseconds vs. milliseconds.
- **No independent scaling need:** NAVA Phase 2 serves one user (the thesis demo) to small teams. The load does not require horizontal scaling.
- **ChromaDB's threading requirement:** ChromaDB's `PersistentClient` has threading constraints that make cross-service sharing difficult. In a monolith, it is loaded once in the main thread, safely.

The monorepo architecture is the right choice for NAVA's current scale and deployment target. If NAVA were to grow to serving thousands of concurrent users, the module boundaries already established would make it straightforward to extract individual modules into services.

---

## Dependency Injection as the Integration Seam

The five modules do not directly instantiate each other. Instead, Gathi's `deps.py` acts as the integration seam: it contains dependency functions that construct and cache the heavy objects (predictors, stores, retrievers) and inject them into route handlers on demand.

This has two important consequences:
1. **Testability:** Any route handler can be tested by providing a mock dependency — the handler doesn't know or care whether it's receiving a real FieldStore or a test double.
2. **Singleton guarantees:** `@lru_cache` on dependency functions ensures that expensive objects (the ONNX session, the ChromaDB client, the SentenceTransformer) are constructed exactly once, not once per request.

The `@lru_cache` pattern is a Python idiom for process-lifetime singletons. It is not a framework feature — it is standard library. This makes the dependency system portable and transparent.

---

## The Startup Sequence and Its Constraints

The lifespan hook in `startup.py` establishes a critical constraint: **ChromaDB must be initialised in the main thread**. This is because ChromaDB's `PersistentClient` uses a Rust-based storage engine via C extensions. Creating it from a worker thread or an async event loop can produce Rust FFI panics.

The solution: ChromaDB is loaded synchronously in `_startup()`, which runs in the main thread before the server accepts any requests. The PyTorch model (EfficientNet-B0) and VNIR ONNX model are loaded in background daemon threads — they have no FFI constraints.

This startup strategy means the server is available immediately after the lifespan hook yields, even though the ML models may still be loading. The first request to `/api/diagnose` or `/api/vnir-upload` that arrives before those models finish loading will block briefly — this is extremely rare in practice.

---

## Data Flow Summary

A complete request flow for a disease detection scan:

1. Browser uploads a leaf image to `POST /api/diagnose` (multipart form)
2. Gathi's `diagnose.py` router handles the request
3. `deps.py`'s `require_user` validates the Bearer token → returns the `UserRecord`
4. `deps.py`'s `get_predictor()` returns the cached `EfficientNetB0Predictor`
5. `deps.py`'s `field_store_for_user()` returns a `FieldStore` bound to the user's DB path
6. The predictor runs `predict()` on the uploaded image → returns `PredictionResult`
7. If `RELIABLE`, `predict_with_cam()` is called → returns prediction + Grad-CAM image
8. `field_store.add_event()` writes the result to the user's database
9. `field_store._refresh_field_context()` regenerates the shared_context for the field (to keep chat context current)
10. The response JSON (including base64-encoded images) is returned to the browser
11. A background task fires `_summarize_if_needed()` (if it were a chat request, this would update memory; for diagnose, no background task is needed)

Total latency: typically 800–1500ms on CPU for a reliable prediction with Grad-CAM. The bottleneck is the PyTorch forward pass.

---

## File System Layout

```
NAVA-AG/
├── run.py                    ← Server entry point (uvicorn)
├── ingest.py                 ← CLI for RAG document ingestion
├── requirements.txt          ← Python dependencies
├── nava_core/
│   ├── gathi/
│   │   ├── api/
│   │   │   ├── main.py       ← FastAPI app creation
│   │   │   ├── startup.py    ← Lifespan hook
│   │   │   ├── deps.py       ← Dependency injection functions
│   │   │   └── routers/      ← auth, fields, weather, diagnose, vnir, chat
│   │   └── frontend/         ← React SPA (Vite build)
│   ├── mizhi/
│   │   ├── detection/        ← EfficientNetB0Predictor, GradCamGenerator
│   │   └── vnir/             ← VNIRPipeline, VNIREngine, VNIRAnalyzer
│   ├── mozhi/
│   │   ├── chat/             ← ChatClient, ChatService
│   │   └── memory/           ← SessionStore
│   ├── yukthi/               ← RAGPipeline, chunker, store, retriever, router, keywords
│   └── shared/
│       ├── config/           ← Settings (pydantic-settings)
│       ├── schemas/          ← Pydantic response models
│       ├── storage/          ← UserStore, FieldStore
│       └── utils/            ← geo_context, image, logging, paths
├── ragsource/                ← Agricultural reference documents (PDF/TXT by crop)
├── models/                   ← EfficientNet .pth checkpoint, VNIR .onnx file
├── logs/                     ← SQLite databases, ChromaDB, uploaded images
└── documentation/            ← Full module documentation
```
