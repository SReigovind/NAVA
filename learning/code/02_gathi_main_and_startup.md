# Gathi: `main.py` and `startup.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/01_system_architecture.md](../technical/01_system_architecture.md) | [technical/10_api_and_auth_design.md](../technical/10_api_and_auth_design.md) | [03_gathi_deps.md](03_gathi_deps.md) | [01_entry_points.md](01_entry_points.md)

**Source files:**
- [`main.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/main.py)
- [`startup.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/startup.py)

---

## `main.py` — The FastAPI Application Object

`main.py` is the composition root of the entire server. It creates the `FastAPI` application, configures middleware, registers all routers, and defines the three non-API endpoints (health, logo, SPA fallback).

### Path Setup

```python
_nava_dir = str(Path(__file__).resolve().parents[3])
if _nava_dir not in sys.path:
    sys.path.insert(0, _nava_dir)
```

`__file__` is `nava_core/gathi/api/main.py`. `.parents[3]` climbs 3 levels up: `api/` → `gathi/` → `nava_core/` → project root. This is the same sys.path manipulation as in `run.py`, but repeated here so that `main.py` is importable from any working directory. Without it, importing from pytest or a Jupyter notebook would fail.

### Application Creation

```python
app = FastAPI(title="NAVA API", version="0.2.0", lifespan=lifespan)
```

The `lifespan` parameter accepts an async context manager that FastAPI calls before accepting requests (startup) and after shutting down. The `lifespan` function is imported from `startup.py`.

### Router Registration

```python
app.include_router(auth.router)
app.include_router(diagnose.router)
app.include_router(vnir.router)
app.include_router(chat.router)
app.include_router(fields.router)
app.include_router(weather.router)
```

Each router module contains an `APIRouter` instance with its routes. `include_router()` merges those routes into the main app. The order matters for route precedence — more specific routes should be registered before more general ones.

### The Three Built-in Endpoints

**`GET /api/health`**
```python
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```
A simple liveness probe. Returns `{"status": "ok"}`. No authentication required. Used by monitoring tools and to verify the server is responsive after deployment.

**`GET /api/logo`**
```python
@app.get("/api/logo")
def logo() -> FileResponse:
    logo_path = PROJECT_DIR / "NAVA-Logo.png"
    if logo_path.exists():
        return FileResponse(logo_path, media_type="image/png")
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")
```
Serves the NAVA logo from the project root. The frontend uses this to display the logo in the navbar and landing page — keeping the logo as a server asset rather than a bundled frontend asset means it can be changed without rebuilding the frontend.

The fallback to `index.html` is a defensive measure: if the logo file doesn't exist (e.g., first-time setup), the endpoint doesn't crash.

**`GET /assets/{file_path:path}` — Static Asset Serving**
```python
@app.get("/assets/{file_path:path}")
async def serve_assets(file_path: str) -> FileResponse:
    asset_path = FRONTEND_DIR / "assets" / file_path
    if asset_path.exists() and asset_path.is_file():
        suffix = asset_path.suffix.lower()
        media_types = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
        }
        return FileResponse(asset_path, media_type=media_types.get(suffix, "application/octet-stream"))
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")
```

Vite's build output places all bundled JS/CSS files in `frontend/dist/assets/`. This endpoint serves those files with the correct MIME types. The explicit MIME type dictionary is necessary because `FileResponse` without a `media_type` argument sends `application/octet-stream` for all files — which causes browsers to download JS and CSS files instead of executing them.

**Why not use `StaticFiles` middleware?** FastAPI's `StaticFiles` middleware mounts a directory as a static file server. This works but conflicts with the `/{path:path}` catch-all — the ordering of middleware and routes in FastAPI requires careful management. The explicit `@app.get("/assets/{file_path:path}")` route is simpler and more debuggable.

**`GET /{path:path}` — SPA Fallback**
```python
@app.get("/{path:path}", response_class=HTMLResponse)
def spa_fallback(path: str) -> HTMLResponse:
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<html><body><h1>NAVA</h1><p>Run <code>npm run build</code> ..."
    )
```

This catch-all route must be registered **last** in the file. FastAPI evaluates routes in registration order — if this route were registered first, it would match every URL including the API routes. By registering it last, all API routes take precedence.

When a user navigates directly to `/fields/3` or refreshes the page at `/fields/3/crops/7`, the browser requests that URL from the server. Without the SPA fallback, FastAPI would return a 404 (it has no route for `/fields/3`). The fallback returns `index.html`, and React's client-side router handles the URL.

The fallback to the inline "run npm run build" HTML serves as a developer-friendly message when the frontend hasn't been built yet.

---

## `startup.py` — Lifespan and Model Preloading

`startup.py` manages the startup sequence: preloading heavy singletons before the server accepts requests.

### The Lifespan Pattern

```python
@asynccontextmanager
async def lifespan(app: "FastAPI"):
    _startup(app)
    yield
    # No explicit shutdown needed
```

FastAPI's lifespan is an async context manager. Code before `yield` runs at startup (before any requests); code after `yield` runs at shutdown. NAVA has no shutdown cleanup (ChromaDB flushes its WAL automatically; SQLite connections are closed by garbage collection).

### `_startup(app)` — The Initialisation Sequence

```python
def _startup(app: "FastAPI") -> None:
    log.info("=== NAVA startup: preloading models and vector store ===")
    
    _load_yukthi()   # Synchronous — ChromaDB Rust FFI MUST run in main thread
    
    t1 = threading.Thread(target=_load_predictor, daemon=True, name="startup-efficientnet")
    t2 = threading.Thread(target=_load_vnir, daemon=True, name="startup-vnir")
    t1.start()
    t2.start()
    
    log.info("=== NAVA startup complete (ML models loading in background) ===")
```

**Critical design constraint — ChromaDB in the main thread:**
ChromaDB's `PersistentClient` uses a Rust extension (via Chroma's `chromadb-rust` binding). Creating a `PersistentClient` from a non-main thread, or from an async event loop's worker thread, triggers Rust FFI panics. The solution: `_load_yukthi()` is called synchronously, directly in `_startup()`, which is called from the lifespan coroutine — which executes in the main thread.

**EfficientNet and VNIR loading in background threads:**
PyTorch and ONNX Runtime have no such threading constraint. Loading them in daemon background threads allows the server to become available immediately after `_startup()` returns, without waiting for the ~3–5 second model load time. The first request to `/api/diagnose` or `/api/vnir-upload` that arrives before the models finish loading will block briefly, but this is extremely rare in practice.

**Daemon threads:** `daemon=True` means these threads are killed automatically when the main process exits. Without this flag, they would keep the process alive even after a Ctrl+C interrupt.

### `_load_yukthi()` in Detail

```python
def _load_yukthi():
    s = get_settings()
    if not s.yukthi_enabled:
        log.info("Startup: Yukthi RAG is disabled.")
        return

    store = YukthiStore(s.yukthi_chroma_dir)
    retriever = RAGRetriever(
        store=store,
        embed_model=s.yukthi_embed_model,
        top_k=s.yukthi_top_k,
        distance_threshold=s.yukthi_distance_threshold,
    )
    retriever.warm_up()

    app.state.yukthi_store = store
    app.state.rag_retriever = retriever
```

`retriever.warm_up()` runs a dummy encode on an empty string to force the SentenceTransformer model to load. Without this, the first real RAG query would bear the full embedding model load latency (~2 seconds). After warm-up, subsequent encodes are fast (milliseconds).

The `store` and `retriever` objects are stored on `app.state` — FastAPI's per-application state bag. They are retrieved in `chat_service_for_user()` (in `deps.py`) via `request.app.state.rag_retriever`.

**Graceful failure:** If ChromaDB or the embedding model fails to load (missing ChromaDB directory, corrupted index, missing model), the exception is caught, a warning is logged, and `app.state.rag_retriever` is set to `None`. The server continues to start. Chat requests with `rag_retriever=None` will skip retrieval and warn in the logs — the chat interface degrades gracefully rather than crashing entirely.
