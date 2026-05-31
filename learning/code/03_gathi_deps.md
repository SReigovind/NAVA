# Gathi: `deps.py` — Dependency Injection

> **Subfolder:** `code/`
> **Cross-references:** [technical/10_api_and_auth_design.md](../technical/10_api_and_auth_design.md) | [02_gathi_main_and_startup.md](02_gathi_main_and_startup.md) | [12_shared_storage.md](12_shared_storage.md)

**Source file:** [`deps.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/gathi/api/deps.py)

---

## What `deps.py` Is

`deps.py` is the dependency injection layer. It contains every function that a FastAPI route handler can declare as a `Depends()` argument. It is the place where the application's heavy objects are constructed, cached, and distributed to handlers that need them.

Route handlers never instantiate stores, predictors, or services directly. They declare a dependency and FastAPI calls the dependency function and injects the result. This keeps route handlers thin and testable.

---

## TYPE_CHECKING Guard for Heavy Imports

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nava_core.mizhi.detection import EfficientNetB0Predictor
    from nava_core.mizhi.vnir import VNIRPipeline
    from nava_core.mozhi.chat import ChatService
    from nava_core.mozhi.memory import SessionStore
    from nava_core.yukthi.retriever import RAGRetriever
    from nava_core.yukthi.store import YukthiStore
```

`TYPE_CHECKING` is `False` at runtime but `True` during static analysis (mypy, Pyright). The imports inside the `if TYPE_CHECKING:` block are only used for type annotations — they are never executed at runtime. This avoids importing PyTorch, ONNX Runtime, and ChromaDB when `deps.py` is imported, which happens at server startup before these modules are intentionally loaded.

The actual imports happen inside each function body (lazy imports). This is intentional — it defers the heavy import cost until the function is first called.

---

## The `@lru_cache` Singleton Pattern

```python
@lru_cache
def get_predictor() -> "EfficientNetB0Predictor":
    from nava_core.mizhi.detection.inference import EfficientNetB0Predictor
    s = get_settings()
    return EfficientNetB0Predictor(
        model_path=s.efficientnet_model_path,
        labels_path=s.efficientnet_labels_path,
        device=s.torch_device,
        confidence_threshold=s.confidence_threshold,
    )
```

`@lru_cache` on a zero-argument function creates a process-lifetime singleton. The first call:
1. Imports `EfficientNetB0Predictor` (triggers PyTorch import)
2. Loads the checkpoint file from `s.efficientnet_model_path`
3. Builds the model architecture and loads weights
4. Returns the initialised predictor

Every subsequent call returns the cached instance — no model loading, no file I/O.

**Why not use a global variable?** A global at module level would be constructed when `deps.py` is imported, which happens before the server is ready. If the model file doesn't exist or the environment isn't configured yet, the import would fail silently or with an uncontrolled exception. `@lru_cache` is lazy — the object is constructed on first *use*, when the settings have been loaded and the file paths are available.

**The same pattern applies to:**
- `get_vnir_pipeline()` — the `VNIRPipeline` (ONNX session + VNIRAnalyzer)
- `get_user_store()` — the `UserStore` (global `users.db`)
- `get_settings()` (in `config.py`, not `deps.py`) — the Pydantic settings object

---

## The Auth Flow: `_extract_token()` and `require_user()`

```python
def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization
```

`_extract_token()` normalises the `Authorization` header. It handles both:
- Standard format: `"Bearer abc123..."` → extracts `"abc123..."`
- Raw token (no prefix): `"abc123..."` → returns as-is

The case-insensitive `parts[0].lower() == "bearer"` check handles clients that send `"bearer"` instead of `"Bearer"`.

```python
def require_user(authorization: str | None = Header(None)) -> UserRecord:
    token = _extract_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    user = get_user_store().get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired auth token")
    return user
```

`Header(None)` tells FastAPI to look for the `Authorization` header and pass its value to the function argument. If the header is absent, `authorization=None`.

The two failure modes produce distinct 401 messages:
- `"Missing auth token"` — the header was entirely absent (useful for debugging client-side token sending)
- `"Invalid or expired auth token"` — the header was present but the token wasn't found in the DB (expired, revoked, or wrong)

Any route handler that declares `user: UserRecord = Depends(require_user)` gets this validation for free:
```python
@router.get("/api/fields")
def list_fields(user: UserRecord = Depends(require_user)):
    ...
```

---

## Per-Request Stores

```python
def field_store_for_user(user: UserRecord) -> FieldStore:
    return FieldStore(Path(user.db_path))

def session_store_for_user(user: UserRecord) -> "SessionStore":
    from nava_core.mozhi.memory import SessionStore
    return SessionStore(Path(user.db_path))
```

These are **not** cached — a new `FieldStore` and `SessionStore` are constructed for each request. This is intentional: each user has a different `db_path`, so caching is not possible without a per-user cache key.

`FieldStore(Path(user.db_path))` opens a SQLite connection to the user's database, runs migration checks if needed, and returns the store object. This is fast (milliseconds) — SQLite file connections are cheap.

Both stores are backed by the same SQLite file (`user_{hash}.db`). They operate on different table sets within that file: `FieldStore` owns `fields`, `crops`, `plants`, `events`, `vnir_history`; `SessionStore` owns `chat_messages`, `chat_summaries`, `chat_sessions`.

---

## Fetching RAG Singletons from `app.state`

```python
def get_rag_retriever(request: Request) -> "RAGRetriever | None":
    return getattr(request.app.state, "rag_retriever", None)
```

The `RAGRetriever` was constructed during startup and stored on `app.state`. Route handlers don't have direct access to `app.state` — they receive a `request: Request` parameter, which provides access via `request.app.state`.

`getattr(..., None)` is used instead of direct attribute access because `rag_retriever` might not exist on `app.state` if Yukthi failed to load at startup. Returning `None` instead of raising `AttributeError` allows the chat service to degrade gracefully (no RAG) rather than crashing.

---

## `chat_service_for_user()` — The Most Complex Dependency

```python
def chat_service_for_user(user: UserRecord, request: Request) -> "ChatService":
    from nava_core.mozhi.chat import ChatService
    s = get_settings()

    rag_retriever = get_rag_retriever(request)
    rag_router = None

    if rag_retriever is not None:
        try:
            from nava_core.yukthi.router import QueryRouter
            from nava_core.mozhi.chat.client import ChatClient
            rag_router = QueryRouter(
                client=ChatClient.from_settings(),
                model=s.hf_summary_model,
            )
        except Exception as e:
            logging.getLogger("nava.deps").warning("QueryRouter init failed: %s", e)

    return ChatService.from_settings_with_store(
        store=session_store_for_user(user),
        field_store=field_store_for_user(user),
        rag_retriever=rag_retriever,
        rag_router=rag_router,
    )
```

This function assembles the entire chat capability:
1. `rag_retriever` from `app.state` (the startup-preloaded singleton)
2. `rag_router` — a `QueryRouter` constructed fresh per request (it contains the `ChatClient`, which is lightweight — just an HTTP client configuration)
3. `session_store_for_user(user)` — the user's conversation history
4. `field_store_for_user(user)` — the user's farm data
5. `ChatService.from_settings_with_store(...)` — assembles all these into a fully configured chat service

**Why is `QueryRouter` constructed per request instead of cached?**
`QueryRouter` wraps a `ChatClient`, which wraps an HTTP client. HTTP clients can enter a broken state after connection errors. Constructing a fresh one per request ensures clean state. The construction cost is negligible.

**Graceful degradation:** If `rag_retriever` is `None` (Yukthi failed at startup), `rag_router` is also `None`. `ChatService` handles both being `None` by skipping the retrieval step. The chat still works — just without knowledge grounding.

The route handler that uses this dependency:
```python
@router.post("/api/chat")
async def chat_endpoint(
    ...,
    user: UserRecord = Depends(require_user),
    request: Request,
):
    service = chat_service_for_user(user, request)
    ...
```

In route handlers, `chat_service_for_user` is called explicitly (not as a `Depends()`) because it requires the `user` object (which itself comes from `Depends(require_user)`) and the `request` object. FastAPI doesn't support passing arguments from one `Depends()` to another without nesting, so the route handler calls it explicitly after receiving both from FastAPI.
