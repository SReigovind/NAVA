# API and Auth Design

> **Subfolder:** `technical/`
> **Cross-references:** [01_system_architecture.md](01_system_architecture.md) | [08_database_design.md](08_database_design.md) | [code/02_gathi_main_and_startup.md](../code/02_gathi_main_and_startup.md) | [code/03_gathi_deps.md](../code/03_gathi_deps.md)

---

## FastAPI: Why Not Flask or Django?

NAVA uses FastAPI (version 0.115.x) as its web framework. The choice was deliberate:

**Async first:** FastAPI is built on Starlette and runs on ASGI (Async Server Gateway Interface). Route handlers can be `async def`, allowing non-blocking I/O without threads. When NAVA calls the Hugging Face LLM API or Open-Meteo, the event loop can handle other requests while waiting — no thread is blocked.

**Automatic schema validation:** FastAPI uses Pydantic models for request and response validation. Declare the expected request body as a Pydantic model; FastAPI automatically validates incoming JSON, returns a structured 422 error for invalid data, and serialises the response model to JSON. No manual `request.json()` parsing, no hand-written validation.

**Automatic documentation:** FastAPI generates an interactive OpenAPI schema at `/docs`. Every endpoint, its parameters, its request/response models, and its authentication requirements are documented automatically. This is valuable for development and debugging.

**Dependency injection:** FastAPI's `Depends()` system provides clean, composable dependency injection. See below for detail.

**Why not Flask?** Flask is synchronous by default. Making ML inference calls and external API calls in Flask requires explicit threading or `gevent`, adding complexity. FastAPI's async model handles this naturally.

**Why not Django?** Django carries significant complexity (ORM, admin interface, template engine, settings system) that NAVA does not need. FastAPI's smaller surface area is a better fit for an API-first service.

---

## Authentication Design

### Token-Based Auth (Not JWT)

NAVA uses simple token-based authentication: a 32-byte random hex string generated at login/register and stored in the database with an expiry timestamp.

```python
session_token = secrets.token_hex(32)  # 64-character hex string
token_expires_at = datetime.utcnow() + timedelta(hours=session_ttl_hours)
```

**Why not JWT?** JWTs are stateless — the server can verify them without a database lookup. This is beneficial for horizontal scaling (any server can verify any JWT). But NAVA is a single-server deployment. The JWT's statefulness advantage doesn't apply.

Simple tokens are:
- Easier to revoke (delete from the database)
- Easier to understand and debug
- Immune to JWT-specific vulnerabilities (algorithm confusion attacks, "none" algorithm attacks)
- Sufficient for the single-server deployment model

The token is stored in `localStorage` on the client and sent as a `Bearer {token}` header on every API request.

### The `require_user` Dependency

```python
async def require_user(
    authorization: str = Header(None),
    user_store: UserStore = Depends(get_user_store),
) -> UserRecord:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    token = authorization.removeprefix("Bearer ").strip()
    user = user_store.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user
```

Any route handler that declares `user: UserRecord = Depends(require_user)` is automatically protected. FastAPI runs the dependency before the route handler — if the token is missing or invalid, the route handler never executes, and a 401 response is returned.

This pattern centralises authentication logic in one place. Adding authentication to a new endpoint requires one line: `user: UserRecord = Depends(require_user)`.

---

## Dependency Injection: The `@lru_cache` Pattern

FastAPI's `Depends()` system calls the dependency function on every request. For expensive objects (database connections, ML models, ChromaDB clients), this would be catastrophically slow. The `@lru_cache` decorator from Python's standard library transforms a dependency function into a singleton:

```python
@lru_cache
def get_user_store() -> UserStore:
    s = get_settings()
    return UserStore(s.users_db_path)
```

`@lru_cache` caches the return value keyed on the function's arguments. Since `get_user_store()` takes no arguments, there is only one cache entry — the first call constructs the `UserStore`, and all subsequent calls return the cached instance. This is a process-lifetime singleton.

**Why `@lru_cache` instead of a global variable?**
A global variable is constructed at import time. If the settings are not yet available (environment variables not set), the global construction would fail. `@lru_cache` is lazy — the object is constructed on first access, after the settings have been loaded. It also integrates cleanly with FastAPI's `Depends()` system.

**Key cached singletons:**
- `get_settings()` — the Pydantic Settings object (environment config)
- `get_user_store()` — the `UserStore` (global users DB)
- `get_predictor()` — the `EfficientNetB0Predictor` (heavy PyTorch model)
- `get_vnir_pipeline()` — the `VNIRPipeline` (ONNX session + analyzer)

### Per-Request Dependencies

Not all dependencies are singletons. The `FieldStore` is per-user — each user has a different DB path. It is constructed per-request:

```python
def field_store_for_user(user: UserRecord = Depends(require_user)) -> FieldStore:
    return FieldStore(Path(user.db_path))
```

A new `FieldStore` instance is created for each request. The `FieldStore` constructor opens a connection to the user's DB file, runs migration checks, and returns the store. This is fast (milliseconds) because SQLite file connections are cheap.

---

## The Lifespan Hook

FastAPI supports lifespan context managers for startup and shutdown logic:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup(app)  # runs before first request
    yield          # server is running
    # teardown here (not needed for NAVA)
```

The lifespan hook runs once, in the main thread, before the ASGI server accepts any requests. This is where NAVA:
1. Loads ChromaDB synchronously (thread-safety requirement — see [01_system_architecture.md](01_system_architecture.md))
2. Starts background threads to load PyTorch/ONNX models
3. Stores the loaded singletons on `app.state`

`app.state` is a per-application attribute bag. Objects stored here are accessible from any request via `request.app.state`.

---

## SPA Serving

NAVA serves the React SPA from the same FastAPI process. This eliminates the need for a separate web server (Nginx, Caddy) in development and simple deployments.

The pattern:
```python
# Serve built JS/CSS assets directly
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

# Catch-all: everything else returns index.html
@app.get("/{path:path}", response_class=HTMLResponse)
def spa_fallback(path: str) -> HTMLResponse:
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
```

React's client-side router handles all URL navigation within the SPA. The server only needs to return `index.html` for any path not matching an API route or static asset. React then reads the URL from `window.location` and renders the appropriate page.

**The SPA fallback must be registered last.** If it were registered before the API routes, every request (including API calls) would return `index.html`. FastAPI evaluates routes in registration order; specific paths match before the catch-all `/{path:path}`.

---

## CORS Configuration

During development, the React Vite dev server runs on port 5173 while the FastAPI server runs on port 8000. Cross-origin requests (from port 5173 to port 8000) are blocked by browser CORS policy unless the server explicitly allows them.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

In production (FastAPI serving the built SPA from the same origin), CORS is not needed — same-origin requests don't require it. The CORS configuration is harmless in production and essential for development.

**Why `allow_credentials=True`?** The frontend sends the `Authorization` header on API requests. Some browsers treat requests with custom headers as "credentialed" and require the server to explicitly allow credentials.

---

## API Security Model

NAVA's security posture for a single-server deployment:

- **Authentication:** Token-based, 168-hour expiry (1 week), stored in localStorage
- **Authorization:** Enforced at the dependency level — every protected route requires `require_user`
- **Data isolation:** Per-user DB files ensure one user's data is never accessible via another user's token
- **Password storage:** bcrypt with a random salt — even if the users.db is compromised, raw passwords are not exposed
- **Input validation:** FastAPI + Pydantic models validate all input structure; SQL injection is prevented by parameterised queries throughout

**What NAVA does not provide (appropriate for current scale):**
- HTTPS termination (should be added via Nginx or Caddy in production)
- Rate limiting (documented in futureWork.md)
- Audit logging
- IP-based access control
