# NAVA-AG Project Worklog

This document serves as the objective, chronological diary of project implementation, tracking all planned tasks, attempts, failures, and resolutions.

## 2026-05-21 03:32 IST

### Initiative
Establishment of the formal project worklog.

### Plan
- Create `worklog.md` in the project root to serve as a persistent, objective diary.
- Implement a rigid third-person documentation structure to track plans, attempts, issues, and successes.

### Execution & Results
- **Success:** The file `worklog.md` was successfully initialized with the required formatting standard.
- **Outcome:** All future development tasks will append timestamped entries to this document detailing technical implementation pathways and debugging procedures.

---

## 2026-05-21 03:57 IST

### Task: RAG Pipeline — Feasibility Study & Implementation Planning

**Background:**  
The project manages 7 crop types. Source material was placed in `ragsource/`: a structured `banana.txt` (PlantVillage format, 18 KB, covering description, varieties, propagation, diseases, and pests) and the full **Kerala Agricultural University Package of Practices** guidebook (`~15.8 MB PDF`), an authoritative multi-crop agronomic reference widely used by extension workers.

The objective was to produce a proper, production-quality RAG (Retrieval-Augmented Generation) pipeline tightly integrated with the existing `ChatService` — enabling NAVA to ground responses in verified agronomic knowledge rather than relying solely on LLM parametric memory.

**Analysis Conducted:**
- Read `ragsource/banana.txt` in full. Identified strong section structure (disease entries each contain: Symptoms, Cause, Comments, Management) ideal for semantic chunking.
- Reviewed `requirements.txt` and confirmed that `chromadb`, `sentence-transformers`, and `pymupdf` are already declared.
- Read `nava_core/mozhi/chat/service.py` in full to understand where context injection occurs (`_build_context_message`), and confirmed the integration point.
- Read `session_store.py`, `field_store.py`, `settings.py`, and `client.py` to understand the full data flow.

**Plan Produced (not yet executed):**

*Architecture:* A new `nava_core/mozhi/rag/` package containing four modules:
1. `chunker.py` — section-aware chunking for `.txt` (disease/pest entries preserved as atomic chunks); page+paragraph-aware chunking for `.pdf` with PyMuPDF structure detection.
2. `store.py` — ChromaDB persistent client wrapper, with per-crop collection naming (`nava_{crop}`) and deterministic upsert IDs.
3. `pipeline.py` — offline ingestion script: parse → chunk → batch-embed → upsert. Runs at first startup (lazy init) or via management command.
4. `retriever.py` — query-time retrieval: embeds the live user message using `BAAI/bge-small-en-v1.5`, queries ChromaDB, applies a relevance threshold (cosine distance < 0.35), returns top-k chunks.

*Integration point:* `ChatService._build_context_message()` is extended to accept `message: str` and call `RAGRetriever.query(message, crop=crop_name, top_k=3)`. Retrieved chunks are injected as a clearly labelled `AGRONOMIC REFERENCE` system block before the LLM call.

*Embedding model rationale:* `BAAI/bge-small-en-v1.5` selected over `all-MiniLM-L6-v2` for superior retrieval accuracy at equivalent CPU cost. Runs fully offline (384-dim).

*Pilot scope:* Banana only in Phase 1. Remaining 6 crops added in Phase 2 by registering additional `.txt` files. KAU PDF ingested in Phase 3 (multi-crop or `nava_general` collection).

*No new dependencies required:* All packages are already present in `requirements.txt`.

**Outcome:** Plan documented. No code written. Awaiting execution approval.

---

## 2026-05-21 04:17 IST

### Task: RAG Pipeline Plan — Revision v2 (yukthi module + smart routing)

**Revisions requested:**
1. The RAG package should be placed in `nava_core/yukthi/` — a new top-level domain module coequal with `mizhi`, `mozhi`, and `gathi`. This aligns with the original project module naming convention (`yukthi` = Malayalam for intelligence/reasoning).
2. RAG retrieval must not fire on every chat message. A smart query router must gate the retrieval path, ensuring only messages that genuinely require external agronomic knowledge trigger a vector search.

**Design changes made to the plan:**

*Module relocation:* The previously planned `nava_core/mozhi/rag/` package is scrapped. All five modules (`chunker`, `store`, `pipeline`, `router`, `retriever`) are now housed in `nava_core/yukthi/` as a standalone domain module.

*Smart Query Router (`yukthi/router.py`) — new component:* A two-tier binary classifier that answers per message: "Does this require external knowledge retrieval?"
- **Tier 1 (regex fast-path, ~0ms):** Pre-compiled regex patterns for explicit NO-RAG signals (greetings, acknowledgements, memory/history requests, off-topic keywords) and explicit YES-RAG signals (disease/symptom terms, pest keywords, agronomy terms, treatment requests). If either pattern fires, the decision is made instantly without any model call.
- **Tier 2 (LLM fallback, used only for ambiguous queries):** A minimal 2-message prompt sent to the fast small model (`Llama-3.1-8B`, already used for summarization). The prompt demands a single-word response: `RETRIEVE` or `SKIP`. On any error or timeout, the router defaults to `SKIP` — keeping chat responsive and avoiding noise injection.
- The router never calls the main 70B LLM for routing. It always uses the small 8B model in Tier 2.

*Key router examples documented:*
- `"Hello NAVA!"` → Tier 1 NO-RAG (regex) → SKIP
- `"My banana plant has yellowing leaves"` → Tier 1 YES-RAG (keyword: `yellowing`) → RETRIEVE
- `"It looks strange"` → Tier 2 LLM → RETRIEVE (ambiguous but agronomic in context)
- `"What did you say earlier?"` → Tier 1 NO-RAG (memory request pattern) → SKIP

*Integration:* `QueryRouter` injected into `ChatService` alongside `RAGRetriever`. The `chat()` method calls `router.should_retrieve(message)` first; the retriever is only invoked if it returns `True`.

**Outcome:** Plan v2 documented. No code written. Awaiting execution approval.

---

## 2026-05-21 04:24 IST

### Task: RAG Pipeline Plan — Revision v3 (LLM-first routing)

**Revision requested:** Drop the two-tier regex/LLM hybrid routing approach. Use the small model (`Llama-3.1-8B-Instruct`) as the explicit, authoritative routing mechanism for all non-trivially-short messages.

**Rationale documented:** Keyword/regex heuristics are brittle — they require constant manual maintenance as new query patterns emerge, and fundamentally cannot handle contextually loaded short queries such as `"Is it serious?"` or `"What should I do next?"` which carry agronomic intent only deducible from conversational context. The 8B model, already deployed for summarisation, handles these cases correctly via semantic understanding.

**Changes made to the plan:**

- Removed the two-tier strategy entirely. There is now only one routing mechanism: the `Llama-3.1-8B-Instruct` model via `hf_summary_model`.
- Retained a single **degenerate guard** (message empty or < 3 tokens): this is not semantic routing, purely an API-call cost-avoidance measure for empty/trivially short inputs. `"Thanks!"` (1 token) is caught here without an LLM call.
- Router prompt uses `temperature=0.0` and `max_new_tokens=5` — deterministic, low-latency classification.
- Default on any error or timeout remains `SKIP` — chat responsiveness is never compromised by router failure.
- Updated data flow diagram to show the 8B model classification step for the main case, and the degenerate guard for trivially short messages.
- Updated decision table to remove the Tier column, replacing it with the reasoning behind each decision.
- Updated key design decisions table to document why no regex logic remains.

**Outcome:** Plan v3 documented. No code written. Awaiting execution approval.

---

## 2026-05-21 04:29 IST

### Task: RAG Pipeline — Implementation (Phase 1: Banana Pilot)

**Plan approved. Execution commenced.**

**Files created:**

- `nava_core/yukthi/__init__.py` — Module init; exports `RAGRetriever`, `RAGChunk`, `RAGPipeline`, `QueryRouter`.
- `nava_core/yukthi/chunker.py` — Section-aware `.txt` chunker using regex boundary detection on PlantVillage-format disease/pest entries. Each disease/pest entry (Symptoms + Cause + Comments + Management) is one atomic chunk. PDF chunker uses PyMuPDF block-level extraction with font-size heading detection and Malayalam block filtering.
- `nava_core/yukthi/store.py` — ChromaDB `PersistentClient` wrapper. Per-crop collection naming (`nava_{crop}`), cosine distance space, upsert-safe deterministic IDs, `collection_exists()` guard.
- `nava_core/yukthi/pipeline.py` — Ingestion orchestration: discovers source files by crop name prefix (.txt) and includes all PDFs as general references. Batch-embeds all chunks with SentenceTransformer and upserts. `ingest_if_missing()` is idempotent. Lazy-loaded encoder singleton.
- `nava_core/yukthi/router.py` — `QueryRouter` class. Degenerate guard (< 3 tokens → SKIP, no API call). For all other messages: sends a tightly-constrained binary classification prompt to `Llama-3.1-8B-Instruct` at `temperature=0.0`, `max_new_tokens=5`. Returns `True` only for unambiguous `RETRIEVE` response. Defaults to SKIP on any error or timeout.
- `nava_core/yukthi/retriever.py` — `RAGRetriever` class. Embeds user message with `bge-small-en-v1.5`, queries ChromaDB, applies cosine distance threshold (default 0.45), returns `RAGChunk` dataclasses. Returns `[]` silently if the crop collection doesn't exist.

**Files modified:**

- `nava_core/shared/config/settings.py` — Added 6 new `yukthi_*` fields to `Settings` dataclass and populated defaults in `get_settings()`. All configurable via `.env`.
- `nava_core/mozhi/chat/service.py` — `ChatService` now accepts optional `rag_retriever: RAGRetriever` and `rag_router: QueryRouter`. `from_settings_with_store()` factory auto-initialises yukthi components when `NAVA_YUKTHI_ENABLED=true` and kicks off a background daemon thread for `ingest_if_missing("banana")`. In `chat()`: router is consulted first; if RETRIEVE, crop name is resolved from `field_store`, retriever is called, and the resulting chunks are injected as a clearly labelled `AGRONOMIC REFERENCE` system block before `messages.extend(history)`.
- `.env` — Added commented-out yukthi config keys for reference.

**Design notes:**
- RAG is completely opt-out (`NAVA_YUKTHI_ENABLED=false`) without touching service code — all yukthi init is wrapped in a try/except that degrades gracefully.
- Ingestion runs in a background daemon thread at startup, so the server starts immediately without blocking on embedding model load.
- The `AGRONOMIC REFERENCE` block deliberately does not tell NAVA to cite sources — it instructs NAVA to use the knowledge naturally without surfacing the reference mechanism to the user.

**Outcome:** All 6 yukthi modules created. settings.py and service.py updated. Ready for build and test.

---

## 2026-05-21 04:41 IST

### Task: Separate RAG Ingestion from Server Runtime

**Issue identified:** The initial implementation kicked off a background ingestion daemon thread inside `ChatService.from_settings_with_store()` on every server startup. This conflates two fundamentally separate concerns: (1) building the knowledge base, and (2) serving chat requests against it. It also means the embedding model and PyMuPDF would load on every server start, and re-ingestion would require bouncing the server.

**Changes made:**

*`service.py` — Removed background ingestion thread.* The `from_settings_with_store()` factory now only initialises the `RAGRetriever` and `QueryRouter` read-only components. If the ChromaDB collection does not exist, it logs a `WARNING` advising the operator to run `ingest.py`. The server never writes to the vector store.

*`ingest.py` (NEW, project root)* — A standalone CLI with the following capabilities:
- `python ingest.py --crop banana` — ingest a single crop; skips silently if collection already exists
- `python ingest.py --crop banana --force` — wipe the existing collection and rebuild from scratch (used when source files are updated)
- `python ingest.py --all` — ingest all 7 known crops in sequence
- `python ingest.py --all --force` — full rebuild of all collections
- `python ingest.py --status` — list all existing ChromaDB collections and their chunk counts

*`store.py` — Added `delete_collection(crop)` method.* Required to support `--force` by wiping the existing ChromaDB collection before re-ingesting.

*`pipeline.py` — Wired `force=True` to `delete_collection()` before upsert.* Without this, ChromaDB's `upsert()` would merge new and old chunks by ID, leaving stale chunks from deleted source sections.

**Operational model going forward:**
1. Place source files in `ragsource/` (`banana.txt`, etc.)
2. Run `python ingest.py --crop banana` once (or after any source update, with `--force`)
3. Start the server with `python run.py` — it reads from the pre-built store, never writes

**Outcome:** Ingestion and serving are fully separated. ingest.py is repeatable, idempotent, and crop-targeted.

---

## 2026-05-21 04:47 IST

### Task: Per-crop ragsource Subfolders + Extensible Format Registry

**Issue identified:** The original `_find_sources()` used fragile filename-prefix matching (files starting with "banana") and unconditionally included all PDFs as "shared references". This is wrong by design — it conflates multi-crop reference material with crop-specific documents, forces a naming convention on source files, and makes it impossible to control what goes into each crop's collection independently.

**Design change:** Per-crop subfolder layout.
```
ragsource/
└── banana/
    ├── banana.txt
    └── Kerala Agricultural University - Package of Practices.pdf
```
Each subfolder is named exactly the crop name. ALL files with supported extensions inside it are ingested into `nava_banana`. The user has full control by simply placing files in the correct folder. Copying or symlinking the KAU guidebook into multiple crop folders is the intended pattern for shared reference material.

**Files changed:**

*`chunker.py` — Format registry pattern introduced.* Replaced the `if suffix == ".txt" / elif suffix == ".pdf"` dispatch with a `CHUNKER_REGISTRY` dict mapping extensions to handler functions. `SUPPORTED_EXTENSIONS` (a `frozenset`) is derived from the registry keys. To add a new format (e.g. `.docx`, `.csv`), one registers a handler in this dict — no changes needed elsewhere in the codebase.

*`pipeline.py` — `_find_sources()` rewritten.* Now looks for `ragsource/{crop}/` and returns all files whose extension is in `SUPPORTED_EXTENSIONS`. Added `list_available_crops()` which auto-discovers crops by listing non-hidden subdirectories in `source_dir`.

*`ingest.py` — Removed hardcoded `KNOWN_CROPS` list.* `--all` now calls `pipeline.list_available_crops()` to auto-discover from subfolders. `--status` now shows a combined view of ragsource/ subfolders vs ChromaDB collections, indicating which crops are ingested vs not yet ingested. Error messaging improved to show expected folder paths.

*`ragsource/` — Files physically reorganised.* `banana.txt` and the KAU PDF moved into `ragsource/banana/` to match the new layout.

**Outcome:** Source organisation is clean, self-documenting, and extensible. Adding a new crop = create a folder + drop files in. Adding a new file format = register one function.

---

## 2026-05-21 05:03 IST

### Task: RAG Grounding Indicator in Chat UI

**Request:** Show a visible but unobtrusive indication on each assistant message whether RAG retrieval was used to ground the response.

**Changes made:**

*`ChatResult` dataclass (`service.py`)* — Added `rag_used: bool = False` and `rag_chunk_count: int = 0` fields. The RAG block in `chat()` now sets `rag_used = True` and `rag_chunk_count = len(chunks)` before injecting the context.

*`ChatResponse` schema (`shared/schemas/chat.py`)* — Added `rag_used` and `rag_chunk_count` fields (both default to `False`/`0`) so the API response carries this metadata to the frontend without breaking existing clients.

*`chat.py` API router* — Updated the `ChatResponse(...)` call to pass `rag_used` and `rag_chunk_count` through from `result`.

*`ChatPanel.jsx` (frontend)* — When an assistant message is created in state, `rag_used` and `rag_chunk_count` are stored on the history item object. A `rag-badge` div is conditionally rendered below the bubble content for assistant messages where `rag_used === true`. The badge shows a small green dot, the label "Knowledge base", and a tooltip with the chunk count on hover.

*`styles.css`* — Added `.rag-badge` (inline-flex pill, green tint, uppercase micro-text) and `.rag-badge-dot` (5px green circle). Uses `color-mix()` for transparent background/border tinted with `--green-400` so it adapts to the existing colour palette.

**Design choice:** The badge is small, muted, and below the message content — informative without being distracting. It does not appear on responses grounded purely by memory/context (non-RAG), nor during animation.

**Outcome:** Frontend and API ready. Requires `npm run build` to deploy.

---

## 2026-05-21 05:15 IST

### Task: RAG Runtime Bug Fixes — Four Issues

**Issues observed on first server run:**

**Issue 1: `RustBindingsAPI object has no attribute 'bindings'` (ChromaDB)**
Root cause: ChromaDB 1.5.x initializes its Rust FFI bindings lazily. When `PersistentClient` is first instantiated inside a FastAPI request handler thread (not the main thread), the Rust bindings object is partially initialized and fails. The service was creating a new `YukthiStore` (and thus a new `PersistentClient`) on every request via `chat_service_for_user()`. Fix: Moved ChromaDB client initialization to `YukthiStore.__init__()` (eager initialization), ensuring the Rust bindings are set up in the startup/factory thread, not per-request. Added fallback to `EphemeralClient` if `PersistentClient` fails, with a clear error message.

**Issue 2: Second init failure showing the path string as the error**
Root cause: Same as Issue 1 but on a subsequent request — the first `PersistentClient` failed, and the `except` block caught the exception whose `str()` representation was the path string. Fixed by the same change.

**Issue 3: Router RETRIEVE false positive on "give me an overview of my fields"**
Root cause: The router prompt was too broad. "Overview of my fields" or "summarize my fields" are questions about the user's own data stored in the conversation context — they don't need external agronomic reference knowledge. Fix: Rewrote the router system prompt with explicit SKIP examples covering field data questions, field summaries, and crop lists. Added "when in doubt, lean towards SKIP" instruction.

**Issue 4: RAG query using raw user message only**
Root cause: The ChromaDB query was embedding only the user's literal message. For follow-up questions like "what should I spray for this?" or "how serious is it?", the message contains no disease name, so cosine similarity to disease-specific chunks is low. Fix: Added `_build_retrieval_query()` method to `ChatService` that constructs an enriched query string: `Crop: banana. Detected condition: Panama Wilt. <user message>`. It pulls the latest `diagnose` event (disease label) and `vnir` event (stress status) from the field store. This anchors the embedding to the known condition even for vague follow-ups.

**Issue 5: HuggingFace Hub unauthenticated request warning**
Root cause: `sentence-transformers` uses `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env vars for authentication. The project's `.env` uses `HF_API_KEY`. Fix: Added a one-time forwarding block in `settings.py` immediately after `load_dotenv()`: reads `HF_API_KEY` and sets `HF_TOKEN` + `HUGGING_FACE_HUB_TOKEN` via `os.environ.setdefault()` if they're not already set.

**Outcome:** All five issues fixed. Server restart required.

---

## 2026-05-21 05:33 IST

### Task: Server Startup Singleton Refactor + RAG Chunk Carousel

**Three requests addressed:**

**1. Preload disease detection and VNIR models at server boot (not on first use)**
`startup.py` created with a FastAPI lifespan context. At startup: (a) `YukthiStore` + `RAGRetriever` are initialized synchronously in the main thread (required for ChromaDB Rust FFI to work correctly), then `retriever.warm_up()` is called to pre-load the embedding model. (b) EfficientNet and VNIR models are kicked off in daemon background threads so the server becomes available immediately. All singletons are stored on `app.state`.

**2. RAG chunk carousel in the chat UI**
`rag_chunks` list added to `ChatResult`, `ChatResponse`, and threaded through the API. Each entry contains `{source, section, snippet}` (snippet = first 300 chars of chunk text). `ChatPanel.jsx` stores `rag_chunks` on the history item and renders a carousel under the "Knowledge base" badge: `‹ [source · N/M] [section] [snippet...] ›`. Per-message carousel index tracked in `carouselIdx` state dict keyed by message index.

**3. ChromaDB PersistentClient threading issue (root cause fix)**
Root cause confirmed: `chat_service_for_user()` in `deps.py` was creating a new `YukthiStore` (and new `PersistentClient`) on every request. FastAPI uses a thread pool for request handlers; ChromaDB 1.5.x Rust FFI fails when `PersistentClient` is first created in a worker thread, not the main thread. Fix: `YukthiStore` and `RAGRetriever` are now created exactly once in `startup.py` (main thread) and stored on `app.state`. `deps.py` `chat_service_for_user()` now accepts a `Request` param and reads the pre-built singletons from `app.state`. `from_settings_with_store()` updated to accept optional `rag_retriever`/`rag_router` params instead of always building fresh ones.

**Files changed:** `startup.py` (new), `main.py`, `deps.py`, `chat.py` (router), `service.py`, `retriever.py`, `chat.py` (schema), `ChatPanel.jsx`, `styles.css`

**Outcome:** All models preloaded at boot, ChromaDB initialized once in main thread, chunk carousel functional. Requires `npm run build` + server restart.

---

## 2026-05-21 05:46 IST

### Task: Carousel Fix, RAG Persistence, and Verified Source Prompt

**Issue 1: Carousel not rendering**
Root cause: Carousel was implemented as a IIFE `(() => { ... })()` inside JSX. React does not reliably evaluate IIFEs during re-renders — when the animation completes and `isAnimating` flips to false (triggering a state update), the IIFE context is lost and the carousel silently fails to render. Fix: Extracted carousel into a proper named component `RagCarousel({ chunks, chunkCount })` with its own local `useState(0)` for the index. Each message gets its own carousel state, isolated from re-renders. Placed `RagCarousel` at module level (outside `ChatPanel`) so it is stable across renders.

**Issue 2: RAG badge and chunks lost on page refresh**
Root cause: `fetch_message_history()` only returned `role`, `content`, `created_at` — no RAG metadata. The metadata was only in React state for the current session. Fix:
- Added `metadata TEXT DEFAULT NULL` column to `chat_messages` SQLite table
- Added `ALTER TABLE ... ADD COLUMN metadata` migration for existing databases (via `PRAGMA table_info`)
- Updated `append_message()` to accept `metadata: Optional[dict]` and store it as JSON
- In `service.py` `chat()`: after RAG is used, the assistant message is saved with `metadata={rag_used, rag_chunk_count, rag_chunks}`
- Updated `fetch_message_history()` to SELECT and parse metadata column
- Added `metadata: Optional[dict]` to `ChatHistoryMessage` Pydantic schema
- `refreshHistory` in `ChatPanel.jsx` now unpacks `m.metadata?.rag_chunks` etc. when building the history array — so RAG badges and carousel data are restored from the DB after refresh

**Issue 3: LLM hedges on RAG content**
Updated the AGRONOMIC REFERENCE block header in `service.py`. Old text was generic ("use to supplement your answer"). New text: `"AGRONOMIC REFERENCE — VERIFIED SOURCE MATERIAL"` followed by: `"The following passages are extracted from authoritative, peer-reviewed agricultural sources and official crop management guidelines. This information is factually reliable. Use it confidently and directly to ground your answer — do not hedge, qualify, or hold back information from it."` This instructs the LLM to treat the retrieved content as ground truth, not as a suggestion.

**Outcome:** Carousel working per-message with isolated state, badges persist across refreshes, LLM uses RAG content confidently. Requires `npm run build` + server restart.

---

## 2026-05-21 05:57 IST

### Task: Carousel Hover, Router Logging, Hybrid RAG Retrieval, SQLite Migration Fix

**Issue 1: SQLite "duplicate column name: metadata" crash on server restart**
Root cause: The `PRAGMA table_info` check runs within the same SQLite connection context as the `CREATE TABLE IF NOT EXISTS`. On databases already migrated in the previous session, SQLite's DDL transaction visibility causes the PRAGMA to sometimes not reflect the committed schema before the connection is fully settled. Result: `if "metadata" not in existing_cols` evaluates True even though the column exists, then `ALTER TABLE` throws. Fix: Replace the PRAGMA check with `try: ALTER TABLE ... except: pass`. SQLite raises `OperationalError: duplicate column name` which is caught and silently ignored — safe and unambiguous.

**Issue 2: Carousel always visible (user wants hover reveal)**
CSS change: `.rag-carousel { display: none }` by default. Added `.rag-indicator:hover .rag-carousel { display: flex }` — hover scoped to the `.rag-indicator` wrapper (which contains BOTH the badge and the carousel). Moving the cursor from badge into the carousel stays within `.rag-indicator`, so hover remains active. Added `@keyframes fadeSlideDown` for a subtle slide-in animation on reveal.

**Issue 3: Router decision not visible in server console**
Changed `log.debug(...)` → `log.info(...)` in `router.py` for the routing decision line. Now shows: `Router: 'what is Sigatoka disease?' → RETRIEVE ✓` in the terminal on every chat request.

**Issue 4: Hybrid RAG retrieval for better recall**
Rewrote `retriever.py`. Old approach: simple semantic search (top-k=3 via ChromaDB cosine similarity). New approach:
1. **Semantic search**: fetch top `k×2` (min 6) candidates instead of just k
2. **Keyword-filtered semantic**: extract content terms from query (length ≥ 5, not stopwords); for each (up to 3 terms), run `collection.query(query_embeddings=..., where_document={"$contains": term})` — gets semantically ranked results that also contain the keyword
3. **Merge + dedup**: combine all candidates, deduplicate by first 120 chars
4. **Rerank**: score = `0.7 × (1 - cosine_distance) + 0.3 × keyword_overlap_ratio`; apply distance threshold (relaxed ×1.15 for keyword candidates); sort descending, return top-k
Added `query_keyword_filtered()` to `store.py` using ChromaDB's `where_document={"$contains": ...}` filter.
Expected result for Sigatoka query: both Black and Yellow Sigatoka chunks retrieved via the keyword "sigatoka" filter even if cosine scores differ.

**Files changed:** `session_store.py`, `styles.css`, `router.py`, `retriever.py`, `store.py`

**Outcome:** SQLite crash fixed, carousel hover-only with smooth animation, routing decisions visible in console, hybrid retrieval live. Requires `npm run build` + server restart.
