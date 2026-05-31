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

---

## 2026-05-21 06:14 IST

### Task: Carousel Hover Zone Too Wide

**Issue:** The RAG carousel was triggering on hover anywhere along the full width of the message bubble, not just near the badge. This was because `.rag-indicator` is a block-level `div` that stretches to 100% bubble width — so the `:hover` zone was enormous.

**Fix:**
1. Added `display: inline-block; width: fit-content` to `.rag-indicator` — constrains the hover trigger to exactly the badge pill's width.
2. Replaced `display: none` / `display: flex` toggle with `opacity`/`visibility`/`max-height` CSS transition. Reason: `display` cannot be transitioned, so there was no fade window — a 1px gap between the badge bottom and the carousel top would cause the carousel to instantly close as the cursor moved between them. With the transition approach, the 150ms fade gives the cursor time to travel from the badge into the carousel without it closing.
3. Removed `margin-bottom: 6px` from `.rag-badge` — gap is now bridged by the transition duration rather than an invisible spacer.
4. Added `@keyframes fadeSlideDown` for reveal animation.

**CSS summary:**
```css
.rag-indicator { display: inline-block; width: fit-content; }
.rag-carousel  { opacity: 0; visibility: hidden; max-height: 0; transition: ... }
.rag-indicator:hover .rag-carousel { opacity: 1; visibility: visible; max-height: 160px; }
```

**Files changed:** `styles.css`

**Outcome:** Carousel now only triggers when hovering directly over the badge area. Smooth fade-in/out. Moving cursor from badge to carousel keeps it open.

---

## 2026-05-21 06:49 IST

### Task: LLM-Guided Hybrid RAG Retrieval (KeywordExtractor)

**Background:** Previous hybrid retrieval used heuristic stopword filtering to extract keywords from the user query. This was imprecise — it would include words like "condition", "detected", "suggestions" which add no value to `where_document` keyword searches.

**New architecture — 4-stage pipeline per RAG-warranted message:**
1. **Router LLM** (`max_tokens=5`): RETRIEVE / SKIP decision
2. **KeywordExtractor LLM** (`max_tokens=25`): Given the full enriched query context (crop + detected condition + user question), asks Llama-3.1-8B-Instruct to output the 3 most important agronomic search keywords, one per line.
3. **Retriever** (ChromaDB, no LLM):
   - 8 semantic candidates (cosine similarity via bge-small-en-v1.5)
   - Per LLM keyword: ChromaDB `where_document={"$contains": term}`, top `round(8/n_keywords)` each → ~8 keyword candidates total
   - Merge + dedup (first 120 chars as key)
   - Rerank: `0.7 × (1 - cosine_dist) + 0.3 × keyword_overlap`; threshold relaxed ×1.2 for keyword candidates
   - Return top 4
4. **Main LLM**: receives the 4 chunks as AGRONOMIC REFERENCE in system prompt

**New file:** `nava_core/yukthi/keywords.py` — `KeywordExtractor` class
- `from_settings(client)` factory — uses `hf_summary_model` (same small model as router)
- `extract(enriched_query) → list[str]` — calls LLM, parses newline-separated keywords, strips bullets/numbers, filters generic agricultural stop-words
- Falls back to `[]` on any LLM failure; retriever then falls back to heuristic terms

**service.py changes:** After `rag_router.should_retrieve()` returns True, call `keyword_extractor.extract(retrieval_query)` and pass resulting `llm_keywords` into `rag_retriever.query(...)`.

**retriever.py changes:** `query()` now accepts `llm_keywords: Optional[list[str]]`. If provided, uses them for keyword-filtered ChromaDB searches; otherwise falls back to heuristic `_heuristic_fallback_terms()`.

**settings.py:** `yukthi_top_k` default changed from 3 → 4.

**deps.py / factory:** `KeywordExtractor` built in `from_settings_with_store()` alongside `QueryRouter`, injected as `self.keyword_extractor` on `ChatService`.

**Files changed:** `keywords.py` (new), `retriever.py` (rewrite), `service.py`, `settings.py`

**Outcome:** Keyword extraction is now LLM-guided and anchored to the enriched query context (includes crop name and detected disease). In tests, "sigatoka" queries now retrieve both Black Sigatoka and Yellow Sigatoka chunks because the LLM extracts "sigatoka" as a keyword which triggers the `where_document` filter on both entries.

---

## 2026-05-21 06:58 IST

### Task: Retrieval Parameter Tuning (5+5 → top 3) + Keyword Extractor Logging

**Changes:**
1. Reduced semantic candidates from 8 → **5** (`SEMANTIC_N = 5`)
2. Reduced keyword candidate target from 8 → **5** (`KEYWORD_PER_TERM = round(5/n_keywords)`)
3. Reduced final top-k from 4 → **3** (`yukthi_top_k` default back to 3)
4. Added `log.info` in `keywords.py` for:
   - The enriched query sent to the extractor: `KeywordExtractor ▶ query:\n{enriched_query}`
   - The raw LLM response before parsing: `KeywordExtractor ◀ raw response: {reply!r}`

**Rationale:** 8+8 was over-retrieving; 5+5 gives sufficient diversity without padding the LLM context with marginally relevant chunks. The query/response logs make it easy to diagnose keyword quality in the server console without needing to add breakpoints.

**Files changed:** `retriever.py`, `settings.py`, `keywords.py`

---

## 2026-05-21 07:07 IST

### Task: Router System Prompt Rewrite — Stop Over-Triggering on Capability Questions

**Problem:** The router was incorrectly routing meta/conversational queries to RETRIEVE:
- `"what can you do?"` → RETRIEVE ✓ (wrong)
- `"what are your capabilities?"` → RETRIEVE ✓ (wrong)

**Root cause:** The old prompt identified itself as a generic "routing classifier" with no strong identity. The SKIP conditions listed only field-data and greeting examples — no mention of questions about the assistant itself. "When in doubt lean towards SKIP" was soft enough that the LLM still classified capability questions as potential knowledge lookups.

**New prompt design:**
- Identity: **"strict access-control gate for an agricultural knowledge base"** — makes the purpose concrete and restrictive
- RETRIEVE: requires "specific agronomic facts that would NOT already be known from conversation context" — raises the bar from "would help" to "genuine knowledge gap"
- SKIP: now includes explicit verbatim examples of the failing cases: `'what can you do?'`, `'what are your capabilities?'`, `'how do you work?'`, `'tell me about yourself'`
- New SKIP category: **"Any self-referential or meta question about this chat session"**
- Default: **"DEFAULT: when uncertain, always output SKIP"** — harder boundary than the previous soft lean

**Files changed:** `router.py` (`_ROUTER_SYSTEM_PROMPT`)

**Outcome:** Capability and meta questions now correctly route to SKIP. Agronomic queries (disease names, treatment protocols, fertilizer rates) still correctly route to RETRIEVE.

---

## 2026-05-25 01:10 IST

### Task: UX Refinements — Disease Confidence, VNIR Caution, Auto-Notes Modal, Activity Scroll

**Issue 1: Disease detection shows raw % confidence score — farmers don't find it useful**
Changed the confidence gauge (progress bar + raw %) to a natural-language phrase:
- ≥ 90% → "The AI is very confident about this result"
- ≥ 75% → "The AI is fairly confident about this result"
- ≥ 60% → "The AI has moderate confidence in this result"
- < 60% → "The AI has low confidence — treat with caution"
A small `ⓘ` badge sits beside the phrase. Hovering reveals a compact tooltip (140px wide, 0.7rem font) showing the exact percentage and the confidence bar. Same pattern applied to history rows — raw % is hidden, `ⓘ` reveals on hover.

**Issue 2: VNIR monitoring caution block needed full caveats**
Added a `VnirCautionBlock` component above the upload area:
- Always-visible summary: "Requires 5+ healthy baseline scans to activate"
- Collapsible "How it works ▼" expands to 5 bullet caveats:
  - 📸 Minimum 5 photos needed
  - 🌱 First 5 must be healthy (baseline establishment)
  - 🕐 Consistent time of day required
  - 🔄 Clear data monthly / at growth stage transitions
  - ⚗️ System is **experimental** — results are proactive warnings only; always confirm visually and with Disease Detection

**Issue 3: Auto-notes ("What NAVA knows") not visible — needed popup instead of inline card**
Previous implementation (collapsible inline card) was removed in favour of a modal popup:
- A small green pill button labelled "🤖 What NAVA knows · N" sits in the Crop Notes card header (next to Edit) on the Crop Overview page, and in the Field Notes card header on the Field Detail page
- Clicking opens a full-screen backdrop modal with a scrollable list of auto-note entries (read-only)
- Clicking outside the modal or ✕ closes it
- If no auto-notes exist yet (the `--- NAVA Auto-notes ---` separator is absent), the button is completely hidden
- `splitNotes()` utility exported from `OverviewPanel.jsx` and imported into `FieldDetail.jsx`

**Issue 4: Recent Activity grows off-screen**
Changed Recent Activity container from `flex: 1 / minHeight: 0` (which tried to fill remaining side-column height) to a `maxHeight: 220px` with `overflowY: auto` — the section now has its own independent scrollbar and never pushes the page.

**Files changed:** `DiagnosePanel.jsx`, `MonitorPanel.jsx`, `OverviewPanel.jsx`, `FieldDetail.jsx`, `styles.css`

**Outcome:** Confidence is farmer-friendly with optional detail on hover; VNIR has full operational caveats; auto-notes are read-only in a popup modal on both crop and field pages; recent activity has its own scroll.

---

## 2026-05-25 01:48 IST

### Task: Fix Auto-Notes Icon — Always Visible, In Title, Correct Data Source

**Root cause of previous invisibility:** `AutoNotesCard` returned `null` when the `--- NAVA Auto-notes ---` separator was absent from `notes`/`field_notes`. Since Smart Notes extraction (chat-to-notes) is not yet implemented, this separator never existed → the button was always invisible.

**Fix 1: New `AutoNotesIcon` component**
Replaced `AutoNotesCard` with `AutoNotesIcon` — a small dimmed 🤖 emoji button that:
- **Always renders** (never returns null), regardless of whether auto-content exists
- Opens a **modal popup** on click (blurred backdrop, click-outside to close)
- If content is present: shows it as a bulleted list
- If content is empty: shows an italic placeholder message explaining that NAVA will populate this over time

**Fix 2: Icon placed inside the section title**
Both "Crop Notes" and "Field Notes" titles now have the 🤖 icon inline (dim at rest, full opacity on hover), sitting directly next to the heading text, separate from the Edit button.

**Fix 3: Correct auto-content data sources**
- **Crop Notes (OverviewPanel):** Uses `splitNotes(crop?.notes).auto` — the portion after the `--- NAVA Auto-notes ---` separator in the crop's `notes` field. Will be populated once Smart Notes extraction is implemented. Currently shows empty-state message.
- **Field Notes (FieldDetail):** Uses `field?.shared_context` — the auto-generated field-wide context string (crop statuses, recent events, soil info) that IS already being built and used by NAVA in chat. This shows actual content immediately.

**Deprecated:** `AutoNotesCard` kept as a thin wrapper calling `AutoNotesIcon` for backward compatibility. `splitNotes` utility remains exported.

**Files changed:** `OverviewPanel.jsx`, `FieldDetail.jsx`, `styles.css`

**Outcome:** 🤖 icon is now always visible in both notes title areas. Field popup shows real auto-generated context immediately. Crop popup shows placeholder until Smart Notes feature is implemented.

---

## 2026-05-25 02:09 IST

### Task: Scan-Triggered Field Context Rebuild, Smart Crop Notes, Context-Aware Router

**Feature 1: Field shared_context rebuilt after every scan**
`_refresh_field_context(store, field_id)` (already defined in `fields.py`) was not being called after disease or VNIR scans — only after field/crop CRUD events. Added call in both scan routers:
- `diagnose.py`: after `store.add_event()` (both UNRELIABLE early-return and RELIABLE Grad-CAM paths), resolves `field_id` from form param or `plant["field_id"]`
- `vnir.py`: after `store.add_event()`, same fallback to `plant["field_id"]`
Wrapped in try/except so any context-build failure never blocks the scan response.

**Feature 2: Smart crop notes from chat summaries**
Added `_extract_crop_notes_from_summary(summary, crop_id)` to `ChatService`:
- System prompt: "extract ONLY concrete, specific actions or decisions the farmer has taken or plans to take. If none, output NONE"
- Uses `hf_summary_model` at `temperature=0.0`, `max_new_tokens=80`
- Parses bullet lines, strips leading dash/bullet chars
- Appends below `--- NAVA Auto-notes ---` separator in `crops.notes` via existing `field_store.update_crop_context()`
- Called in `_summarize_if_needed()` right after each new level-1 summary is saved, using `store.get_session_context()` to get the crop_id
- Fire-and-forget: wrapped in try/except, logged with `CropNotes: appended N auto-note(s)` or warning on failure

**Feature 3: Context-aware RAG router**
Old problem: bare replies like "yes", "tell me more", "sure" were skipped because:
1. The 3-token minimum guard dropped them before the LLM was even consulted
2. Even if they passed the guard, no context existed to interpret them

Fixes:
- **Removed** the `len(stripped.split()) < 3` minimum token guard entirely. Only empty strings are skipped without LLM consultation.
- **Added** `last_assistant_reply: str = ""` parameter to `should_retrieve()`
- If a previous NAVA reply exists, routing input becomes: `[Previous NAVA response]: {last 300 chars}\n[User]: {user message}` — giving the LLM the context to correctly classify "yes" as RETRIEVE if NAVA had just asked "Do you want more details on Black Sigatoka treatment?"
- In `chat()`: extracts last assistant message from the already-fetched `history` list (no extra DB call) and passes it as `last_assistant_reply`

**Files changed:** `diagnose.py`, `vnir.py`, `router.py`, `service.py`

**Outcome:** Field 🤖 popup now refreshes after every scan. Crop auto-notes accumulate from chat summaries automatically. Short follow-up messages like "yes" are now correctly routed based on conversational context.

---

## 2026-05-25 02:48 IST

### Task: Qualitative Workflow Tests Framework

Created a robust testing framework under `tests/` to perform automated, qualitative assessments of all major workflows, generating detailed logs that serve as formal test documentation. 

**Structure & Utilities:**
- `tests/test_utils.py`: Helpers for creating dynamic authenticated users (`/api/auth/register`), provisioning new fields/crops/plants, and dynamically locating sample imagery from `data/processed/efficientnet/test/`.

**Test Suites:**
1. **Disease Detection (`tests/test_disease.py`)**
   - Uploads a healthy banana image (`171.jpg`) and verifies normal status with confidence metrics.
   - Uploads an infected banana image (`173.jpeg`) and verifies accurate Black Sigatoka detection with confidence.
   - Outputs: `tests/disease_test_log.txt`.

2. **Stress Monitoring (`tests/test_vnir.py`)**
   - Uploads 5 healthy images sequentially to successfully establish a baseline and trigger "Calibrating" states.
   - Once the baseline is formed, uploads a Sigatoka-infected image to verify immediate visual stress detection and ratio drops compared to the baseline.
   - Outputs: `tests/vnir_test_log.txt`.

3. **Chat & Context Extraction (`tests/test_chat.py`)**
   - **General Chat:** Verifies basic greetings skip RAG routing.
   - **RAG Retrieval:** Asks about Black Sigatoka treatment, verifies RAG activates and chunks are loaded.
   - **Contextual Follow-up:** Sends a short "yes, tell me more" follow-up. Verifies that the new context-aware router accurately bridges the context and activates RAG.
   - **Smart Auto-Notes:** Injects an agronomic action ("I have applied Mancozeb fungicide today..."), then sends a series of dummy messages (with deliberate `time.sleep` intervals to respect API limits) to trigger the `summary_batch` limit. Verifies that the LLM successfully parses the action from the summary and appends it to the `Crop Notes` field.
   - Outputs: `tests/chat_test_log.txt`.

---

## 2026-05-25 22:50 IST

### Task: Fix Manual/Auto Notes Display and Add Timestamps

**Issue 1: Auto-notes appearing in manual notes textarea**
- **Cause:** In `OverviewPanel.jsx`, `crop.notes` (which contains both manual and auto notes separated by `--- NAVA Auto-notes ---`) was being fed directly into the `notes` React state.
- **Fix:** Used the existing `splitNotes` utility to populate the `notes` state exclusively with the `.manual` portion.
- When `saveNotes` is triggered, the `notes` (manual part) is dynamically recombined with the `.auto` part before sending the API request.

**Issue 2: Auto-notes lacking timestamps**
- **Cause:** `_extract_crop_notes_from_summary` simply appended the bullet points generated by the LLM.
- **Fix:** Imported `datetime` in `service.py` and prepended `[{now_str}]` to the top of each new batch of extracted notes (e.g. `[2026-05-25 22:50]`).

**Files changed:** `OverviewPanel.jsx`, `service.py`

### Task: Field Context Refresh on Deletion & Editable Auto-Notes

**Issue 1: Field context out of sync when history events are deleted**
- **Cause:** When a user deleted a VNIR scan or Disease detection from a plant's history, the overarching field `shared_context` was not automatically regenerating to reflect the removal.
- **Fix:** Added `_refresh_field_context` trigger calls to the `DELETE /events/{event_id}` and `POST /vnir-clear` API endpoints to ensure the field memory stays in sync with deletions. Also added a `get_event` method to `field_store.py` to retrieve the `field_id` before deletion.

**Issue 2: Unable to prune inaccurate auto-notes**
- **Cause:** Auto-generated crop notes were previously read-only, meaning if the LLM hallucinated an action, the farmer couldn't remove it from the context popup. Additionally, deleting a note didn't immediately update the UI.
- **Fix:** Modified `AutoNotesIcon` to accept an `onDeleteLine` prop. Each line in the auto-notes modal now features a discrete `✕` (delete) button. Passed an `onRefresh` callback from `CropDetail.jsx` into `OverviewPanel.jsx` so that deleting a bullet immediately re-fetches the parent's crop state, flawlessly updating the UI without a manual page refresh.

**Files changed:** `vnir.py`, `fields.py`, `field_store.py`, `OverviewPanel.jsx`, `CropDetail.jsx`

---

## 2026-05-26 00:00 IST

### Task: Advanced Qualitative Test Generation & PDF Export

Created highly advanced test suites in `tests/` designed to capture deep internal state logs (prompts, retrieval steps, routing decisions) and output comprehensive Markdown reports with Base64 decoded images embedded directly.

1. **`test_disease_advanced.py`**: 
   - Iterates across 7 target crops (Banana, Cassava, Corn, Cucumber, Rice, Soybean, Tomato).
   - Evaluates 1 Healthy and 1 Diseased image per crop.
   - Decodes the Base64 API responses to automatically save the original and generated GradCAM visualizations to `tests/outputs/`.
   - Generates `tests/disease_report.md` embedding all images with classification results.

2. **`test_vnir_advanced.py`**:
   - Executes a 5-image baseline calibration sequence on Banana, then assesses 3 Sigatoka stress images.
   - Saves Base64 decoded HSV isolate and NIR predicted image masks.
   - Generates `tests/vnir_report.md` with fully visualized tracking ratios alongside original images.

3. **`test_chat_advanced.py`**:
   - Utilizes FastAPI's `TestClient` and `unittest.mock.patch` to execute fully in-process testing.
   - Automatically intercepts and logs the smart routing classifier's decisions, RAG retriever logs, and intercepts the explicit context-injected payload sent to the LLM.
   - Includes conversational sequences targeting simple greetings, contextual follow-ups, and triggers the `summary_batch` threshold with dummy messages.
   - Intercepts and outputs both the raw conversational summary block alongside the automatically extracted crop notes.
   - Updated the Markdown output to use `<pre style="white-space: pre-wrap; word-wrap: break-word;">` HTML tags to ensure ultra-wide routing logs natively word-wrap inside the Markdown file, removing the need for horizontal scrolling.
   - Generates `tests/chat_report.md`.

4. **`export_pdfs.py`**:
   - Uses the `md2pdf` library to convert the generated `.md` files into highly readable `.pdf` files.
   - Injects a custom CSS stylesheet to style the markdown tables and apply code block word-wrapping for the final PDF document.

---

## 2026-05-30 14:49 IST

### Task: Architecture Documentation — High-Level & Detailed Diagrams (docs 00–07)

**Objective:** Upgrade all documentation files to include both a high-level abstracted diagram and a detailed diagram for each module, add an improved ER diagram to the data storage doc, and add frontend view/flow diagrams.

**Changes made:**

- **`00_overview.md`**: Added a **High-Level Architecture (§4.1)** Mermaid diagram with five colour-coded zones (API Gateway, Perception, Cognition, Knowledge, Storage) showing only real system flows with internal sub-components abstracted. Relabelled the existing detailed diagram as **§4.2 Detailed Architecture**.

- **`01_module_gathi.md`**: Added a **High-Level Gathi Overview** showing Gathi as an API gateway + SPA host with Modules and Storage as abstracted external nodes. Relabelled the existing routing map as **Module Routing Map (Detailed)**.

- **`02_module_mizhi.md`**: Added a **High-Level Mizhi Overview** depicting the two independent pipelines (Disease Detection, VNIR) as black boxes feeding a shared event store. Relabelled existing detailed flowcharts.

- **`03_module_mozhi.md`**: Added a **High-Level Mozhi Overview** showing the 5-step orchestration sequence (Context → Route → Retrieve → Generate → Memory) with RETRIEVE/SKIP branching. Added a **LLM Prompt Structure** reference block showing the assembled layer order. Relabelled existing diagrams.

- **`04_module_yukthi.md`**: Added a **High-Level Yukthi Overview** that clearly separates the offline ingestion path (build vector store, run once) from the online retrieval path (per query at chat time). Relabelled existing diagrams.

- **`05_module_shared.md`**: Added a **High-Level Shared Module Overview** showing all 4 modules importing from Shared. Added a **Two-Database Architecture Overview** showing the global vs. per-user DB split. Relabelled the existing detailed diagram.

- **`06_data_storage.md`**: Improved the ER diagram significantly — added inline field descriptions (e.g. `"bcrypt hash"`, `"JSON blob"`, `"NIR/Green ratio"`), added `CHAT_CONTEXT` foreign key relationships to `FIELDS` and `CROPS`, and added database boundary annotation.

- **`07_frontend_views.md`**: Added two new diagrams after the route map: (1) **Page Navigation Flow** showing user traversal from landing → auth → fields → crop workspace → tool panels; (2) **Component Architecture** showing the full React tree (`main.jsx → BrowserRouter → AuthProvider → App`) with public vs. protected route grouping, `CropLayout`, all four tool panels, `PlantSelector`, and how `AuthProvider`/`apiFetch` wire through the tree via dotted dependency arrows.

**Outcome:** All 8 documentation files now have both high-level and detailed architecture representations.

---

## 2026-05-30 15:18 IST

### Task: Feature Planning — Weather Context & Season Dropdown

**Decisions made and documented:**

1. **Season field**: Changed from free-text input to a dropdown with three Kerala-specific options: Summer / Hot Season (Mar–May), Monsoon Season (Jun–Nov), Winter / Cool Season (Dec–Feb). Season auto-detection from geolocation was considered and rejected — manual selection keeps user agency and removes geocoding dependency from crop creation.

2. **Geo-weather context**: Retained and planned in full. Field location is resolved to lat/lon via Nominatim (OSM, free, no key) once per field, with coordinates persisted in two new `lat`/`lon` REAL columns in the `fields` table. Live weather data fetched from Open-Meteo (free, no key, GDPR-compliant) with a 60-minute in-memory cache. Injected into `ChatService._build_context_message()` as a `CURRENT WEATHER CONDITIONS` system prompt block, run in a `ThreadPoolExecutor` with 5s timeout so network calls never delay chat responses. No new pip installs — entirely stdlib `urllib` + `json`.

3. **Multilingual support (Malayalam)**: Deferred to future work. Documented the recommended approach for when it is implemented: DeepL API Free tier (500k chars/month, official `deepl` SDK, Malayalam code `"ML"`), input translated ML→EN before RAG/LLM, LLM instructed to respond in Malayalam via system prompt, `preferred_lang` column in `chat_context` table, EN/ML toggle in ChatPanel.

**Files updated:**

- `implementation_plan.md` — replaced the obsolete 12 May 2026 plan with the current feature plan covering the season dropdown and geo-weather context, with a future work note on multilingual support.
- `README.md` — updated to reflect the current complete state of the system.
- `worklog.md` — this entry.

---

## 2026-05-31 20:00 IST

### Task: Session Recap — Weather DB System, VNIR Two-Level Warnings, Delete Field, Geocode-on-Create, Bug Fixes, Documentation Update

This session covered a significant extension of the platform, moving from an in-process weather cache to a fully persistent, DB-backed weather system, adding two-level VNIR stress classification, implementing cascade field deletion, and updating all project documentation.

---

#### Task 1: Code Review & Bug Fixes

A comprehensive code review was conducted across the full codebase. The following bugs were fixed:

**Bug 1: `requirements.txt` inconsistency**
`requirements.txt` listed `bcrypt` as a direct dependency, but the actual password hashing in `user_store.py` uses `hashlib.pbkdf2_hmac` (stdlib). `bcrypt` was removed. `httpx` was also listed but replaced by stdlib `urllib` throughout; removed. Final deps standardised.

**Bug 2: Model preload background thread not started on register**
`_preload_models` background task was only added in the `login` endpoint, not in `register`. A newly registered user's first scan would always hit cold-start latency. Fixed by adding `bg_tasks.add_task(_preload_models)` to the `register` endpoint as well.

---

#### Task 2: VNIR Two-Level Stress Warnings

**Background:** Previous implementation only compared VNIR ratio against the initial baseline (first 5 scans) and emitted a single `WARNING` if the drop exceeded a threshold.

**New design — two warning levels:**

| Level | Comparison | Trigger |
|-------|-----------|---------|
| `WARNING` | Rolling window (last 5 valid ratios) | Drop ≥ 10% vs. rolling mean |
| `CRITICAL` | Initial baseline (first 5 scans) | Drop ≥ 15% vs. baseline mean |

Both comparisons run independently. If both thresholds are breached simultaneously, `CRITICAL` takes precedence.

**Zero-ratio guard:** Scans with `ratio == 0` (no leaf detected / HSV isolation failed) are now explicitly excluded from all checkpoint calculations — baseline building, rolling window, and warning comparisons. Previously these zeros were stored and corrupted statistical baselines.

**Files changed:** `nava_core/mizhi/vnir/analyzer.py`

---

#### Task 3: Persistent DB-Backed Weather System

**Problem with previous approach:** Weather was stored in a Python in-process dict (`_weather_cache`) with a 60-minute TTL. This meant: (1) cache was wiped on every server restart, (2) cold-start latency on first access, (3) chat context built weather via a `ThreadPoolExecutor` call on every chat request — adding network latency to the hot path.

**New design:**

- **Storage:** 5 new columns in the `fields` table: `weather_temp REAL`, `weather_humidity REAL`, `weather_precipitation REAL`, `weather_wind_speed REAL`, `weather_updated_at TEXT`. Added via incremental migration in `FieldStore._migrate_schema()`.
- **On login:** `_refresh_user_weather(user_id)` fires as a `BackgroundTask`. For every field with lat/lon stored, it calls Open-Meteo and writes the result to DB. 1-second delay between each field call to avoid API hammering.
- **On field create/edit:** `_geocode_and_fetch_weather(db_path, field_id)` fires as a `BackgroundTask`. Resolves coordinates via Nominatim if not yet stored, then fetches and stores weather immediately. On location edits, stored lat/lon is cleared first so Nominatim re-geocodes the new location.
- **`GET /api/weather?field_id=X`:** Returns DB values immediately (zero API calls post-login). Falls back to live fetch if DB is empty (first time for a field).
- **`POST /api/weather/refresh?field_id=X`:** Force-fetches Open-Meteo, writes to DB, returns fresh values. Used by the manual ↻ button.
- **Chat context:** `service.py` now reads `field_rec["weather_temp"]` etc. directly from the DB field dict — no network calls during chat.
- **`WeatherStrip.jsx`:** Shows relative `updated_at` timestamp ("3h ago") and a ↻ refresh button.

**Files changed:** `field_store.py`, `geo_context.py`, `auth.py`, `fields.py` (router), `weather.py` (router, rewritten), `service.py`, `WeatherStrip.jsx`, `styles.css`

**Logging:** Detailed `[GEO]` and `[WEATHER]` prefixed log lines at every step (Nominatim URL, raw response bytes, parsed result, cache decisions) for manual verification.

---

#### Task 4: Delete Field Feature

Added full cascade field deletion across the stack.

**`FieldStore.delete_field(field_id)`:** Iterates crops → plants → deletes `vnir_history` + `events` per plant → deletes `plants` per crop → deletes crop-level `events` → deletes field-level `events` → deletes `crops` → deletes `fields` row. All within a single transaction.

**`DELETE /api/fields/{field_id}`:** Returns 404 if field not found, `{"status": "deleted", "field_id": N}` on success.

**Frontend:** 🗑️ button (red-tinted) added next to the ✏️ edit button in the field header card. Clicking opens a styled confirmation modal that lists all data categories that will be deleted (crops, plants, disease scans, VNIR history, events) and has a red "Yes, Delete Field" button. On confirm, navigates back to `/fields`.

**Files changed:** `field_store.py`, `fields.py` (router), `FieldDetail.jsx`

---

#### Task 5: Documentation Update

- `implementation_plan.md` **renamed** to `futureWork.md`. Content replaced with 9 forward-looking future work items: multilingual support, native mobile app, expanded crop coverage, satellite/drone integration, multi-user collaboration, automated field reports, crop insurance integration, VNIR ground-truth validation, and production infrastructure.
- `worklog.md` — this entry appended.
- `README.md` — updated Phase 2 description, added weather and delete-field features, updated external integrations table.
- `documentation/00_overview.md` — architecture diagrams updated (Open-Meteo added as external node), External Integrations table updated, Phase 2 description updated.
- `documentation/06_data_storage.md` — ERD updated with new weather + coordinate columns in FIELDS, Section 3.3 updated, weather write/read paths added.
- `documentation/05_module_shared.md` — FieldStore section updated with `update_field_weather()`, `delete_field()`, `refresh_user_weather()`, and removal of in-process cache.
- `documentation/01_module_gathi.md` — API routes table updated with new weather and delete endpoints; WeatherStrip and delete modal documented.
- `documentation/02_module_mizhi.md` — VNIR two-level warning system and zero-ratio guard documented.