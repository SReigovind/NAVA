# Yukthi: `pipeline.py`, `store.py`, `retriever.py`, `router.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/05_rag_and_knowledge_grounding.md](../technical/05_rag_and_knowledge_grounding.md) | [01_entry_points.md](01_entry_points.md) | [02_gathi_main_and_startup.md](02_gathi_main_and_startup.md)

**Source files:**
- [`pipeline.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/yukthi/pipeline.py)
- [`store.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/yukthi/store.py)
- [`retriever.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/yukthi/retriever.py)
- [`router.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/yukthi/router.py)

---

## `pipeline.py` — The Ingestion Pipeline

### `RAGPipeline._get_encoder()` — Lazy Embedding Model

```python
def _get_encoder(self):
    if self._encoder is None:
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(self.embed_model)
    return self._encoder
```

The `SentenceTransformer` model is loaded lazily on first call. In `ingest.py`, this is called only when actually ingesting — not when running `--status`. In `startup.py`, `retriever.warm_up()` triggers early loading.

### `_find_sources(crop)` — Source Discovery

```python
def _find_sources(self, crop: str) -> list[Path]:
    crop_dir = self.source_dir / crop.lower().strip()
    if not crop_dir.exists():
        return []
    return sorted(
        f for f in crop_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
```

Returns all files in `ragsource/{crop}/` with supported extensions (`.txt`, `.pdf`). `sorted()` ensures deterministic processing order — the chunk indices are assigned in file-then-line order, so consistent ordering produces consistent chunk IDs.

### `list_available_crops()` — Auto-Discovery

```python
def list_available_crops(self) -> list[str]:
    return sorted(
        d.name for d in self.source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
```

Scans `ragsource/` for non-hidden subdirectories. Each subdirectory is a crop. No hardcoded list required — adding a new crop means creating a new folder and placing source documents in it.

### `ingest(crop, force)` — The Main Ingestion Method

```python
def ingest(self, crop: str, force: bool = False) -> int:
    if force and self.store.collection_exists(crop):
        self.store.delete_collection(crop)  # wipe existing

    all_chunks: list[Chunk] = []
    for source_path in sources:
        chunks = chunk_file(source_path, crop)
        all_chunks.extend(chunks)

    # Batch embedding — all chunks in one call
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    # Deterministic IDs for idempotent upsert
    ids = [f"{c.source}_{c.chunk_index}" for c in all_chunks]

    self.store.upsert(crop=crop, ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(all_chunks)
```

**Batch embedding:** All chunks are embedded in a single `encoder.encode(texts, batch_size=64)` call. This is far more efficient than encoding one chunk at a time — the SentenceTransformer model has a fixed overhead per call (tokenisation, batching), so batching 64 chunks at once uses the same overhead as encoding 1 chunk, at 64× the throughput.

**Deterministic IDs:** `"{source}_{chunk_index}"` produces the same ID for the same chunk on every ingestion run. ChromaDB's `upsert()` (not `add()`) is used: if the ID already exists, the document and embedding are updated; if not, they are inserted. This makes ingestion idempotent — running it twice without `--force` produces the same result as running it once.

**`force` mode:** Deletes the entire collection before re-ingesting. Necessary when the source document has been updated (the old chunks must be removed, not just overlaid, in case the new version is shorter).

---

## `store.py` — The ChromaDB Wrapper

### `YukthiStore.__init__` and `_init_client()`

```python
def __init__(self, chroma_dir: Path):
    self.chroma_dir = chroma_dir
    self.chroma_dir.mkdir(parents=True, exist_ok=True)
    self._client = self._init_client()
```

ChromaDB is initialised eagerly at construction time. As the module docstring explains: *"ChromaDB 1.5.x has threading issues when PersistentClient is first created inside a FastAPI/uvicorn request thread."* By constructing it in `__init__` (which is called from `startup.py`'s `_load_yukthi()`, running in the main thread), the Rust FFI initialisation happens before any request threads are spawned.

**Graceful fallback to `EphemeralClient`:**
```python
try:
    client = chromadb.PersistentClient(path=str(self.chroma_dir))
    return client
except Exception as e:
    log.error("Falling back to EphemeralClient — data will NOT persist.")
    return chromadb.EphemeralClient()
```

`EphemeralClient` is an in-memory ChromaDB instance. If the persistent client fails (corrupted index, file permissions), the fallback lets the server start — RAG won't survive restarts, but it won't crash the server.

### `collection_name(crop)` — Naming Convention

```python
def collection_name(self, crop: str) -> str:
    return "nava_" + crop.lower().strip().replace(" ", "_").replace("-", "_")
```

Collection names are normalised: `"Banana"` → `"nava_banana"`. This prevents case and separator mismatches between ingestion (which creates the collection) and retrieval (which queries it). The `nava_` prefix namespaces NAVA's collections from any other ChromaDB collections in the same directory.

### `query()` vs. `query_keyword_filtered()`

**`query()` — Pure semantic search:**
```python
return collection.query(
    query_embeddings=[embedding],
    n_results=min(n_results, count),
    include=["documents", "metadatas", "distances"],
)
```

Returns the N nearest neighbours by cosine distance. `min(n_results, count)` prevents requesting more results than exist in the collection (which would raise a ChromaDB error).

**`query_keyword_filtered()` — Keyword + semantic:**
```python
return collection.query(
    query_embeddings=[embedding],
    where_document={"$contains": term},  # hard filter: document MUST contain term
    n_results=min(n_results, count),
    include=["documents", "metadatas", "distances"],
)
```

`where_document={"$contains": term}` is ChromaDB's full-text substring filter. It is case-sensitive and applied before nearest-neighbour search — only documents containing `term` are included in the search space. The query then returns the nearest neighbours within that filtered set.

**Why allow query failure for keyword searches?** If no document contains the keyword (e.g., a typo in the extracted keyword), ChromaDB raises an exception rather than returning an empty result. The `except Exception` in `query_keyword_filtered()` catches this and returns `None`, which the retriever treats as "no results from this keyword". The semantic results are still returned.

---

## `retriever.py` — The Hybrid Retrieval Logic

### `RAGChunk` Dataclass

```python
@dataclass
class RAGChunk:
    text: str
    source: str
    section: str
    score: float   # cosine distance — lower is more similar
```

The public API of the retriever. Route handlers and `ChatService` work with `RAGChunk` objects — they don't need to know about ChromaDB's internal response format.

### The Hybrid Pipeline in `query()`

**Step 1 — Semantic search (5 candidates):**
```python
SEMANTIC_N = 5
semantic_results = self.store.query(crop=crop_norm, embedding=embedding, n_results=SEMANTIC_N)
```

**Step 2 — Deduplication-aware accumulation:**
```python
seen: set[str] = set()
def _add(doc, meta, dist):
    key = doc[:120]   # first 120 chars as dedup key
    if key not in seen:
        seen.add(key)
        candidates.append((doc, meta, dist))
```

The deduplication key is the first 120 characters of the document text. This catches exact duplicates (the same chunk retrieved by both semantic and keyword search) while being robust to minor whitespace differences.

**Step 3 — Keyword searches:**
```python
KEYWORD_PER_TERM = max(1, round(5 / max(len(search_keywords), 1)))
for term in search_keywords:
    kw_results = self.store.query_keyword_filtered(crop=crop_norm, embedding=embedding, term=term, n_results=KEYWORD_PER_TERM)
```

For 3 keywords: `round(5 / 3)` = 2 results per keyword, yielding up to 6 keyword candidates (after deduplication: typically 3–5 new unique ones).

**Step 4 — Reranking:**
```python
for doc, meta, dist in candidates:
    if dist > self.distance_threshold * 1.2:
        continue  # drop far-away chunks
    kw = self._keyword_score(doc, search_keywords)
    combined = 0.7 * max(0.0, 1.0 - dist) + 0.3 * kw
```

`1.0 - dist` converts cosine distance to cosine similarity. Keyword candidates are given a relaxed distance threshold (`* 1.2`) because the keyword filter already guarantees topic relevance — a slightly higher distance is acceptable if the keyword is present.

`combined = 0.7 × semantic_similarity + 0.3 × keyword_overlap`: Semantic similarity dominates (we want topically relevant chunks), with keyword overlap providing a precision boost.

### `warm_up()`

```python
def warm_up(self) -> None:
    try:
        self._get_encoder()
        log.info("RAGRetriever embedding model pre-warmed.")
    except Exception as e:
        log.warning("RAGRetriever warm_up failed: %s", e)
```

Called during `startup.py`'s `_load_yukthi()`. Forces the SentenceTransformer model to load (download weights if needed, load into memory). Without this, the first RAG query would bear a 2-second model loading delay.

---

## `router.py` — The RAG Routing Gate

### `_ROUTER_SYSTEM_PROMPT`

The system prompt is carefully engineered. Key design decisions:

**"Reply with exactly one word — either RETRIEVE or SKIP."** — Hard constraint on output format. `max_new_tokens=5` ensures the model cannot exceed this.

**RETRIEVE conditions** are explicit: named diseases, treatment protocols, fertiliser rates, any question where agronomic reference material would materially improve the answer.

**SKIP conditions** are specific:
- Meta-questions about NAVA's capabilities
- Questions about the user's own farm data (those are answered from context, not the knowledge base)
- Simple conversational follow-ups

**"DEFAULT: when uncertain, always output SKIP."** — A SKIP false negative (the model misses a retrieval opportunity) is less harmful than a RETRIEVE false positive (unnecessary latency, potentially irrelevant material injected into the prompt).

### `should_retrieve()` — Context-Aware Routing

```python
def should_retrieve(self, message: str, last_assistant_reply: str = "") -> bool:
    if not stripped:
        return False

    if last_assistant_reply and last_assistant_reply.strip():
        routing_input = (
            f"[Previous NAVA response]: {last_assistant_reply.strip()[:300]}\n"
            f"[User]: {stripped}"
        )
    else:
        routing_input = stripped
    
    return self._llm_classify(routing_input)
```

The previous NAVA reply (truncated to 300 characters) is included in the routing input. This is the critical detail that makes the router handle short follow-ups correctly. Without it:
- NAVA: "Black Sigatoka requires Propiconazole treatment. Would you like dosage details?"
- User: "Yes"
- Router input: `"Yes"` → classified as SKIP (it's an acknowledgement)

With the previous reply:
- Router input: `"[Previous NAVA response]: Black Sigatoka requires... Would you like dosage details?\n[User]: Yes"` → classified as RETRIEVE (the user is confirming they want detailed agronomic information)

The system prompt explicitly includes this case: "digital agronomist prompts for more information... user acknowledges to provide more information by responding with 'yes' or 'yeah'...".

### `_llm_classify()` — The Classification Call

```python
reply, error = self.client.send(
    prompt,
    model_override=self.model,       # 8B model
    temperature_override=0.0,        # fully deterministic
    max_new_tokens_override=5,       # "RETRIEVE" is 1 token, "SKIP" is 1 token
)
decision = reply.strip().upper()
result = decision.startswith("RETRIEVE")
```

`startswith("RETRIEVE")` rather than `== "RETRIEVE"` handles cases where the model produces `"RETRIEVE."` or `"RETRIEVE\n"` — common trailing punctuation despite the instruction.

On error or timeout, the function returns `False` (SKIP). This is the safe default — a network error during routing should not cause the chat request to fail.
