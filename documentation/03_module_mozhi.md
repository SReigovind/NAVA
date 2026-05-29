# Mozhi — Multilingual Chatbot & Contextual Memory

> **Module role:** The cognition layer. Mozhi transforms NAVA from a scan tool into a persistent digital agronomist — one that remembers a farm's entire history, retrieves verified knowledge on demand, and speaks naturally about what it knows.

---

## 1. What is Mozhi?

The name *Mozhi* (മൊഴി) means "language" or "voice" in Malayalam. Mozhi is the conversational intelligence of NAVA. It orchestrates the full lifecycle of a chat interaction: assembling context from the farm database, deciding whether the question needs verified reference material, retrieving relevant knowledge chunks, constructing the complete prompt, calling the LLM, persisting the conversation, and summarising old messages when the history grows too long.

Mozhi is not a thin wrapper around an LLM. It is a sophisticated orchestration engine that makes every LLM response more grounded, more contextual, and more trustworthy.

---

## 2. File Structure

```
nava_core/mozhi/
├── __init__.py
├── chat/
│   ├── __init__.py
│   ├── client.py     ← ChatClient: raw HTTP calls to HF Router API
│   └── service.py    ← ChatService: full conversation orchestration
└── memory/
    ├── __init__.py
    └── session_store.py  ← SessionStore: SQLite-backed message and summary persistence
```

---

## 3. The HTTP Client (`client.py`)

`ChatClient` is a thin, dependency-free HTTP client that calls the Hugging Face Router API — an OpenAI-compatible `/v1/chat/completions` endpoint.

```python
class ChatClient:
    def send(
        self,
        messages: list[dict],
        model_override: str | None = None,
        temperature_override: float | None = None,
        max_new_tokens_override: int | None = None,
    ) -> tuple[str | None, str | None]:
        ...
        # returns (reply_text, error_message_or_None)
```

The method sends a POST request with a standard OpenAI-compatible JSON body (model, messages, temperature, max_tokens, stream=False). It uses the `HF_API_KEY` as a Bearer token, with a configurable timeout (`HF_TIMEOUT`, default 30 seconds).

The interface accepts `model_override` and `temperature_override` parameters. This is important: Mozhi uses **two different models**:
- `HF_MODEL` (default: `meta-llama/Meta-Llama-3-70B-Instruct:novita`) — the large, high-quality model for user-facing chat responses
- `HF_SUMMARY_MODEL` (default: `meta-llama/Llama-3.1-8B-Instruct:novita`) — a smaller, faster model for internal tasks: summarisation, rollup, RAG routing, keyword extraction, and auto-notes extraction

Using the smaller model for housekeeping tasks dramatically reduces cost and latency without affecting response quality.

---

## 4. Session Storage (`session_store.py`)

`SessionStore` manages chat persistence using four SQLite tables:

```sql
CREATE TABLE chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,         -- 'user' or 'assistant'
    content     TEXT NOT NULL,
    metadata    TEXT DEFAULT NULL,     -- JSON: {rag_used, rag_chunk_count, rag_chunks}
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_state (
    session_id         TEXT PRIMARY KEY,
    last_summarized_id INTEGER DEFAULT 0  -- message ID up to which we've summarised
);

CREATE TABLE chat_summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    level      INTEGER NOT NULL,      -- 1 = recent summary, 2 = long-term rollup
    content    TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE chat_context (
    session_id TEXT PRIMARY KEY,
    field_id   INTEGER,
    crop_id    INTEGER               -- which farm entity this session is anchored to
);
```

**Key design decisions:**

- **Session IDs** are UUID hex strings generated client-side. The backend is stateless w.r.t. session creation — it accepts any session ID and initialises state on demand.
- **`metadata` column** on `chat_messages` stores RAG attribution data as JSON. When an assistant message is generated using retrieved knowledge chunks, the chunk source, section, and snippet are persisted alongside the message. This means RAG citations survive page refreshes and appear correctly in the history view.
- **`last_summarized_id`** tracks the highest message ID that has been included in a summary. When fetching recent messages for context, only messages with `id > last_summarized_id` are included — older messages are represented by their summary instead. This keeps the context window bounded regardless of conversation length.

---

## 5. The `ChatService` — Full Conversation Orchestration

`ChatService` is the central class in Mozhi. Its `chat()` method assembles the complete prompt, calls the LLM, and manages all side effects (memory update, event writing, auto-notes extraction).

### 5.1 Initialisation

`ChatService` is constructed with all its dependencies injected:
- `client: ChatClient` — the LLM HTTP client
- `store: SessionStore` — the message persistence backend
- `field_store: FieldStore` — access to farm data for context assembly
- `rag_retriever: RAGRetriever | None` — the vector search engine
- `rag_router: QueryRouter | None` — the retrieval routing classifier
- `keyword_extractor: KeywordExtractor | None` — the keyword extraction LLM

This is assembled per-request in `deps.py`'s `chat_service_for_user()`, which reuses the startup-preloaded RAG singletons from `app.state`.

### 5.2 The `chat()` Method — Step by Step

```
chat(message, session_id, field_id, crop_id)
```

**Step 1 — Session resolution:**
If `session_id` is None, a new session is created. If `field_id` or `crop_id` are provided, they are bound to the session via `store.set_session_context()`. If not provided, the previously bound context is retrieved — this means the user doesn't need to re-specify their crop on every message.

**Step 2 — History fetch:**
```python
history = self.store.fetch_messages(session, limit=self.max_history)
```
Only messages after `last_summarized_id` are fetched (unsummarised recent messages). The default window is 12 messages (6 interaction pairs).

**Step 3 — System prompt construction:**
The prompt is assembled as a sequence of system messages:

```python
messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},  # NAVA persona + rules
]
```

The system prompt establishes NAVA's persona, prohibits non-agricultural responses, and instructs it not to emit placeholders or hallucinate missing context.

**Step 4 — Farm context injection:**
```python
ctx = self._build_context_message(field_id, crop_id)
if ctx:
    messages.append({"role": "system", "content": ctx})
```

For crop-level chat, `_build_context_message()` calls `field_store.get_rich_crop_context(crop_id)`, which generates a structured multi-section string containing:
- Field metadata (location, area, soil type)
- All sibling crops with their latest health summaries
- Current crop details (variety, stage, season, notes)
- Priority rules (disease detection > VNIR stress monitoring)
- Full per-plant disease detection history (up to 5 entries, high priority)
- Full per-plant VNIR monitoring history (up to 5 entries, lower priority)

This context is injected as a system message that the LLM is instructed to use silently — it should answer as if it already knows this information, not announce "according to your farm record..."

**Step 5 — Memory injection:**
```python
summary = self._summary_context(session)
if summary:
    messages.append({"role": "system", "content": summary})
```

The summary context is fetched from `chat_summaries`: up to 1 level-2 (long-term) rollup and up to 2 level-1 (recent) summaries. These are injected as a system message with instructions not to parrot the bullet format — the LLM should speak conversationally.

**Step 6 — RAG routing and retrieval:**
```python
if self.rag_router and self.rag_retriever and crop_id is not None:
    if self.rag_router.should_retrieve(message, last_assistant_reply):
        # Build enriched query
        retrieval_query = self._build_retrieval_query(message, crop_name, crop_id)
        # Extract keywords
        llm_keywords = self.keyword_extractor.extract(retrieval_query)
        # Retrieve chunks
        chunks = self.rag_retriever.query(retrieval_query, crop=crop_name, llm_keywords=llm_keywords)
        if chunks:
            # Build RAG block and inject as system message
            rag_block = "AGRONOMIC REFERENCE — VERIFIED SOURCE MATERIAL\n..."
            messages.append({"role": "system", "content": rag_block})
```

The RAG block is explicitly framed as authoritative: *"This information is factually reliable. Use it confidently and directly to ground your answer — do not hedge, qualify, or hold back information from it."* This prevents the LLM from being overly cautious about information from its own retrieved sources.

**Step 7 — History + user message append:**
```python
messages.extend(history)
messages.append({"role": "user", "content": message})
```

**Step 8 — LLM call + persistence:**
```python
reply, error = self.client.send(messages)
self.store.append_message(session, "user", message)
if reply:
    self.store.append_message(session, "assistant", reply, metadata=rag_meta)
```

RAG metadata (chunk count, source/section/snippet for each chunk) is persisted alongside the assistant message.

**Step 9 — Return `ChatResult`:**
```python
return ChatResult(
    session_id=session,
    reply=reply,
    rag_used=rag_used,
    rag_chunk_count=rag_chunk_count,
    rag_chunks=rag_chunks,  # sent to frontend for the RAG carousel UI
)
```

### 5.3 Retrieval Query Enrichment

A raw user message like "how do I treat this?" is ambiguous for semantic search. The `_build_retrieval_query()` method enriches it:

```python
parts = [f"Crop: {crop_name}."]
# Add latest disease detection result if any
if latest_diag and label not in ("healthy", ""):
    parts.append(f"Detected condition: {label}.")
# Add latest VNIR status if stressed
if latest_vnir and vstatus not in ("healthy", ""):
    parts.append(f"Stress monitoring status: {vstatus}.")
parts.append(message)
# Result: "Crop: banana. Detected condition: banana_black_sigatoka. How do I treat this?"
```

This enriched query is what gets embedded and used for ChromaDB similarity search — anchoring vague queries to the actual agronomic context.

---

## 6. Hierarchical Memory System

The hierarchical memory system is one of Mozhi's most sophisticated features. It solves the fundamental tension between LLM context window limits and long conversations.

### 6.1 Level-1 Summarisation

When the number of unsummarised messages exceeds `summary_batch` (default: 12), `_summarize_if_needed()` fires:

```python
batch = self.store.fetch_messages_with_ids(session, after_id=last_id, limit=summary_batch)
summary, _ = self.client.send(
    self._build_summary_prompt(batch),
    model_override=self.summary_model,    # small model
    temperature_override=0.2,
    max_new_tokens_override=200,
)
self.store.add_summary(session, level=1, content=summary)
self.store.set_last_summarized_id(session, max_id_in_batch)
```

The summary prompt asks the small LLM to produce 4–8 bullet points in the format `User: ... | NAVA: ...` — capturing both sides of the interaction. This preserves the conversational structure, not just the NAVA replies.

### 6.2 Level-2 Rollup

When 5 or more level-1 summaries accumulate, the oldest 5 are rolled up into a single level-2 long-term memory:

```python
if self.store.count_summaries(session, level=1) >= self.summary_rollup:
    oldest = self.store.fetch_oldest_summaries(session, level=1, limit=5)
    rollup, _ = self.client.send(self._build_rollup_prompt([r[1] for r in oldest]))
    self.store.add_summary(session, level=2, content=rollup)
    self.store.delete_summaries([r[0] for r in oldest])  # remove the 5 old level-1s
```

This creates a two-tier memory:
- **Level-1 (recent):** 4–8 bullets covering the last ~12 exchanges
- **Level-2 (long-term):** a compressed rollup of many level-1 summaries, representing the deep history

When injected into the prompt, both levels appear as a system message. The LLM sees the broad long-term context and the recent detail simultaneously, without the context window being overwhelmed by the raw message history.

### 6.3 Auto-Notes Extraction

After each new level-1 summary is generated, Mozhi scans it for concrete agronomic actions or decisions made by the farmer:

```python
def _extract_crop_notes_from_summary(self, summary, crop_id):
    prompt = [
        {"role": "system", "content": _NOTES_EXTRACT_SYSTEM},
        # "Scan the summary and extract ONLY concrete, specific actions or decisions the farmer has taken..."
        {"role": "user", "content": summary},
    ]
    reply, _ = self.client.send(prompt, model_override=self.summary_model, temperature_override=0.0, max_new_tokens_override=80)
```

If the LLM extracts actions (e.g., "Applied Carbendazim fungicide", "Removed diseased plants"), they are appended to the crop's `notes` field in the `FieldStore` under a `--- NAVA Auto-notes ---` separator, timestamped. This transforms a chat conversation into a permanent, structured farm record — the farmer's actions become part of the crop context that is injected into future conversations.

---

## 7. Module Interactions

Mozhi has three key dependencies:

| Dependency | How it's used |
|-----------|--------------|
| `FieldStore` | Fetches rich crop context (field, all crops, plant scan history) for context injection |
| `RAGRetriever` + `QueryRouter` | Routes and retrieves relevant knowledge chunks from Yukthi's ChromaDB store |
| `SessionStore` | Persists all messages, summaries, RAG metadata, and session-to-crop context bindings |

These are injected at construction time via `ChatService.from_settings_with_store()`, called from `deps.py`'s `chat_service_for_user()`. The RAG singletons are passed directly from `app.state` (preloaded at startup) — Mozhi never creates them itself in the normal server path.

---

## 8. The LLM Prompt Architecture

Every chat request assembles a prompt with this structure (ordered):

```
[system] NAVA persona + rules
[system] Farm/crop context (field, sibling crops, plant history, priority rules)
[system] Memory: Long-term rollup summary (if exists)
[system] Memory: Recent level-1 summaries (up to 2)
[system] RAG reference material (VERIFIED SOURCE MATERIAL block) — only if retrieved
[user]   message 1
[assistant] reply 1
[user]   message 2
[assistant] reply 2
... (up to 12 most recent unsummarised messages)
[user]   CURRENT MESSAGE
```

This structure ensures that:
1. The NAVA persona is always the first thing the LLM sees
2. Farm context grounds all subsequent responses
3. Memory prevents repetition across a long conversation
4. RAG material is marked as authoritative, encouraging confident use
5. Recent history provides conversational coherence
6. The current message is always last (standard LLM convention)
