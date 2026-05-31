# Mozhi: `client.py` and `service.py`

> **Subfolder:** `code/`
> **Cross-references:** [technical/06_llm_and_prompt_engineering.md](../technical/06_llm_and_prompt_engineering.md) | [technical/07_hierarchical_memory.md](../technical/07_hierarchical_memory.md) | [technical/05_rag_and_knowledge_grounding.md](../technical/05_rag_and_knowledge_grounding.md) | [09_mozhi_session_store.md](09_mozhi_session_store.md)

**Source files:**
- [`client.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mozhi/chat/client.py)
- [`service.py`](file:///Users/dhanus/Desktop/nava/NAVA-AG/nava_core/mozhi/chat/service.py)

---

## `client.py` — The LLM API Client

### `ChatConfig` and `ChatClient`

```python
@dataclass(frozen=True)
class ChatConfig:
    model: str
    url: str
    api_key: str
    timeout: int
    temperature: float
    max_new_tokens: int
```

`frozen=True` makes `ChatConfig` immutable after construction — it's a value object representing a static configuration snapshot. Using a dataclass rather than passing individual parameters through `ChatClient.__init__` makes the configuration self-documenting and easy to inspect in debugging.

**`ChatClient.send()` — The Core Method**

```python
def send(
    self,
    messages: list[dict],
    model_override: Optional[str] = None,
    temperature_override: Optional[float] = None,
    max_new_tokens_override: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    payload = {
        "model": model_override or self.config.model,
        "messages": messages,
        "temperature": temperature_override if temperature_override is not None else self.config.temperature,
        "max_new_tokens": max_new_tokens_override or self.config.max_new_tokens,
    }
    headers = {"Authorization": f"Bearer {self.config.api_key}"}

    resp = requests.post(self.config.url, headers=headers, json=payload, timeout=self.config.timeout)
    content = resp.json()["choices"][0]["message"]["content"]
    return content, None
```

The method returns a `tuple[reply, error]` instead of raising exceptions. This is an explicit "Either" type in Python: callers always check both fields (`if error: ...` / `if reply: ...`). This pattern prevents silent failures: if the API call fails, the caller receives `(None, "error_message")` and can handle it gracefully.

**Override parameters** (`model_override`, `temperature_override`, `max_new_tokens_override`) allow the same `ChatClient` instance to be used for different tasks with different settings. The summarisation calls use a different model and temperature than the main chat calls, but both go through the same `client.send()`.

**`requests.post()` with `timeout`:** Synchronous HTTP call with a hard timeout (configurable, default 120 seconds). If the LLM API is slow or unavailable, the call fails with a `requests.exceptions.Timeout` — caught by the `except Exception as exc` block — rather than blocking indefinitely.

---

## `service.py` — The Chat Orchestrator

`ChatService` is the largest file in the codebase (575 lines). It contains all the logic for context assembly, RAG integration, summarisation, and the main chat loop.

### `DEFAULT_SYSTEM_PROMPT`

```python
DEFAULT_SYSTEM_PROMPT = (
    "You are NAVA, a digital agronomist. Use the structured context provided in system messages "
    "(Field, Crop, Context, Events, Memory) to ground your answers. Do not mention missing fields "
    "or emit placeholders like 'null'. If context is missing, ask concise follow-up questions. "
    "Only answer agricultural questions; politely refuse non-agricultural requests. "
    "If asked for regulated chemical dosages or exact pesticide quantities, advise consulting local "
    "guidelines or an agronomist. Keep responses short to medium length, practical, and clear."
)
```

Key constraints embedded in the system prompt:
- **Don't mention missing data:** The context injection includes many optional fields (weather, VNIR status, recent scans). If any are absent, the LLM should not say "I notice you haven't provided weather data." Just answer without them.
- **No placeholders:** Without this instruction, some LLMs produce responses with template-style placeholders ("Apply [fungicide name] at [recommended rate]"). The instruction suppresses this.
- **Agricultural-only:** Polite refusal for non-agricultural questions preserves NAVA's identity and prevents abuse.
- **Chemical dosage disclaimer:** Specific pesticide quantities are regulated and jurisdiction-specific. The instruction ensures NAVA always directs users to local authorities for exact dosages rather than risking incorrect advice.

### `_build_context_message()` — Context Assembly

This method has two code paths based on whether a `crop_id` or only a `field_id` is available.

**Crop-level context (most informative):**
```python
if crop_id is not None:
    rich = self.field_store.get_rich_crop_context(crop_id)
    # Append weather from DB
    field_rec = self.field_store.get_field(crop_rec["field_id"])
    if field_rec and field_rec.get("weather_updated_at") is not None:
        rich += "\n\n=== CURRENT WEATHER CONDITIONS ===\n..."
    return "CROP CONTEXT (use silently to ground answers):\n" + rich
```

`get_rich_crop_context(crop_id)` returns a comprehensive text block with plant histories, latest diagnoses, VNIR statuses, and crop metadata. The instruction "use silently" tells the LLM to incorporate this knowledge without announcing it ("According to your farm records, your banana plants have...").

**Weather injection — zero latency:**
Weather is read directly from `field_rec["weather_temp"]` etc. — DB columns updated at login. No network call during the chat request. This is the key design decision explained in [technical/09_weather_and_geocoding.md](../technical/09_weather_and_geocoding.md).

**Field-level context (minimal):**
Used when there's no specific crop selected (e.g., general farm-level questions). Includes field metadata (name, location, soil type, area) plus weather. Less detailed than crop-level but still grounds the LLM in the specific farm context.

### `_extract_crop_notes_from_summary()` — Auto-notes Extraction

```python
_NOTES_EXTRACT_SYSTEM = (
    "Scan the following chat summary and extract ONLY concrete, specific actions "
    "or decisions the farmer has taken or plans to take. "
    "Output ONLY the relevant facts, one per line, starting with a dash (- ). "
    "If no concrete agronomic action or decision is present, output exactly: NONE"
)
```

The extraction prompt is tightly constrained:
1. Only user actions (not NAVA's advice, unless user confirmed they followed it)
2. Only concrete, specific facts (not general observations)
3. "NONE" if nothing actionable is present (prevents the auto-notes from filling up with non-actions)

The auto-notes are appended below the `--- NAVA Auto-notes ---` separator in the crop's notes field. The separator allows future code to distinguish auto-generated notes from manually written ones.

```python
now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
new_entries = f"[{now_str}]\n" + "\n".join(f"- {l}" for l in lines)
```

Each extraction is timestamped (`[2026-05-15 14:30]`) so the farmer can see when NAVA extracted these notes.

### `_summarize_if_needed()` — The Summarisation Trigger

```python
def _summarize_if_needed(self, session_id: str) -> None:
    last_id = self.store.get_last_summarized_id(session_id)
    pending = self.store.count_messages_after(session_id, last_id)
    if pending < self.summary_batch:  # default: 12
        return

    batch = self.store.fetch_messages_with_ids(session_id, after_id=last_id, limit=self.summary_batch)
    summary, error = self.client.send(
        self._build_summary_prompt(batch),
        model_override=self.summary_model,       # 8B model
        temperature_override=self.summary_temperature,  # 0.2
        max_new_tokens_override=self.summary_max_new_tokens,  # 200
    )
    
    self.store.add_summary(session_id, level=1, content=summary)
    self.store.set_last_summarized_id(session_id, max_id)

    # Auto-notes extraction (best-effort)
    self._extract_crop_notes_from_summary(summary, crop_id)

    # L2 rollup if 5+ L1 summaries exist
    if self.store.count_summaries(session_id, level=1) >= self.summary_rollup:
        oldest = self.store.fetch_oldest_summaries(session_id, level=1, limit=self.summary_rollup)
        rollup, _ = self.client.send(self._build_rollup_prompt(...))
        self.store.add_summary(session_id, level=2, content=rollup)
        self.store.delete_summaries([r[0] for r in oldest])
```

The summarisation trigger, L1→L2 rollup, and auto-notes extraction all happen in sequence, called as a background task after the chat response is sent.

### `_build_retrieval_query()` — Context-Enriched RAG Query

```python
def _build_retrieval_query(self, message, crop_name, crop_id):
    parts = [f"Crop: {crop_name}."]
    
    # Latest disease detection
    diag_events = self.field_store.list_events(crop_id=crop_id, event_type="diagnose", limit=1)
    if diag_events:
        label = diag_events[0]["payload"].get("class_label", "")
        if label not in ("healthy", "no scan", ""):
            parts.append(f"Detected condition: {label}.")

    # VNIR stress status
    vnir_events = self.field_store.list_events(crop_id=crop_id, event_type="vnir", limit=1)
    if vnir_events:
        vstatus = vnir_events[0]["payload"].get("status", "")
        if vstatus not in ("healthy", "no scan", ""):
            parts.append(f"Stress monitoring status: {vstatus}.")

    parts.append(message)
    return " ".join(parts)
```

The enriched query includes:
- The crop name (anchors the ChromaDB embedding to the correct crop collection)
- The latest disease label (resolves anaphora like "how do I treat this?" — "this" becomes `banana_black_sigatoka`)
- The VNIR stress status (provides additional context for stress-related queries)

This enrichment is what makes RAG queries far more precise than sending the raw user message.

### `chat()` — The Main Method

The `chat()` method assembles the full message list in order:
1. System persona prompt
2. Farm/crop context (from `_build_context_message()`)
3. Memory summaries (from `_summary_context()`)
4. RAG material (if `rag_router.should_retrieve()` returns True)
5. History messages (last 12 exchanges)
6. The current user message

Then calls `client.send()` and persists both the user message and the assistant reply (with RAG metadata if used) to the `SessionStore`.

**RAG metadata persistence:**
```python
rag_meta = {"rag_used": rag_used, "rag_chunk_count": rag_chunk_count, "rag_chunks": rag_chunks} if rag_used else None
self.store.append_message(session, "assistant", reply, metadata=rag_meta)
```

RAG metadata is stored alongside the assistant message in the session store. When the frontend reloads conversation history via `GET /api/chat/history`, it receives the RAG metadata for each message — allowing previously RAG-grounded responses to show their source citations even after a page refresh.
