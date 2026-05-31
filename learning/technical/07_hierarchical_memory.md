# Hierarchical Memory

> **Subfolder:** `technical/`
> **Cross-references:** [06_llm_and_prompt_engineering.md](06_llm_and_prompt_engineering.md) | [code/08_mozhi_client_and_service.md](../code/08_mozhi_client_and_service.md) | [code/09_mozhi_session_store.md](../code/09_mozhi_session_store.md)

---

## The Context Window Problem

Every LLM has a fixed context window — the maximum number of tokens it can process in a single forward pass. For Llama-3 70B on Hugging Face's API, this is typically 8,000–32,000 tokens. Each token is approximately 3–4 characters of text.

A typical chat message exchange is 100–300 tokens. Twelve such exchanges (the default history window NAVA uses) is 1,200–3,600 tokens. With the system prompt, farm context, RAG material, and summaries, a full NAVA prompt might consume 3,000–6,000 tokens before the conversation history.

This arithmetic reveals the problem: a long-running farming season conversation — dozens of sessions over six months — would overflow the context window within weeks if all messages were included directly. Something must be discarded or compressed.

---

## Why Simple Truncation Fails

The simplest solution is to truncate: keep only the last N messages and discard everything older. This works for short-term coherence but fails for agricultural applications:

- A treatment decision made in March may be directly relevant to a question asked in September
- A disease diagnosis from two months ago explains why the farmer made certain cultivation choices
- The "what we tried last time" context is exactly what the farmer wants NAVA to remember

Truncation silently discards this long-term context. The farmer experiences NAVA as forgetful.

---

## The Two-Level Summarisation Pyramid

NAVA's solution is hierarchical summarisation: compress old messages into progressively denser representations that preserve the key information while using far fewer tokens.

### Level 1: Recent Summaries

When the number of unsummarised messages exceeds 12 (the `summary_batch` threshold), the oldest batch of 12 messages is passed to the small LLM (Llama-3.1-8B) with instructions to produce a 4–8 bullet summary in the format:

```
• User: asked about sigatoka treatment | NAVA: recommended Propiconazole 25EC at 0.1%
• User: reported new lesions on plant-3 | NAVA: advised increasing spray frequency to weekly
```

The `User: ... | NAVA: ...` format is deliberate. It preserves both sides of the interaction — not just NAVA's responses, but the farmer's questions and context. A summary that only captures NAVA's outputs would lose the information about what the farmer observed and reported.

The summary is stored in `chat_summaries` with `level=1`. The `last_summarized_id` is updated to the highest message ID in the batch. Future history fetches exclude those summarised messages — they are represented by the summary instead.

**Cost:** One call to the 8B model (200ms, very cheap) per 12 exchanges. The trade: 12 full messages (1,200–3,600 tokens) compressed to one summary (100–150 tokens). A 10–24x token reduction.

### Level 2: Long-Term Rollup

When 5 or more Level-1 summaries have accumulated, the oldest 5 are compressed into a single Level-2 long-term memory. The 5 summaries (each 100–150 tokens = 500–750 tokens total) are passed to the 8B model with instructions to produce a compact narrative of the conversation arc — what the farm situation was, what was tried, what worked.

The 5 old L1 summaries are deleted after the L2 is created.

The L2 rollup typically covers 60 message pairs — the full arc of a significant farming season episode. It is injected into future prompts as the "Long-term memory" system message.

**The two-layer structure in the prompt:**
- L2 rollup (broad arc, months of context) → injected first among the memory layers
- Recent L1 summaries (recent detail, last few weeks) → injected after L2, closer to the current message
- Unsummarised messages (last few exchanges) → injected as actual message pairs

This gives the LLM access to the entire conversation arc at appropriate granularity: broad historical context from L2, recent summary from L1, detailed current context from the raw messages.

---

## Auto-Notes Extraction: Memory as Farm Record

After each new L1 summary is generated, NAVA runs a second pass with a different prompt: "Scan this conversation summary and extract ONLY concrete, specific actions or decisions the farmer has taken."

The small model (0.0 temperature — fully deterministic) produces a brief list like:

```
• Applied Propiconazole 25EC at 0.1% on 15 May
• Removed 3 diseased plants from plot B on 18 May
• Scheduled follow-up spray for 25 May
```

These extracted actions are appended to the crop's `notes` field in `FieldStore`, under a `--- NAVA Auto-notes ---` separator with a timestamp.

**Why write to crop notes?**
This transforms ephemeral chat conversation into permanent structured farm records. The auto-notes appear in:
1. The `OverviewPanel` (visible to the farmer in the UI)
2. The farm context block injected into every future chat session (as part of `get_rich_crop_context()`)

This means NAVA accumulates a structured treatment history automatically, without the farmer having to explicitly log it. The chat interface doubles as a farm diary.

**Why 0.0 temperature?**
Auto-notes extraction must not invent actions that weren't mentioned. At temperature 0.0, the model produces deterministic output and is far less likely to confabulate. The conservative instruction ("ONLY concrete, specific actions or decisions") further constrains the output.

---

## Session Management

Each conversation session is identified by a UUID hex string, generated client-side. Sessions are bound to a specific field-crop combination via `set_session_context()`. A farmer can have multiple sessions for the same crop (one per growing period, for example), each with independent message history and summaries.

Sessions are stored in `localStorage` in the frontend — the backend is stateless with respect to session creation. Any valid token holder can send messages with any session ID; the backend initialises state for unknown sessions on demand.

---

## What This Looks Like in Practice

A farmer who has used NAVA for 6 months and accumulated 200+ messages in a session:

- The last 12 messages are included as raw exchange pairs
- 2 recent L1 summaries (covering the previous ~24 exchanges) are included
- 1 L2 rollup (covering the earlier 160+ exchanges) is included

Total memory tokens in the prompt: approximately 350–600 tokens. Without summarisation, those 200 messages would require 20,000–60,000 tokens — exceeding the context window entirely.

The farmer experiences NAVA as remembering everything. NAVA is actually remembering compressed representations, but those representations capture the agronomically relevant information that matters for farming decisions.

---

## Limitations

**Information loss:** Summarisation is lossy. Specific numbers (exact dosages, exact dates) may not survive compression if the summariser judges them less important than the overall action. This is why the auto-notes extraction step explicitly tries to capture specific facts before they disappear into a L1 summary.

**Summariser quality:** The L1 summaries are generated by the 8B model. The quality of these summaries determines the quality of the long-term memory. If the 8B model produces poor summaries, the L2 rollup of those summaries will be equally poor. This is a known limitation; improving the summarisation prompt or using a better small model would improve memory quality.

**No cross-session memory:** Each session has its own memory. If a farmer starts a new session, they lose access to the previous session's L2 rollup within that session. The auto-notes written to `FieldStore.crops.notes` persist across sessions (they are in the farm context, not the session context), which partially mitigates this.
