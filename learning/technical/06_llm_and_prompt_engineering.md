# LLM and Prompt Engineering

> **Subfolder:** `technical/`
> **Cross-references:** [05_rag_and_knowledge_grounding.md](05_rag_and_knowledge_grounding.md) | [07_hierarchical_memory.md](07_hierarchical_memory.md) | [code/08_mozhi_client_and_service.md](../code/08_mozhi_client_and_service.md)

---

## The Two-Model Strategy

NAVA uses two different LLMs with very different characteristics. Understanding why requires understanding what each model is asked to do.

### The Primary Model (Meta-Llama-3-70B-Instruct)

The 70 billion parameter Llama-3 model handles all user-facing responses: answering questions about plant diseases, giving treatment advice, explaining observations. This requires:
- High-quality reasoning and synthesis
- Nuanced understanding of the farm context
- Natural, helpful writing style
- Accurate use of retrieved knowledge material

The 70B model excels at all of these. It is also slower (1–3 seconds per response on Hugging Face's hosted API) and more expensive per token.

### The Summary Model (Meta-Llama-3.1-8B-Instruct)

The 8 billion parameter Llama-3.1 model handles all internal tasks:
- **Conversation summarisation** (L1 and L2) — compressing message history
- **Auto-notes extraction** — identifying concrete actions from summaries
- **RAG routing** — binary RETRIEVE/SKIP decision
- **Keyword extraction** — 3 agronomic terms from the enriched query

These tasks are structurally simpler than open-ended chat:
- Summarisation is a well-defined reduction task
- RAG routing is binary classification
- Keyword extraction produces a JSON array of 3 strings

The 8B model completes these tasks in 100–300ms and at much lower cost. Using the 70B model for summarisation would be expensive, slow, and unnecessary.

**The critical insight:** using two models of different sizes for different task types is a standard cost-quality optimisation in LLM system design. Small models for housekeeping, large models for quality-critical outputs.

---

## The Hugging Face Router API

NAVA calls models hosted on Hugging Face's Router API, which provides an OpenAI-compatible `/v1/chat/completions` endpoint. This means the API call format is identical to OpenAI's API — a JSON body with a `messages` array, `model` identifier, `temperature`, `max_tokens`, and `stream=False`.

**Why Hugging Face over OpenAI?**
- Access to Llama-3 70B — a high-quality open-weight model with competitive chat performance
- Per-token pricing rather than subscription
- Flexibility to switch models without changing the API call format (just change the model string)

**Why not a locally hosted model?**
A 70B model requires approximately 35–70 GB of GPU VRAM for efficient inference. NAVA's deployment target (CPU server, Raspberry Pi-class hardware) cannot host a 70B model. The Hugging Face Router API provides GPU-hosted inference as a service.

---

## Context Window Assembly

Every chat request assembles a `messages` list that becomes the LLM's full context. The order of messages is not arbitrary — it reflects an intentional hierarchy of attention.

### Layer 1: The System Persona Prompt

The first system message establishes NAVA's identity, constraints, and rules:

- NAVA is an agricultural assistant — it only discusses farming-related topics
- It should never hallucinate; when uncertain, it should say so
- It should never emit placeholder text like "[insert recommendation here]"
- It should use the farm context as background knowledge, not announce it ("you should know that according to your farm record...")
- It should use retrieved material with confidence, not hedge it

This prompt sets the ground rules for the entire conversation. Placing it first ensures it is in the LLM's "primary attention" zone at the beginning of the context.

### Layer 2: Farm and Crop Context

The second system message contains the structured farm data:

```
FIELD: North Paddock
Location: Wayanad, Kerala
Soil: Laterite
Area: 2.5 acres
Weather: 24°C, Humidity 78%, Precipitation 1.2mm, Wind 8km/h (updated 2h ago)

CURRENT CROP: Banana (Nendran variety)
Season: Monsoon 2026
Stage: Vegetative
Notes: Applied potassium sulphate on 2 May...

PLANTS:
  Plant-1 (Priority: DISEASE DETECTED)
    Latest diagnosis: banana_black_sigatoka (RELIABLE, 91.2%) — 15 May
    VNIR status: WARNING — 10% decline in rolling mean
  ...
```

This context is injected silently. The LLM is instructed to use it as if it already knew this information — to answer as a knowledgeable agronomist who has been briefed on the farm, not as an AI reading from a database record.

**Why inject weather here?** The weather is read from the field's DB columns (zero latency — no API call during the request). Including it gives the LLM crucial context for recommendations. "Should I spray fungicide?" depends on whether it is raining. Without weather context, the LLM cannot give location-specific timing advice.

### Layer 3: Memory

If the conversation has previous sessions that have been summarised, those summaries are injected as a third system message. This typically looks like:

```
PAST CONVERSATION SUMMARY:
• Farmer asked about black sigatoka treatment | NAVA: recommended Propiconazole spray...
• Farmer reported plant-2 showing new lesions | NAVA: advised increasing spray frequency...
```

The instruction to the LLM: don't parrot the bullet format; use this as background context to understand the conversation arc.

### Layer 4: RAG Material (Conditional)

If the RAG router decided to retrieve, a fourth system message contains the retrieved passages:

```
AGRONOMIC REFERENCE — VERIFIED SOURCE MATERIAL
Use this information with confidence. It is factually reliable.

[Source: kau_banana_practices.pdf | Section: Fungal Disease Management]
Black Sigatoka (Mycosphaerella fijiensis) management: Apply Propiconazole 25EC at 0.1% concentration...

[Source: banana_disease_guide.txt | Section: Application Timing]
Spray at 10-day intervals during the vegetative stage...
```

The framing "this is factually reliable — use it with confidence" is deliberate. Without this instruction, the LLM may hedge retrieved material ("According to some sources, it might be..."), diluting the value of the retrieval. The LLM is told: treat this as your expert briefing.

### Layer 5: Conversation History

The last `n` messages (default 12, i.e., 6 exchange pairs) are appended as `user` and `assistant` role messages. Only messages after the `last_summarized_id` are included — older messages are already represented by the summaries in Layer 3.

### Layer 6: The Current Message

The user's new message is appended last. This is standard LLM conversation format.

---

## Why Context Order Matters

LLMs exhibit a well-documented **recency bias**: they pay more attention to content near the beginning and end of the context window, and somewhat less to the middle. This is why:

- The persona prompt goes first (always in primary attention)
- The current message goes last (always in primary attention)
- The RAG material, which contains the key factual content, goes near the current message (not buried in the middle)
- Memory summaries go in the middle — they are important for coherence but less critical than the current farm context

---

## Temperature and Max Tokens

NAVA uses different temperature settings for different tasks:

| Task | Temperature | Why |
|------|-------------|-----|
| User-facing chat | 0.7 | Some creativity; natural, non-robotic responses |
| Summarisation | 0.2 | Low creativity; deterministic compression |
| RAG routing | 0.0 | Fully deterministic; this is a classification task |
| Keyword extraction | 0.0 | Fully deterministic; the keywords should be the same on re-runs |
| Auto-notes extraction | 0.0 | Fully deterministic; don't invent actions that weren't mentioned |

Max tokens are capped per task:
- Summaries: 200 tokens (a concise set of bullets, not an essay)
- Keyword extraction: 40 tokens (a JSON array of 3 short strings)
- RAG routing: 5 tokens ("RETRIEVE" or "SKIP")
- User chat: model default (typically 2000–4096 tokens)

These caps prevent runaway generation costs and ensure the small model doesn't produce verbose output for housekeeping tasks.

---

## Why Llama-3 Over Other Models?

At the time of NAVA's implementation, Llama-3 70B represented the highest-quality open-weight model available on Hugging Face's hosted API. Its agricultural domain knowledge — while imperfect — was sufficient as a reasoning backbone when paired with RAG grounding. The model's instruction-following capability (critically important for strict persona and citation constraints) was strong.

The architecture is flexible: changing the model string in the configuration file switches the primary model without code changes. As better models become available (Llama-3.1, 3.2, Mistral, Qwen), NAVA can adopt them with a one-line configuration change.
