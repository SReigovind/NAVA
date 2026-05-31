# RAG and Knowledge Grounding

> **Subfolder:** `technical/`
> **Cross-references:** [non_technical/02_research_foundation.md](../non_technical/02_research_foundation.md) | [06_llm_and_prompt_engineering.md](06_llm_and_prompt_engineering.md) | [code/10_yukthi_pipeline_and_store.md](../code/10_yukthi_pipeline_and_store.md) | [code/11_yukthi_retriever_router_keywords.md](../code/11_yukthi_retriever_router_keywords.md)

---

## The Hallucination Problem in Agricultural AI

Large language models are trained on internet-scale text. That training gives them broad, impressive knowledge across many domains — including agriculture. But "broad" does not mean "accurate." For any narrow, specialised question — the correct pre-harvest interval for a specific fungicide on banana in Kerala, the recommended variety for paddy cultivation in laterite soil — the LLM may produce a plausible but incorrect answer.

The failure mode is not random noise. LLMs tend to produce coherent, grammatically correct, confidently stated errors. A farmer who doesn't know the right answer cannot distinguish the correct response from the incorrect one. This is the hallucination problem.

For a medical or agricultural advisory system, hallucination is not an academic concern — it causes measurable harm.

---

## What RAG Does

Retrieval-Augmented Generation inserts verified information into the LLM's context before generation. Instead of asking the LLM to recall factual information from its weights (where it may be wrong), you give it the right information directly and ask it to reason about it.

The process:
1. A user asks a question ("How do I treat rice blast?")
2. Before calling the LLM, retrieve the most relevant passages from the verified knowledge base
3. Inject those passages into the prompt as a "VERIFIED SOURCE MATERIAL" block
4. The LLM reasons over the retrieved material to compose its answer

The LLM's role shifts from "factual retrieval" (which it is unreliable at) to "synthesis and explanation" (which it excels at). The facts come from the knowledge base; the explanation comes from the LLM.

---

## The Knowledge Base Structure

NAVA's knowledge base contains agricultural extension documents: Kerala Agricultural University's Package of Practices (PoP) for various crops, disease management guides, pest management guides, and crop-specific cultivation guides. These are stored as PDF and TXT files in `ragsource/`, organised by crop.

Each document is processed by the ingestion pipeline into chunks — passages of a few sentences to a few paragraphs. The chunks are the units that get retrieved. Too-large chunks dilute the relevance; too-small chunks lose context. NAVA's chunker splits on paragraph boundaries and detects section headers, aiming for self-contained passages.

---

## Embedding: From Text to Numbers

Retrieval requires a way to measure semantic similarity between a question and a passage. Pure keyword matching (does the passage contain the words in the question?) is brittle: it misses paraphrases, synonyms, and implied concepts.

**Dense embeddings** solve this. A trained encoder model converts a piece of text into a fixed-size vector (384 dimensions in NAVA's case). The encoder is trained so that semantically similar texts produce geometrically similar vectors — vectors that are close to each other in 384-dimensional space.

NAVA uses `BAAI/bge-small-en-v1.5`:
- 33M parameters — large enough to capture semantic nuance, small enough to run locally on CPU
- 384-dimensional output vectors
- Specifically trained for semantic retrieval tasks (not generation)
- Runs entirely locally via `sentence-transformers` — no API call, no usage limit

The same encoder is used at ingestion time (to embed the document chunks) and at retrieval time (to embed the query). This consistency is critical: the similarity measure is only meaningful when both vectors were produced by the same model.

---

## ChromaDB: The Vector Store

ChromaDB is an open-source vector database that stores embeddings alongside their original text and metadata, and provides efficient nearest-neighbour search.

**Why ChromaDB over alternatives (Pinecone, Weaviate, Qdrant)?**
- Runs locally as a `PersistentClient` — no cloud account, no API key, no per-query cost
- Data persists to disk (HNSW index files + SQLite metadata)
- Simple Python API that doesn't require a separate server process
- Actively maintained with a clean embedding-focused API

**Collection naming:** Each crop gets its own collection (`nava_banana`, `nava_rice`, etc.). This prevents cross-crop contamination — a rice query cannot retrieve banana passages. It also makes targeted re-ingestion simple: `--force --crop banana` deletes and rebuilds only the banana collection.

**Deterministic IDs:** Chunk IDs are constructed as `{source_filename}_{chunk_index}`. This makes the upsert operation idempotent — re-running ingestion with the same source files produces the same ChromaDB state, with no duplicates.

---

## Hybrid Retrieval: Why Pure Semantic Search Isn't Enough

**Pure semantic search** retrieves the 5 most semantically similar chunks to the query embedding. This works well for general questions ("how do I improve soil health?") but fails for specific agronomic terminology.

Consider the query: "How do I treat banana black sigatoka?" The query embedding is close to passages about banana diseases in general, but may rank a passage specifically about Sigatoka below passages about other banana diseases that are semantically similar to the query for the wrong reasons.

**Keyword-filtered search** retrieves only chunks whose text contains a specific keyword, then ranks them by semantic similarity. The keyword "sigatoka" directly filters the collection to passages that mention that disease.

NAVA combines both:

**Stage 1: Semantic search** retrieves 5 candidates by pure cosine similarity.

**Stage 2: LLM keyword extraction** asks the small model (Llama-3.1-8B) to extract exactly 3 precise agronomic terms from the enriched query. Why LLM-based extraction rather than heuristic keyword extraction? Because agronomic terms like "Sigatoka," "Moko disease," or "Glufosinate ammonium" would be stopped by standard stopword filters, stemmed incorrectly, or split at wrong boundaries. The LLM understands domain vocabulary.

**Stage 3: Keyword-filtered semantic search** runs one ChromaDB query per keyword, retrieving a small number of candidates that both contain the keyword and are semantically close to the query.

**Stage 4: Deduplication** merges all candidates, removing duplicates keyed on the first 120 characters of text.

**Stage 5: Reranking** scores each candidate as `0.7 × (1 − cosine_distance) + 0.3 × keyword_overlap_ratio`. The cosine similarity dominates (topical relevance), with keyword overlap providing a precision boost for domain-specific matches.

**Stage 6: Top-4** — the top-4 reranked chunks are injected into the LLM prompt.

---

## The Query Router: Don't Retrieve for Every Message

Not every chat message benefits from retrieval. "Thanks, that's helpful" doesn't need a knowledge lookup. "How's the weather?" doesn't need agriculture documents. Retrieving for every message adds ~1 second of latency (keyword extraction + ChromaDB query) and can dilute the prompt with irrelevant material.

`QueryRouter.should_retrieve()` asks the small LLM (Llama-3.1-8B, temperature 0.0) to make a binary RETRIEVE/SKIP decision based on the user's message and the previous assistant reply.

The router is instructed to RETRIEVE if the message asks for:
- Disease treatment, pesticides, fungicides
- Fertiliser recommendations
- Cultivation practices, variety selection
- Irrigation, crop management advice
- Pest identification or management

And to SKIP for:
- Greetings, acknowledgements
- Questions already answered in the previous reply
- Vague follow-ups that don't require factual knowledge

The small model is used here (not the 70B model) because this is a binary classification task. It can be done correctly with far fewer parameters, and the latency difference is significant: the 8B model responds in ~200ms, versus ~1–2 seconds for the 70B model.

---

## Fallback: Heuristic Keywords

If LLM keyword extraction fails (the model is unavailable, times out, or produces invalid JSON), `_heuristic_fallback_terms()` extracts content words from the enriched query: tokens of 4+ characters that are not in a standard stopword list. These heuristic keywords are less precise than LLM-extracted terms, but they ensure the hybrid retrieval continues to function degradedly rather than failing completely.

---

## Source Attribution in the UI

Every assistant message that used RAG is tagged `rag_used: True` in the API response. The specific chunks (source filename, section header, text snippet) are included in the response as `rag_chunks`. The frontend renders a collapsible carousel below the assistant message showing each chunk's source and content.

This is not just a transparency feature — it is an accountability mechanism. If the advice given in a chat message is incorrect, the farmer (or a reviewing extension officer) can look at the source passage that grounded it and identify whether the passage itself was wrong, or whether the LLM deviated from it.
