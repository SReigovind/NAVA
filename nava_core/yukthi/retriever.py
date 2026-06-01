"""Query-time RAG retrieval from ChromaDB — LLM-keyword-guided hybrid search.

Pipeline (per query):
  1. [Caller] Router LLM decides RETRIEVE
  2. [Caller] Build enriched query (crop + disease + user message)
  3. [Caller] KeywordExtractor LLM call → 3 precise agronomic keywords
  4. [This module] Semantic search  — top 8 candidates from ChromaDB
  5. [This module] Keyword searches — for each LLM keyword, top 3 from
     ChromaDB where_document filter → ~9 keyword candidates (deduped to 8)
  6. Merge + deduplicate all candidates (up to 16 unique chunks)
  7. Rerank: combined score = 0.7 × semantic_similarity + 0.3 × keyword_overlap
  8. Return top 4

Why 8+8 → top 4:
  - 8 semantic ensures broad topical coverage
  - LLM-derived keywords (not heuristic stopword filtering) target specific
    disease names and agronomic terms, pulling in chunks that narrowly miss
    the semantic top-8 (e.g. two different Sigatoka disease entries)
  - Reranking to 4 (up from 3) gives the LLM richer reference material
    without overwhelming the context window
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .store import YukthiStore
from nava_core.shared.utils.logging import get_logger

log = get_logger("yukthi.retriever")

# Fallback heuristic stop-words used when LLM keyword extraction fails
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "my", "your", "our", "its", "this",
    "that", "these", "those", "what", "how", "why", "when", "where",
    "which", "who", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through",
    "i", "me", "we", "you", "it", "he", "she", "they", "crop", "banana",
    "tell", "give", "show", "want", "know", "please", "need", "help",
})


@dataclass
class RAGChunk:
    text: str
    source: str
    section: str
    score: float   # cosine distance — lower is more similar (0.0 = identical)


class RAGRetriever:
    """Hybrid semantic + LLM-keyword retriever backed by ChromaDB + sentence-transformers."""

    def __init__(
        self,
        store: YukthiStore,
        embed_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 4,
        distance_threshold: float = 0.45,
    ) -> None:
        self.store = store
        self.embed_model = embed_model
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self._encoder = None  # singleton, lazy-loaded

    @classmethod
    def from_settings(cls, store: Optional[YukthiStore] = None) -> "RAGRetriever":
        from nava_core.shared.config import get_settings
        s = get_settings()
        return cls(
            store=store or YukthiStore(s.yukthi_chroma_dir),
            embed_model=s.yukthi_embed_model,
            top_k=s.yukthi_top_k,
            distance_threshold=s.yukthi_distance_threshold,
        )

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info("Loading retrieval embedding model: %s", self.embed_model)
                self._encoder = SentenceTransformer(self.embed_model)
                log.info("Embedding model ready.")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers is required. Install it with: pip install sentence-transformers"
                )
        return self._encoder

    # ── Keyword scoring ─────────────────────────────────────────────────────

    def _keyword_score(self, document: str, search_terms: list[str]) -> float:
        """Keyword overlap ratio [0.0, 1.0] — how many search terms appear in document."""
        if not search_terms:
            return 0.0
        doc_lower = document.lower()
        hits = sum(1 for t in search_terms if t.lower() in doc_lower)
        return hits / len(search_terms)

    def _heuristic_fallback_terms(self, text: str) -> list[str]:
        """Fallback: extract content words when LLM keyword extraction fails."""
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        return list(dict.fromkeys(w for w in words if w not in _STOPWORDS))[:3]

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        message: str,
        crop: str,
        top_k: Optional[int] = None,
        llm_keywords: Optional[list[str]] = None,
    ) -> list[RAGChunk]:
        """Hybrid retrieval: 8 semantic + 8 keyword-guided candidates → rerank → top-k.

        Args:
            message:      The enriched retrieval query (crop + disease + user message).
            crop:         Crop name — selects the correct ChromaDB collection.
            top_k:        Override the default final chunk count (default 4).
            llm_keywords: Pre-extracted keywords from KeywordExtractor. If None or [],
                          falls back to heuristic term extraction.

        Returns:
            List of RAGChunk objects, reranked and capped at top_k.
            Returns [] if the crop collection doesn't exist.
        """
        k = top_k if top_k is not None else self.top_k
        crop_norm = crop.lower().strip()

        if not self.store.collection_exists(crop_norm):
            log.debug("Retriever: no collection for crop '%s' — returning empty.", crop_norm)
            return []

        encoder = self._get_encoder()
        embedding = encoder.encode(message, show_progress_bar=False).tolist()

        # Determine keyword list: prefer LLM-generated, fall back to heuristic
        search_keywords = (
            llm_keywords if llm_keywords
            else self._heuristic_fallback_terms(message)
        )
        source_label = "LLM" if llm_keywords else "heuristic"

        # ── 1. Semantic search: 10 candidates ─────────────────────────────────
        SEMANTIC_N = 10
        semantic_results = self.store.query(
            crop=crop_norm, embedding=embedding, n_results=SEMANTIC_N
        )

        # Accumulate candidates: (text, metadata, distance)
        candidates: list[tuple[str, dict, float]] = []
        seen: set[str] = set()

        def _add(doc: str, meta: dict, dist: float) -> None:
            key = doc[:120]  # dedup key = first 120 chars
            if key not in seen:
                seen.add(key)
                candidates.append((doc, meta, dist))

        n_semantic = 0
        if semantic_results:
            docs  = semantic_results.get("documents", [[]])[0]
            metas = semantic_results.get("metadatas", [[]])[0]
            dists = semantic_results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                _add(doc, meta, dist)
                n_semantic += 1

        # ── 2. Keyword-filtered semantic: up to 10 keyword candidates ─────────
        # Each of the LLM keywords → top results with where_document filter
        # That's ~10 raw keyword candidates → deduplicated
        import math
        KEYWORD_PER_TERM = math.ceil(10 / max(len(search_keywords), 1))
        n_keyword = 0
        for term in search_keywords:
            clean_term = term.replace("_", " ")
            kw_results = self.store.query_keyword_filtered(
                crop=crop_norm, embedding=embedding, term=clean_term,
                n_results=KEYWORD_PER_TERM,
            )
            if not kw_results:
                continue
            kw_docs  = kw_results.get("documents", [[]])[0]
            kw_metas = kw_results.get("metadatas", [[]])[0]
            kw_dists = kw_results.get("distances", [[]])[0]
            for doc, meta, dist in zip(kw_docs, kw_metas, kw_dists):
                _add(doc, meta, dist)
                n_keyword += 1

        log.info(
            "Retriever: %d semantic + %d keyword candidates | keywords=%s (%s) | crop='%s'",
            n_semantic, n_keyword, search_keywords, source_label, crop_norm,
        )

        # ── 3. Rerank by combined score ───────────────────────────────────────
        # Threshold: slightly relaxed for keyword candidates (they had a hard filter,
        # so a higher distance is acceptable if the keyword matches strongly)
        ranked: list[tuple[float, RAGChunk]] = []
        for doc, meta, dist in candidates:
            if dist > self.distance_threshold * 1.2:
                log.debug(
                    "Retriever: dropped '%s' dist=%.3f > threshold×1.2=%.3f",
                    meta.get("section", "?"), dist, self.distance_threshold * 1.2,
                )
                continue
            kw = self._keyword_score(doc, search_keywords)
            combined = 0.7 * max(0.0, 1.0 - dist) + 0.3 * kw
            log.info(
                "Retriever: ✓ '%s'  dist=%.3f  kw=%.2f  combined=%.3f",
                meta.get("section", "?"), dist, kw, combined,
            )
            ranked.append((combined, RAGChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                section=meta.get("section", "unknown"),
                score=dist,
            )))

        ranked.sort(key=lambda x: x[0], reverse=True)
        result = [chunk for _, chunk in ranked[:k]]

        log.info(
            "Retriever: returning %d final chunks (from %d candidates) for crop='%s'",
            len(result), len(candidates), crop_norm,
        )
        return result

    def warm_up(self) -> None:
        """Eagerly load the embedding model in the current thread.

        Called at server startup so the model is ready before the first request.
        """
        try:
            self._get_encoder()
            log.info("RAGRetriever embedding model pre-warmed.")
        except Exception as e:
            log.warning("RAGRetriever warm_up failed: %s", e)
