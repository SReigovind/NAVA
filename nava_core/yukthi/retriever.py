"""Query-time RAG retrieval from ChromaDB — hybrid semantic + keyword search.

Strategy (per query):
  1. Semantic search  — embed the enriched query, fetch top (k × 2) candidates
  2. Keyword-filtered semantic search — for each key term extracted from the
     query, re-query ChromaDB with a where_document filter so term-specific
     chunks that narrowly missed the top-k semantic cutoff are included
  3. Merge + deduplicate candidates
  4. Rerank by a combined score: 0.7 × semantic_similarity + 0.3 × keyword_overlap
  5. Return the top-k after reranking

This significantly improves recall for disease-specific queries (e.g. both
"Black Sigatoka" and "Yellow Sigatoka" chunks are retrieved when the user asks
about Sigatoka, even if one has a slightly lower cosine similarity).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .store import YukthiStore
from nava_core.shared.utils.logging import get_logger

log = get_logger("yukthi.retriever")

# Words to ignore when extracting keyword search terms
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
    """Hybrid semantic + keyword retriever backed by ChromaDB + sentence-transformers."""

    def __init__(
        self,
        store: YukthiStore,
        embed_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 3,
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

    # ── Term extraction & keyword scoring ───────────────────────────────────

    def _extract_terms(self, text: str) -> list[str]:
        """Extract meaningful content words from a text for keyword scoring."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in _STOPWORDS]

    def _keyword_score(self, document: str, query_terms: list[str]) -> float:
        """Keyword overlap ratio [0.0, 1.0] between document and query terms."""
        if not query_terms:
            return 0.0
        doc_lower = document.lower()
        hits = sum(1 for t in query_terms if t in doc_lower)
        return hits / len(query_terms)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(self, message: str, crop: str, top_k: Optional[int] = None) -> list[RAGChunk]:
        """Hybrid retrieval: semantic (k×2) + keyword-filtered, then rerank → top-k.

        Args:
            message: The (optionally enriched) retrieval query.
            crop:    Crop name — selects the correct ChromaDB collection.
            top_k:   Override the default top-k count.

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
        query_terms = self._extract_terms(message)

        # ── 1. Semantic search: fetch k×2 candidates ────────────────────────
        semantic_k = max(k * 2, 6)
        semantic_results = self.store.query(crop=crop_norm, embedding=embedding, n_results=semantic_k)

        # Collect candidates: (text, metadata, distance)
        candidates: list[tuple[str, dict, float]] = []
        seen: set[str] = set()

        def _add_candidate(doc: str, meta: dict, dist: float) -> None:
            key = doc[:120]  # dedup by first 120 chars
            if key not in seen:
                seen.add(key)
                candidates.append((doc, meta, dist))

        if semantic_results:
            docs = semantic_results.get("documents", [[]])[0]
            metas = semantic_results.get("metadatas", [[]])[0]
            dists = semantic_results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                _add_candidate(doc, meta, dist)

        # ── 2. Keyword-filtered semantic search: additional candidates ───────
        # Take the longest unique content terms (most likely to be disease names)
        content_terms = sorted(
            [t for t in set(query_terms) if len(t) >= 5],
            key=len, reverse=True
        )[:3]  # up to 3 keyword searches

        kw_total = 0
        for term in content_terms:
            kw_results = self.store.query_keyword_filtered(
                crop=crop_norm, embedding=embedding, term=term, n_results=k
            )
            if kw_results:
                kw_docs = kw_results.get("documents", [[]])[0]
                kw_metas = kw_results.get("metadatas", [[]])[0]
                kw_dists = kw_results.get("distances", [[]])[0]
                for doc, meta, dist in zip(kw_docs, kw_metas, kw_dists):
                    kw_total += 1
                    _add_candidate(doc, meta, dist)

        log.info(
            "Retriever: %d semantic + %d keyword candidates for crop='%s' (terms=%s)",
            min(len(semantic_results.get("documents", [[]])[0]) if semantic_results else 0, semantic_k),
            kw_total,
            crop_norm,
            content_terms,
        )

        # ── 3. Rerank by combined score, apply distance threshold ────────────
        ranked: list[tuple[float, RAGChunk]] = []
        for doc, meta, dist in candidates:
            # Apply distance threshold (permissive — keyword may rescue borderline chunks)
            if dist > self.distance_threshold * 1.15:  # slightly relaxed for keyword candidates
                log.debug("Retriever: skipped '%s' dist=%.3f (threshold=%.3f)",
                          meta.get("section", "?"), dist, self.distance_threshold)
                continue
            kw = self._keyword_score(doc, query_terms)
            # combined: semantic similarity (1-dist, higher=better) + keyword overlap
            combined = 0.7 * max(0.0, 1.0 - dist) + 0.3 * kw
            log.info(
                "Retriever: candidate '%s' — dist=%.3f  kw=%.2f  combined=%.3f",
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
            "Retriever: %d final chunks returned after hybrid rerank for crop='%s'",
            len(result), crop_norm,
        )
        return result

    def warm_up(self) -> None:
        """Eagerly load the embedding model in the current thread.

        Called at server startup so the model is ready before the first request.
        Avoids the 8-10 second delay on the first RAG query and prevents the
        model from being loaded inside a FastAPI worker thread.
        """
        try:
            self._get_encoder()
            log.info("RAGRetriever embedding model pre-warmed.")
        except Exception as e:
            log.warning("RAGRetriever warm_up failed: %s", e)
