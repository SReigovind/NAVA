"""LLM-based keyword extraction for hybrid RAG search.

After the router decides RETRIEVE, this module asks the LLM to extract the
most agriculturally relevant keywords from the enriched query context.  These
keywords are then used for ChromaDB where_document filtered searches — giving
precision that pure vector similarity cannot (e.g. both "Black Sigatoka" and
"Yellow Sigatoka" are retrieved when the keyword "Sigatoka" is present).

The extractor is intentionally lightweight:
  - Uses the same small model as the router (Llama-3.1-8B-Instruct)
  - max_new_tokens=20 — forces a terse, list-only output
  - temperature=0.0 — deterministic keywords
  - On any failure (timeout, parse error) returns [] — caller falls back to
    heuristic term extraction
"""

from __future__ import annotations

import re

from nava_core.mozhi.chat.client import ChatClient
from nava_core.shared.utils.logging import get_logger

log = get_logger("yukthi.keywords")

_KEYWORD_SYSTEM_PROMPT = (
    "You are a search keyword extractor for an agricultural document knowledge base. "
    "Given a query context, extract the single most important search keywords or short phrases "
    "that would best retrieve relevant sections from crop management and disease reference documents.\n\n"
    "Rules:\n"
    "  - Output ONLY the keywords, one per line\n"
    "  - No numbers, bullets, explanations, or extra text\n"
    "  - Prefer specific agronomic terms (disease names, chemical names, techniques)\n"
    "  - Ignore generic words like 'crop', 'plant', 'field', 'banana'\n"
    "  - If the query mentions a condition or disease, that disease name MUST be a keyword"
)


class KeywordExtractor:
    """LLM-based keyword extractor — wraps the small model for targeted keyword generation."""

    def __init__(self, client: ChatClient, model: str, n: int = 3) -> None:
        """
        Args:
            client: Shared ChatClient instance (no new connection created).
            model:  Small/fast model identifier (e.g. hf_summary_model).
            n:      Number of keywords to extract (default 3).
        """
        self.client = client
        self.model = model
        self.n = n

    @classmethod
    def from_settings(cls, client: ChatClient) -> "KeywordExtractor":
        from nava_core.shared.config import get_settings
        s = get_settings()
        return cls(client=client, model=s.hf_summary_model)

    def extract(self, enriched_query: str) -> list[str]:
        """Extract up to self.n search keywords from the enriched query context.

        Returns a list of keyword strings, or [] on any failure.
        The caller should treat [] as "no LLM keywords available" and fall back
        to heuristic extraction.

        Args:
            enriched_query: The full enriched retrieval query, e.g.:
                "Crop: banana. Detected condition: Black Sigatoka.
                 What fungicide should I apply and at what dosage?"
        """
        user_prompt = (
            f"Extract exactly {self.n} search keywords from this agricultural query context:\n\n"
            f"{enriched_query}\n\n"
            f"Output {self.n} keywords, one per line:"
        )
        prompt = [
            {"role": "system", "content": _KEYWORD_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        log.info("KeywordExtractor ▶ query:\n%s", enriched_query)

        reply, error = self.client.send(
            prompt,
            model_override=self.model,
            temperature_override=0.0,
            max_new_tokens_override=25,  # n keywords × ~5 tokens each + newlines
        )

        log.info("KeywordExtractor ◀ raw response: %r", reply)

        if error or not reply:
            log.warning("KeywordExtractor: LLM call failed (%s) — no keywords", error)
            return []

        keywords = self._parse_keywords(reply)
        log.info("KeywordExtractor: extracted %d keywords → %s", len(keywords), keywords)
        return keywords

    def _parse_keywords(self, reply: str) -> list[str]:
        """Parse newline-separated keywords from the LLM reply.

        Strips bullets, numbers, and any non-alphabetic prefix characters.
        Filters out blank lines and generic stop-words.
        """
        _GENERIC = frozenset({
            "banana", "crop", "plant", "field", "farm", "growth", "grow",
            "disease", "issue", "problem", "information", "help", "context",
        })

        raw_lines = reply.strip().splitlines()
        keywords: list[str] = []
        for line in raw_lines:
            # Strip leading bullets / numbers / punctuation
            clean = re.sub(r"^[\s\d\.\-\*\•]+", "", line).strip()
            # Take only the first "phrase" if the model added an explanation after a colon
            if ":" in clean:
                clean = clean.split(":", 1)[0].strip()
            if not clean or len(clean) < 3:
                continue
            if clean.lower() in _GENERIC:
                continue
            keywords.append(clean)
            if len(keywords) >= self.n:
                break

        return keywords
