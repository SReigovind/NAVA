"""Chat orchestration with context injection and memory summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .client import ChatClient
from nava_core.mozhi.memory.session_store import SessionStore
from nava_core.shared.config import get_settings
from nava_core.shared.storage.field_store import FieldStore

if TYPE_CHECKING:
    from nava_core.yukthi.router import QueryRouter
    from nava_core.yukthi.retriever import RAGRetriever, RAGChunk

DEFAULT_SYSTEM_PROMPT = (
    "You are NAVA, a digital agronomist. Use the structured context provided in system messages "
    "(Field, Crop, Context, Events, Memory) to ground your answers. Do not mention missing fields "
    "or emit placeholders like 'null'. If context is missing, ask concise follow-up questions. "
    "Only answer agricultural questions; politely refuse non-agricultural requests. "
    "If asked for regulated chemical dosages or exact pesticide quantities, advise consulting local "
    "guidelines or an agronomist. Keep responses short to medium length, practical, and clear."
)


@dataclass
class ChatResult:
    session_id: str
    reply: str
    error: Optional[str] = None
    rag_used: bool = False
    rag_chunk_count: int = 0
    rag_chunks: list = None  # list of dicts: {source, section, snippet}

    def __post_init__(self):
        if self.rag_chunks is None:
            self.rag_chunks = []


class ChatService:
    def __init__(
        self,
        client: ChatClient,
        store: SessionStore,
        field_store: Optional[FieldStore] = None,
        max_history: int = 20,
        summary_batch: int = 14,
        summary_rollup: int = 5,
        summary_model: str = "",
        summary_temperature: float = 0.2,
        summary_max_new_tokens: int = 200,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        rag_retriever: Optional[RAGRetriever] = None,
        rag_router: Optional[QueryRouter] = None,
    ) -> None:
        self.client = client
        self.store = store
        self.field_store = field_store
        self.max_history = max_history
        self.summary_batch = summary_batch
        self.summary_rollup = summary_rollup
        self.summary_model = summary_model
        self.summary_temperature = summary_temperature
        self.summary_max_new_tokens = summary_max_new_tokens
        self.system_prompt = system_prompt
        self.rag_retriever = rag_retriever
        self.rag_router = rag_router

    @classmethod
    def from_settings(cls) -> "ChatService":
        s = get_settings()
        return cls(
            client=ChatClient.from_settings(),
            store=SessionStore(s.users_db_path.parent / "mozhi_sessions.db"),
            max_history=12,
            summary_batch=12,
            summary_rollup=s.mozhi_summary_rollup,
            summary_model=s.hf_summary_model,
            summary_temperature=s.hf_summary_temperature,
            summary_max_new_tokens=s.hf_summary_max_new_tokens,
            # RAG not wired in from_settings() — use from_settings_with_store() for full setup
        )

    @classmethod
    def from_settings_with_store(
        cls,
        store: SessionStore,
        field_store: Optional[FieldStore] = None,
        rag_retriever: Optional["RAGRetriever"] = None,
        rag_router: Optional["QueryRouter"] = None,
    ) -> "ChatService":
        """Build a ChatService, optionally accepting pre-built RAG singletons.

        When rag_retriever and rag_router are provided (from app.state, preloaded
        at startup), they are used directly. If not provided and yukthi is enabled,
        new instances are created — but this path is only used outside of the
        normal server context (e.g. tests, CLI tools).
        """
        s = get_settings()
        client = ChatClient.from_settings()

        # Use injected singletons if provided (normal server path)
        if rag_retriever is None and s.yukthi_enabled:
            # Fallback: create fresh instances (test/CLI path only)
            try:
                from nava_core.yukthi.store import YukthiStore
                from nava_core.yukthi.retriever import RAGRetriever as _Ret
                yukthi_store = YukthiStore(s.yukthi_chroma_dir)
                rag_retriever = _Ret(
                    store=yukthi_store,
                    embed_model=s.yukthi_embed_model,
                    top_k=s.yukthi_top_k,
                    distance_threshold=s.yukthi_distance_threshold,
                )
            except Exception as e:
                import logging
                logging.getLogger("mozhi.service").warning(
                    "Yukthi RAG fallback init failed: %s", e
                )

        if rag_router is None and rag_retriever is not None:
            try:
                from nava_core.yukthi.router import QueryRouter as _Router
                rag_router = _Router(client=client, model=s.hf_summary_model)
            except Exception as e:
                import logging
                logging.getLogger("mozhi.service").warning("QueryRouter init failed: %s", e)

        return cls(
            client=client,
            store=store,
            field_store=field_store,
            max_history=12,
            summary_batch=12,
            summary_rollup=s.mozhi_summary_rollup,
            summary_model=s.hf_summary_model,
            summary_temperature=s.hf_summary_temperature,
            summary_max_new_tokens=s.hf_summary_max_new_tokens,
            rag_retriever=rag_retriever,
            rag_router=rag_router,
        )


    # ── Context building ────────────────────────────────────────────

    def _build_context_message(self, field_id: Optional[int], crop_id: Optional[int]) -> Optional[str]:
        if not self.field_store:
            return None

        def _c(v: Optional[str]) -> Optional[str]:
            return v.strip() if isinstance(v, str) and v.strip() else None

        # Crop-specific: use rich context with full plant history
        if crop_id is not None:
            rich = self.field_store.get_rich_crop_context(crop_id)
            if rich:
                return "CROP CONTEXT (use silently to ground answers):\n" + rich
            return None

        # Field-level only: minimal metadata
        if field_id is not None:
            ctx = self.field_store.get_field_context(field_id)
            if not ctx:
                return None
            field = ctx.get("field") or {}
            sections = []
            fl = []
            for key, label in [("name", "Name"), ("location", "Location"), ("area", "Size"), ("soil_type", "Soil type")]:
                val = _c(field.get(key))
                if val:
                    fl.append(f"- {label}: {val}")
            if fl:
                sections.append("FIELD METADATA:\n" + "\n".join(fl))
            sc = _c(field.get("shared_context"))
            if sc:
                sections.append("FIELD CONTEXT (auto-generated):\n" + sc)
            fn = _c(field.get("field_notes"))
            if fn:
                sections.append("FIELD NOTES (user-written):\n" + fn)

            return ("STRUCTURED CONTEXT (use silently):\n" + "\n\n".join(sections)) if sections else None

        return None

    # ── Summary prompts ─────────────────────────────────────────────

    def _build_summary_prompt(self, messages: list[tuple]) -> list[dict]:
        lines = [f"{role.upper()}: {content}" for _, role, content in messages]
        return [
            {"role": "system", "content": (
                "Create a chat memory that preserves who asked what and how NAVA replied. "
                "Output 4-8 bullet points only. No headings or preamble. "
                "Each bullet must include both parts: 'User: ... | NAVA: ...'. "
                "Read the full assistant reply before summarizing; capture all key recommendations. "
                "Focus on agricultural content. Exclude out-of-scope requests. "
                "Do not invent missing data. Keep each bullet concise."
            )},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def _build_rollup_prompt(self, summaries: list[str]) -> list[dict]:
        content = "\n".join(f"- {s}" for s in summaries)
        return [
            {"role": "system", "content": (
                "Condense these chat memory bullets into a shorter memory for long-term context. "
                "Output 4-8 bullet points only. No headings or preamble. "
                "Each bullet: 'User: ... | NAVA: ...'. Agricultural only. "
                "Preserve actionable guidance. Do not invent missing data."
            )},
            {"role": "user", "content": content},
        ]

    def _summarize_if_needed(self, session_id: str) -> None:
        last_id = self.store.get_last_summarized_id(session_id)
        pending = self.store.count_messages_after(session_id, last_id)
        if pending < self.summary_batch:
            return

        batch = self.store.fetch_messages_with_ids(session_id, after_id=last_id, limit=self.summary_batch)
        if not batch:
            return

        summary, error = self.client.send(
            self._build_summary_prompt(batch),
            model_override=self.summary_model,
            temperature_override=self.summary_temperature,
            max_new_tokens_override=self.summary_max_new_tokens,
        )
        if error or not summary:
            return

        max_id = max(row[0] for row in batch)
        self.store.add_summary(session_id, level=1, content=summary)
        self.store.set_last_summarized_id(session_id, max_id)

        if self.store.count_summaries(session_id, level=1) >= self.summary_rollup:
            oldest = self.store.fetch_oldest_summaries(session_id, level=1, limit=self.summary_rollup)
            if not oldest:
                return
            rollup, rollup_error = self.client.send(
                self._build_rollup_prompt([r[1] for r in oldest]),
                model_override=self.summary_model,
                temperature_override=self.summary_temperature,
                max_new_tokens_override=self.summary_max_new_tokens,
            )
            if rollup_error or not rollup:
                return
            self.store.add_summary(session_id, level=2, content=rollup)
            self.store.delete_summaries([r[0] for r in oldest])

    def _summary_context(self, session_id: str) -> Optional[str]:
        level2 = self.store.fetch_recent_summaries(session_id, level=2, limit=1)
        level1 = self.store.fetch_recent_summaries(session_id, level=1, limit=2)
        sections = []
        if level2:
            sections.append("Long-term memory:\n" + level2[0])
        if level1:
            sections.append("Recent memory:\n" + "\n".join(level1))
        if not sections:
            return None
        return (
            "BACKGROUND MEMORY (Condensed past conversation for your reference. "
            "DO NOT reply using this bulleted format. Speak conversationally naturally as NAVA):\n" + 
            "\n\n".join(sections)
        )

    def _build_retrieval_query(
        self,
        message: str,
        crop_name: str,
        crop_id: int,
    ) -> str:
        """Build an enriched retrieval query for ChromaDB similarity search.

        Combines the user's message with available agronomic context:
          - Crop name (always included — anchors the embedding)
          - Latest detected disease label (if any) — critical for follow-up
            queries like "how do I treat this?" where 'this' needs anchoring
          - VNIR stress status (if recently detected)

        This produces a richer embedding that retrieves more relevant chunks
        than the raw user message alone.
        """
        parts = [f"Crop: {crop_name}."]

        if self.field_store:
            try:
                # Get the latest disease detection event for this crop
                diag_events = self.field_store.list_events(
                    crop_id=crop_id, event_type="diagnose", limit=1
                )
                if diag_events:
                    payload = diag_events[0].get("payload") or {}
                    label = payload.get("class_label", "")
                    if label and label.lower() not in ("healthy", "no scan", ""):
                        parts.append(f"Detected condition: {label}.")

                # VNIR stress if present
                vnir_events = self.field_store.list_events(
                    crop_id=crop_id, event_type="vnir", limit=1
                )
                if vnir_events:
                    vpayload = vnir_events[0].get("payload") or {}
                    vstatus = vpayload.get("status", "")
                    if vstatus and vstatus.lower() not in ("healthy", "no scan", ""):
                        parts.append(f"Stress monitoring status: {vstatus}.")
            except Exception:
                pass  # context enrichment is best-effort

        parts.append(message)
        return " ".join(parts)

    # ── Public API ──────────────────────────────────────────────────

    def get_summary_display(self, session_id: str) -> Optional[str]:
        level2 = self.store.fetch_recent_summaries(session_id, level=2, limit=1)
        level1 = self.store.fetch_recent_summaries(session_id, level=1, limit=2)
        parts = []
        if level2:
            parts.append("Long-term summary:\n" + level2[0])
        if level1:
            parts.append("Recent summaries:\n" + "\n".join(level1))
        return "\n\n".join(parts) if parts else None

    def get_history(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        return self.store.fetch_message_history(session_id, limit=limit)

    def chat(
        self,
        message: str,
        session_id: Optional[str],
        field_id: Optional[int] = None,
        crop_id: Optional[int] = None,
    ) -> ChatResult:
        session = session_id or self.store.create_session_id()
        if field_id is not None or crop_id is not None:
            self.store.set_session_context(session, field_id, crop_id)
        else:
            stored = self.store.get_session_context(session)
            if stored:
                field_id = stored.get("field_id")
                crop_id = stored.get("crop_id")

        history = self.store.fetch_messages(session, limit=self.max_history)
        messages = [{"role": "system", "content": self.system_prompt}]
        ctx = self._build_context_message(field_id, crop_id)
        if ctx:
            messages.append({"role": "system", "content": ctx})
        summary = self._summary_context(session)
        if summary:
            messages.append({"role": "system", "content": summary})

        # RAG: route first, then retrieve if warranted
        rag_used = False
        rag_chunk_count = 0
        rag_chunks: list[dict] = []
        if self.rag_router and self.rag_retriever and self.field_store and crop_id is not None:
            if self.rag_router.should_retrieve(message):
                crop = self.field_store.get_crop(crop_id)
                crop_name = (crop.get("name") or "").lower().strip() if crop else ""
                if crop_name:
                    retrieval_query = self._build_retrieval_query(
                        message=message,
                        crop_name=crop_name,
                        crop_id=crop_id,
                    )
                    chunks = self.rag_retriever.query(retrieval_query, crop=crop_name)
                    if chunks:
                        rag_used = True
                        rag_chunk_count = len(chunks)
                        rag_chunks = [
                            {
                                "source": c.source,
                                "section": c.section,
                                # Send a 300-char snippet for the UI tooltip
                                "snippet": c.text[:300].rstrip() + ("…" if len(c.text) > 300 else ""),
                            }
                            for c in chunks
                        ]
                        rag_block = (
                            "AGRONOMIC REFERENCE — VERIFIED SOURCE MATERIAL\n"
                            "The following passages are extracted from authoritative, peer-reviewed agricultural "
                            "sources and official crop management guidelines. This information is factually reliable. "
                            "Use it confidently and directly to ground your answer — do not hedge, qualify, or hold "
                            "back information from it. Do not tell the user you are consulting a reference.\n\n"
                        )
                        for chunk in chunks:
                            rag_block += f"[{chunk.source} — {chunk.section}]\n{chunk.text}\n\n"
                        messages.append({"role": "system", "content": rag_block.strip()})


        messages.extend(history)
        messages.append({"role": "user", "content": message})

        reply, error = self.client.send(messages)
        self.store.append_message(session, "user", message)
        if reply:
            # Persist RAG metadata alongside the assistant message so it
            # survives page refreshes and is returned by /api/chat/history
            rag_meta = (
                {
                    "rag_used": rag_used,
                    "rag_chunk_count": rag_chunk_count,
                    "rag_chunks": rag_chunks,
                }
                if rag_used
                else None
            )
            self.store.append_message(session, "assistant", reply, metadata=rag_meta)
        if error:
            return ChatResult(session_id=session, reply="", error=error)

        return ChatResult(
            session_id=session,
            reply=reply or "",
            rag_used=rag_used,
            rag_chunk_count=rag_chunk_count,
            rag_chunks=rag_chunks,
        )



    def clear_session(self, session_id: str) -> None:
        self.store.delete_session(session_id)
