"""Chat orchestration with context injection and memory summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .client import ChatClient
from nava_core.mozhi.memory.session_store import SessionStore
from nava_core.shared.config import get_settings
from nava_core.shared.storage.field_store import FieldStore

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

    @classmethod
    def from_settings(cls) -> "ChatService":
        s = get_settings()
        return cls(
            client=ChatClient.from_settings(),
            store=SessionStore(s.users_db_path.parent / "mozhi_sessions.db"),
            max_history=12, # 6 interactions
            summary_batch=12, # 6 interactions triggers summary
            summary_rollup=s.mozhi_summary_rollup,
            summary_model=s.hf_summary_model,
            summary_temperature=s.hf_summary_temperature,
            summary_max_new_tokens=s.hf_summary_max_new_tokens,
        )

    @classmethod
    def from_settings_with_store(
        cls, store: SessionStore, field_store: Optional[FieldStore] = None
    ) -> "ChatService":
        s = get_settings()
        return cls(
            client=ChatClient.from_settings(),
            store=store,
            field_store=field_store,
            max_history=12, # 6 interactions
            summary_batch=12, # 6 interactions triggers summary
            summary_rollup=s.mozhi_summary_rollup,
            summary_model=s.hf_summary_model,
            summary_temperature=s.hf_summary_temperature,
            summary_max_new_tokens=s.hf_summary_max_new_tokens,
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
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        reply, error = self.client.send(messages)
        self.store.append_message(session, "user", message)
        if reply:
            self.store.append_message(session, "assistant", reply)
        if error:
            return ChatResult(session_id=session, reply="", error=error)

        return ChatResult(session_id=session, reply=reply or "")

    def clear_session(self, session_id: str) -> None:
        self.store.delete_session(session_id)
