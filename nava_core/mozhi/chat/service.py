from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .client import ChatClient
from nava_core.mozhi.memory.session_store import SessionStore
from nava_core.shared.config import get_settings

DEFAULT_SYSTEM_PROMPT = (
    "You are NAVA, a digital agronomist. Only answer agricultural questions. "
    "If the query is not relevant to agriculture or is about non-agricultural topics reply politely but refuse to answer. General talk is accepted."
    "If asked for regulated chemical dosages or exact pesticide quantities, "
    "ask for local guidelines or advise consulting an agronomist. "
    "Try not to give too much of a lecture. Aim for a short to medium length response that is informative but not overwhelming. Use human-like language."
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
        max_history: int = 16,
        summary_batch: int = 10,
        summary_rollup: int = 5,
        summary_model: str = "",
        summary_temperature: float = 0.2,
        summary_max_new_tokens: int = 200,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.store = store
        self.max_history = max_history
        self.summary_batch = summary_batch
        self.summary_rollup = summary_rollup
        self.summary_model = summary_model
        self.summary_temperature = summary_temperature
        self.summary_max_new_tokens = summary_max_new_tokens
        self.system_prompt = system_prompt

    @classmethod
    def from_settings(cls) -> "ChatService":
        settings = get_settings()
        store = SessionStore(settings.mozhi_session_db_path)
        client = ChatClient.from_settings()
        return cls(
            client=client,
            store=store,
            max_history=settings.mozhi_max_history,
            summary_batch=settings.mozhi_summary_batch,
            summary_rollup=settings.mozhi_summary_rollup,
            summary_model=settings.hf_summary_model,
            summary_temperature=settings.hf_summary_temperature,
            summary_max_new_tokens=settings.hf_summary_max_new_tokens,
        )

    def _build_summary_prompt(self, messages: list[tuple]) -> list[dict]:
        lines = [f"{role.upper()}: {content}" for _, role, content in messages]
        content = "\n".join(lines)
        summary_system = (
            "Create a chat memory that preserves who asked what and how NAVA replied. "
            "Output 4-8 bullet points only. No headings or preamble. "
            "Each bullet must include both parts in this format: "
            "'User: ... | NAVA: ...'. "
            "Read the full assistant reply before summarizing; capture all key recommendations, "
            "steps, and cautions, not just the opening lines. "
            "Merge related details into one bullet; keep concrete numbers, timing, and constraints. "
            "Focus on agricultural content: crops, symptoms, suspected issues, actions, advice, "
            "constraints, locations, timelines, and open questions. "
            "Exclude any out-of-scope requests and refusals. "
            "Keep each bullet concise and specific."
        )
        return [
            {"role": "system", "content": summary_system},
            {"role": "user", "content": content},
        ]

    def _build_rollup_prompt(self, summaries: list[str]) -> list[dict]:
        content = "\n".join(f"- {summary}" for summary in summaries)
        rollup_system = (
            "Condense these chat memory bullets into a shorter memory for long-term context. "
            "Output 4-8 bullet points only. No headings or preamble. "
            "Each bullet must include both parts in this format: "
            "'User: ... | NAVA: ...'. "
            "Keep it agricultural only and ignore out-of-scope content or refusals. "
            "Preserve the most actionable guidance from NAVA's replies."
        )
        return [
            {"role": "system", "content": rollup_system},
            {"role": "user", "content": content},
        ]

    def _summarize_if_needed(self, session_id: str) -> None:
        last_id = self.store.get_last_summarized_id(session_id)
        pending = self.store.count_messages_after(session_id, last_id)
        if pending < self.summary_batch:
            return

        batch = self.store.fetch_messages_with_ids(
            session_id,
            after_id=last_id,
            limit=self.summary_batch,
        )
        if not batch:
            return

        messages = self._build_summary_prompt(batch)
        summary, error = self.client.send(
            messages,
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
            oldest = self.store.fetch_oldest_summaries(
                session_id,
                level=1,
                limit=self.summary_rollup,
            )
            if not oldest:
                return
            rollup_messages = self._build_rollup_prompt([row[1] for row in oldest])
            rollup, rollup_error = self.client.send(
                rollup_messages,
                model_override=self.summary_model,
                temperature_override=self.summary_temperature,
                max_new_tokens_override=self.summary_max_new_tokens,
            )
            if rollup_error or not rollup:
                return
            self.store.add_summary(session_id, level=2, content=rollup)
            self.store.delete_summaries([row[0] for row in oldest])

    def _summary_sections(self, session_id: str) -> list[str]:
        level2 = self.store.fetch_recent_summaries(session_id, level=2, limit=1)
        level1 = self.store.fetch_recent_summaries(session_id, level=1, limit=2)
        sections = []
        if level2:
            sections.append("Long-term summary:\n" + level2[0])
        if level1:
            sections.append("Recent summaries:\n" + "\n".join(level1))
        return sections

    def _summary_context(self, session_id: str) -> Optional[str]:
        sections = self._summary_sections(session_id)
        if not sections:
            return None
        return "Memory (use for context, do not quote):\n" + "\n\n".join(sections)

    def get_summary_display(self, session_id: str) -> Optional[str]:
        sections = self._summary_sections(session_id)
        if not sections:
            return None
        return "\n\n".join(sections)

    def chat(self, message: str, session_id: Optional[str]) -> ChatResult:
        session = session_id or self.store.create_session_id()
        history = self.store.fetch_messages(session, limit=self.max_history)
        messages = [{"role": "system", "content": self.system_prompt}]
        summary_context = self._summary_context(session)
        if summary_context:
            messages.append({"role": "system", "content": summary_context})
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        reply, error = self.client.send(messages)
        self.store.append_message(session, "user", message)
        if reply:
            self.store.append_message(session, "assistant", reply)

        if error:
            return ChatResult(session_id=session, reply="", error=error)

        self._summarize_if_needed(session)

        return ChatResult(session_id=session, reply=reply or "")

    def clear_session(self, session_id: str) -> None:
        self.store.delete_session(session_id)
