"""Smart query router for RAG retrieval gating.

Uses Llama-3.1-8B-Instruct (the same small model used for summarisation)
as the authoritative routing classifier. Only a degenerate guard (empty /
< 3 tokens) runs before the model is consulted — no keyword/regex routing.

Decision contract:
  should_retrieve(message) -> True  = RAG retrieval warranted
  should_retrieve(message) -> False = skip RAG, normal chat flow
"""

from __future__ import annotations

from nava_core.mozhi.chat.client import ChatClient
from nava_core.shared.utils.logging import get_logger

log = get_logger("yukthi.router")

_ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier for an agricultural AI assistant. "
    "Reply with exactly one word — either RETRIEVE or SKIP. No explanation, no punctuation.\n\n"

    "RETRIEVE = the message requires external agronomic reference knowledge to answer well. "
    "This includes: specific crop disease names or symptoms, pest identification, "
    "treatment or management recommendations, fertilizer or chemical application, "
    "irrigation schedules, harvesting techniques, variety selection, soil amendment, "
    "or any question where knowing detailed agronomic facts would help.\n\n"

    "SKIP = the message does NOT need an external knowledge lookup. Skip for:\n"
    "  - Greetings, acknowledgements, or filler (e.g. 'hi', 'thanks', 'okay')\n"
    "  - Questions about the user's own field data, field summary, or crop list "
    "(e.g. 'summarize my fields', 'what crops do I have', 'show my field overview')\n"
    "  - Questions about the chat history or previous messages\n"
    "  - Requests for general advice where the answer is already in the conversation context\n"
    "  - Completely non-agricultural topics\n\n"

    "When in doubt about a borderline case, lean towards SKIP."
)


class QueryRouter:
    """Routes user messages to determine whether RAG retrieval is needed."""

    def __init__(self, client: ChatClient, model: str, timeout: int = 8) -> None:
        """
        Args:
            client: The existing ChatClient instance (shared — no new connection).
            model:  Model identifier for the small router model (hf_summary_model).
            timeout: Max seconds to wait for the routing decision.
                     On timeout or error, defaults to SKIP.
        """
        self.client = client
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_settings(cls, client: ChatClient) -> "QueryRouter":
        from nava_core.shared.config import get_settings
        s = get_settings()
        return cls(client=client, model=s.hf_summary_model)

    def should_retrieve(self, message: str) -> bool:
        """Return True if RAG retrieval should be triggered for this message."""
        # Degenerate guard: skip API call for empty or trivially short inputs
        stripped = message.strip()
        if not stripped or len(stripped.split()) < 3:
            log.debug("Router: SKIP (degenerate guard — too short)")
            return False

        return self._llm_classify(stripped)

    def _llm_classify(self, message: str) -> bool:
        prompt = [
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user",   "content": message},
        ]
        reply, error = self.client.send(
            prompt,
            model_override=self.model,
            temperature_override=0.0,
            max_new_tokens_override=5,
        )

        if error or not reply:
            log.warning("Router: classification failed (%s) — defaulting to SKIP", error)
            return False

        decision = reply.strip().upper()
        result = decision.startswith("RETRIEVE")
        log.info("Router: %r → %s", message[:80], "RETRIEVE ✓" if result else "SKIP")
        return result
