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
    "You are a strict access-control gate for an agricultural knowledge base used by a digital agronomist. "
    "Your ONLY job is to decide whether a user message requires looking up external crop-science documents. "
    "You will be given the conversation history, including what was already said and asked by both the user and the digital agronomist."
    "Reply with exactly one word — either RETRIEVE or SKIP. No explanation, no punctuation.\n\n"

    "RETRIEVE — open the knowledge base — ONLY when the message asks for specific agronomic facts "
    "that would NOT already be known from the conversation context alone. Examples that warrant RETRIEVE:\n"
    "  - Named crop diseases or pest identification (e.g. 'what is Black Sigatoka?')\n"
    "  - Treatment or management protocols (e.g. 'what fungicide for Panama wilt?')\n"
    "  - Fertilizer rates, irrigation schedules, or soil amendment specifics\n"
    "  - digital agronomist prompts for more information based on the conversation context and user acknowledges to provide more information or confirms to get the information by responding with 'yes' or 'yeah' or 'ya' or any other similar positive word\n"
    "  - Harvesting techniques, variety selection, or post-harvest handling\n"
    "  - Any question where knowing detailed agronomic reference material would materially improve the answer\n\n"

    "SKIP — do NOT open the knowledge base — for ALL of the following:\n"
    "  - Greetings, thanks, or acknowledgements ('hi', 'okay', 'great', 'thanks')\n"
    "  - Negative replies to digital agronomist prompts ('no', 'nope', 'nah' or any other similar negative word)\n"
    "  - Questions about the assistant itself: its capabilities, features, what it can do, how it works "
    "('what can you do?', 'what are your capabilities?', 'how do you work?', 'tell me about yourself')\n"
    "  - Questions about the user's own field data, crop list, or summary "
    "('show my field overview', 'what crops do I have', 'summarize my field')\n"
    "  - Conversational follow-ups that the AI can answer from existing context "
    "('why did you say that?', 'can you explain more?', 'what did you mean?')\n"
    "  - Non-agricultural topics or off-topic chat\n"
    "  - Any self-referential or meta question about this chat session\n\n"

    "DEFAULT: when uncertain, always output SKIP."
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

    def should_retrieve(self, message: str, last_assistant_reply: str = "") -> bool:
        """Return True if RAG retrieval should be triggered for this message.

        Args:
            message:              The user's current message.
            last_assistant_reply: The most recent NAVA reply (if any).
                                  Included as context so that short follow-ups
                                  like 'yes' or 'tell me more' are interpreted
                                  correctly given what NAVA just said.
        """
        stripped = message.strip()
        if not stripped:
            log.debug("Router: SKIP (empty message)")
            return False

        # Build the routing input: previous NAVA context + current user message
        if last_assistant_reply and last_assistant_reply.strip():
            routing_input = (
                f"[Previous NAVA response]: {last_assistant_reply.strip()[:300]}\n"
                f"[User]: {stripped}"
            )
            log.info("Router input with NAVA context: %r", routing_input)
        else:
            routing_input = stripped
            log.info("Router input: %r", routing_input)

        return self._llm_classify(routing_input)

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
        log.info("Router: %r → %s", message, "RETRIEVE ✓" if result else "SKIP")
        return result
