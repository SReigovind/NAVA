"""HF Router chat client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from nava_core.shared.config import get_settings
from nava_core.shared.utils.logging import get_logger

log = get_logger("mozhi.client")


@dataclass(frozen=True)
class ChatConfig:
    model: str
    url: str
    api_key: str
    timeout: int
    temperature: float
    max_new_tokens: int


class ChatClient:
    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    @classmethod
    def from_settings(cls) -> "ChatClient":
        s = get_settings()
        return cls(ChatConfig(
            model=s.hf_model,
            url=s.hf_router_url,
            api_key=s.hf_api_key,
            timeout=s.hf_timeout_seconds,
            temperature=s.hf_temperature,
            max_new_tokens=s.hf_max_new_tokens,
        ))

    def send(
        self,
        messages: list[dict],
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        if not self.config.api_key:
            return None, "HF_API_KEY not set"

        payload = {
            "model": model_override or self.config.model,
            "messages": messages,
            "temperature": temperature_override if temperature_override is not None else self.config.temperature,
            "max_new_tokens": max_new_tokens_override if max_new_tokens_override is not None else self.config.max_new_tokens,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            resp = requests.post(self.config.url, headers=headers, json=payload, timeout=self.config.timeout)
            if resp.status_code != 200:
                log.warning("LLM API returned %d: %s", resp.status_code, resp.text[:200])
                return None, f"HTTP {resp.status_code}: {resp.text}"
            choices = resp.json().get("choices", [])
            if not choices:
                return None, "Empty response"
            content = choices[0].get("message", {}).get("content", "")
            return (content or None), (None if content else "Empty response")
        except Exception as exc:
            log.error("LLM request failed: %s", exc)
            return None, f"Network Error: {exc}"
