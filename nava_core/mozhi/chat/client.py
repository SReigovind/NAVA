from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from nava_core.shared.config import get_settings


def _extract_assistant_content(resp_json: dict) -> Optional[str]:
    try:
        choices = resp_json.get("choices", [])
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content", "")
        return content if content else None
    except Exception:
        return None


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
        settings = get_settings()
        config = ChatConfig(
            model=settings.hf_model,
            url=settings.hf_router_url,
            api_key=settings.hf_api_key,
            timeout=settings.hf_timeout_seconds,
            temperature=settings.hf_temperature,
            max_new_tokens=settings.hf_max_new_tokens,
        )
        return cls(config)

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
            "temperature": (
                self.config.temperature
                if temperature_override is None
                else temperature_override
            ),
            "max_new_tokens": (
                self.config.max_new_tokens
                if max_new_tokens_override is None
                else max_new_tokens_override
            ),
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}

        try:
            resp = requests.post(
                self.config.url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}: {resp.text}"
            content = _extract_assistant_content(resp.json())
            if not content:
                return None, "Empty response"
            return content, None
        except Exception as exc:
            return None, f"Network Error: {exc}"
