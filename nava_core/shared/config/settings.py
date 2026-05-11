"""Application settings loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a, **_kw):
        pass

from nava_core.shared.utils.paths import models_dir, logs_dir, project_root

# Load .env from nava/ project root
load_dotenv(project_root() / ".env")


def _path_env(key: str, default: Path) -> Path:
    return Path(os.getenv(key, str(default)))


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Mizhi — disease detection
    efficientnet_model_path: Path
    efficientnet_labels_path: Path
    torch_device: str
    confidence_threshold: float

    # Mizhi — VNIR
    vnir_model_path: Path
    vnir_stress_threshold_pct: float

    # Mozhi — LLM chat
    hf_api_key: str
    hf_model: str
    hf_router_url: str
    hf_timeout_seconds: int
    hf_temperature: float
    hf_max_new_tokens: int
    hf_summary_model: str
    hf_summary_temperature: float
    hf_summary_max_new_tokens: int

    # Mozhi — memory
    mozhi_max_history: int          # in messages (20 = 10 interaction pairs)
    mozhi_summary_batch: int        # in messages (14 = 7 interaction pairs)
    mozhi_summary_rollup: int       # number of level-1 summaries before rollup

    # Auth
    session_ttl_hours: int          # 0 = no expiry

    # Storage
    users_db_path: Path


@lru_cache
def get_settings() -> Settings:
    m = models_dir()  # nava/models/ — flat layout
    lg = logs_dir()
    return Settings(
        # Models are flat: nava/models/EfficientNet-B0.pth
        efficientnet_model_path=_path_env(
            "NAVA_EFFICIENTNET_PATH",
            m / "EfficientNet-B0.pth",
        ),
        efficientnet_labels_path=_path_env(
            "NAVA_EFFICIENTNET_LABELS",
            m / "EfficientNet-B0-labels.txt",
        ),
        vnir_model_path=_path_env(
            "NAVA_VNIR_PATH",
            m / "ThanalModel.onnx",
        ),
        torch_device=os.getenv("NAVA_TORCH_DEVICE", "cpu"),
        confidence_threshold=_float_env("NAVA_CONFIDENCE_THRESHOLD", 0.85),
        vnir_stress_threshold_pct=_float_env("NAVA_STRESS_THRESHOLD", 15.0),
        hf_api_key=os.getenv("HF_API_KEY", ""),
        hf_model=os.getenv(
            "HF_MODEL",
            "meta-llama/Meta-Llama-3-70B-Instruct:novita",
        ),
        hf_router_url=os.getenv(
            "HF_ROUTER_CHAT_URL",
            "https://router.huggingface.co/v1/chat/completions",
        ),
        hf_timeout_seconds=_int_env("HF_TIMEOUT", 30),
        hf_temperature=_float_env("HF_TEMPERATURE", 0.4),
        hf_max_new_tokens=_int_env("HF_MAX_NEW_TOKENS", 400),
        hf_summary_model=os.getenv(
            "HF_SUMMARY_MODEL",
            "meta-llama/Llama-3.1-8B-Instruct:novita",
        ),
        hf_summary_temperature=_float_env("HF_SUMMARY_TEMPERATURE", 0.2),
        hf_summary_max_new_tokens=_int_env("HF_SUMMARY_MAX_NEW_TOKENS", 200),
        mozhi_max_history=_int_env("NAVA_MOZHI_MAX_HISTORY", 20),
        mozhi_summary_batch=_int_env("NAVA_MOZHI_SUMMARY_BATCH", 14),
        mozhi_summary_rollup=_int_env("NAVA_MOZHI_SUMMARY_ROLLUP", 5),
        session_ttl_hours=_int_env("NAVA_SESSION_TTL_HOURS", 168),  # 7 days
        users_db_path=_path_env("NAVA_USERS_DB", lg / "users" / "users.db"),
    )
