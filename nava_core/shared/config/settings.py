from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    efficientnet_model_path: Path
    efficientnet_labels_path: Path
    vnir_model_path: Path
    torch_device: str
    confidence_threshold: float
    vnir_stress_threshold_pct: float
    hf_api_key: str
    hf_model: str
    hf_router_url: str
    hf_timeout_seconds: int
    hf_temperature: float
    hf_max_new_tokens: int
    hf_summary_model: str
    hf_summary_temperature: float
    hf_summary_max_new_tokens: int
    mozhi_session_db_path: Path
    mozhi_max_history: int
    mozhi_summary_batch: int
    mozhi_summary_rollup: int


@lru_cache
def get_settings() -> Settings:
    repo_root = _repo_root()
    return Settings(
        efficientnet_model_path=_path_env(
            "NAVA_EFFICIENTNET_PATH",
            repo_root / "models" / "mizhi" / "EfficientNet-B0.pth",
        ),
        efficientnet_labels_path=_path_env(
            "NAVA_EFFICIENTNET_LABELS",
            repo_root / "models" / "mizhi" / "EfficientNet-B0-labels.txt",
        ),
        vnir_model_path=_path_env(
            "NAVA_VNIR_PATH",
            repo_root / "models" / "thanal" / "ThanalModel.onnx",
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
        mozhi_session_db_path=_path_env(
            "NAVA_MOZHI_DB",
            repo_root / "logs" / "mozhi" / "sessions.db",
        ),
        mozhi_max_history=_int_env("NAVA_MOZHI_MAX_HISTORY", 16),
        mozhi_summary_batch=_int_env("NAVA_MOZHI_SUMMARY_BATCH", 10),
        mozhi_summary_rollup=_int_env("NAVA_MOZHI_SUMMARY_ROLLUP", 5),
    )
