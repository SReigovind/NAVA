from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


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


@dataclass(frozen=True)
class Settings:
    efficientnet_model_path: Path
    efficientnet_labels_path: Path
    vnir_model_path: Path
    torch_device: str
    confidence_threshold: float
    vnir_stress_threshold_pct: float


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
    )
