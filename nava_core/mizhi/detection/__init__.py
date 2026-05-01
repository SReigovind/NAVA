"""EfficientNet inference and Grad-CAM utilities."""

from .inference import (
    EfficientNetB0Predictor,
    PredictionResult,
    default_labels_path,
    default_model_path,
)

__all__ = [
    "EfficientNetB0Predictor",
    "PredictionResult",
    "default_labels_path",
    "default_model_path",
]
