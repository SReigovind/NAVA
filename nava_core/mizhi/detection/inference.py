"""EfficientNet-B0 inference — single forward pass for both prediction and Grad-CAM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torchvision import models, transforms

from nava_core.shared.utils.logging import get_logger
from nava_core.shared.utils.paths import models_dir
from .gradcam import GradCamGenerator
from .labels import load_labels

log = get_logger("mizhi.detection")


@dataclass
class PredictionResult:
    class_index: int
    class_label: str
    confidence: float
    reliability: str


def _default_model_path() -> Path:
    return models_dir() / "EfficientNet-B0.pth"


def _default_labels_path() -> Path:
    return models_dir() / "EfficientNet-B0-labels.txt"


def _build_model(num_classes: int) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model


def _extract_state_dict(checkpoint: object) -> Tuple[Optional[torch.nn.Module], Optional[dict]]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint, None
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                return None, checkpoint[key]
        return None, checkpoint
    return None, None


def _clean_state_dict(state_dict: dict) -> dict:
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


class EfficientNetB0Predictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.85,
    ) -> None:
        self.device = torch.device(device)
        self.model_path = model_path or _default_model_path()
        self.labels_path = labels_path or _default_labels_path()
        self.labels = load_labels(self.labels_path)
        self.confidence_threshold = confidence_threshold

        self.model = _build_model(num_classes=len(self.labels))
        checkpoint = torch.load(self.model_path, map_location="cpu")
        model_obj, state_dict = _extract_state_dict(checkpoint)

        if model_obj is not None:
            self.model = model_obj
        elif state_dict is not None:
            self.model.load_state_dict(_clean_state_dict(state_dict), strict=True)
        else:
            raise ValueError("Unsupported checkpoint format")

        self.model.to(self.device)
        self.model.eval()

        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self._cam = GradCamGenerator(self.model, self.model.features[-1])
        log.info("EfficientNet-B0 loaded: %d classes, device=%s", len(self.labels), device)

    def _preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
        image = image.convert("RGB")
        # Keep the cropped version for Grad-CAM overlay
        cropped = transforms.CenterCrop(224)(transforms.Resize(256)(image))
        tensor = self._transform(image)
        return tensor, cropped

    def predict(self, image: Image.Image) -> PredictionResult:
        """Run inference only — no Grad-CAM."""
        tensor, _ = self._preprocess(image)
        input_tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_index = torch.max(probs, dim=1)

        idx = int(class_index.item())
        conf = float(confidence.item())
        label = self.labels[idx] if idx < len(self.labels) else "unknown"
        reliability = "RELIABLE" if conf >= self.confidence_threshold else "UNRELIABLE"
        return PredictionResult(class_index=idx, class_label=label, confidence=conf, reliability=reliability)

    def predict_with_cam(self, image: Image.Image) -> Tuple[PredictionResult, Image.Image]:
        """Single forward pass for prediction + Grad-CAM overlay.

        NOTE: No torch.no_grad() wrapper — Grad-CAM requires gradient flow.
        """
        tensor, cropped = self._preprocess(image)
        input_tensor = tensor.unsqueeze(0).to(self.device)

        # Forward pass (Grad-CAM will compute gradients internally)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, class_index = torch.max(probs, dim=1)

        idx = int(class_index.item())
        conf = float(confidence.item())
        label = self.labels[idx] if idx < len(self.labels) else "unknown"
        reliability = "RELIABLE" if conf >= self.confidence_threshold else "UNRELIABLE"

        cam_image = self._cam.generate(input_tensor, cropped, idx)

        result = PredictionResult(class_index=idx, class_label=label, confidence=conf, reliability=reliability)
        log.info("Prediction: %s (%.2f%%) — %s", label, conf * 100, reliability)
        return result, cam_image
