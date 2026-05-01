from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import models, transforms

from .gradcam import GradCamGenerator
from .labels import load_labels


@dataclass
class PredictionResult:
    class_index: int
    class_label: str
    confidence: float
    reliability: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path() -> Path:
    return _repo_root() / "models" / "mizhi" / "EfficientNet-B0.pth"


def default_labels_path() -> Path:
    return _repo_root() / "models" / "mizhi" / "EfficientNet-B0-labels.txt"


def _build_model(num_classes: int) -> torch.nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def _load_checkpoint(path: Path) -> object:
    return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: object) -> Tuple[torch.nn.Module | None, dict | None]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint, None

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                return None, checkpoint[key]
        return None, checkpoint

    return None, None


def _clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


class EfficientNetB0Predictor:
    def __init__(
        self,
        model_path: Path | None = None,
        labels_path: Path | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.85,
    ) -> None:
        self.device = torch.device(device)
        self.model_path = model_path or default_model_path()
        self.labels_path = labels_path or default_labels_path()
        self.labels = load_labels(self.labels_path)
        self.confidence_threshold = confidence_threshold

        self.model = _build_model(num_classes=len(self.labels))
        checkpoint = _load_checkpoint(self.model_path)
        model_obj, state_dict = _extract_state_dict(checkpoint)

        if model_obj is not None:
            self.model = model_obj
        elif state_dict is not None:
            cleaned = _clean_state_dict(state_dict)
            self.model.load_state_dict(cleaned, strict=True)
        else:
            raise ValueError("Unsupported checkpoint format")

        self.model.to(self.device)
        self.model.eval()

        self._resize = transforms.Resize(256)
        self._crop = transforms.CenterCrop(224)
        self._to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._cam = GradCamGenerator(self.model, self.model.features[-1])

    def _preprocess(self, image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
        image = image.convert("RGB")
        resized = self._resize(image)
        cropped = self._crop(resized)
        tensor = self._to_tensor(cropped)
        return tensor, cropped

    def predict(self, image: Image.Image) -> PredictionResult:
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

        return PredictionResult(
            class_index=idx,
            class_label=label,
            confidence=conf,
            reliability=reliability,
        )

    def predict_with_cam(self, image: Image.Image) -> Tuple[PredictionResult, Image.Image]:
        tensor, cropped = self._preprocess(image)
        input_tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, class_index = torch.max(probs, dim=1)

        idx = int(class_index.item())
        conf = float(confidence.item())
        label = self.labels[idx] if idx < len(self.labels) else "unknown"
        reliability = "RELIABLE" if conf >= self.confidence_threshold else "UNRELIABLE"

        cam_image = self._cam.generate(input_tensor, cropped, idx)

        result = PredictionResult(
            class_index=idx,
            class_label=label,
            confidence=conf,
            reliability=reliability,
        )
        return result, cam_image
