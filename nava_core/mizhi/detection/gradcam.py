from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCamGenerator:
    """Generate Grad-CAM overlays for a classifier and target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._cam: Optional[GradCAM] = None

    def _ensure_cam(self) -> None:
        if self._cam is None:
            self._cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        class_index: int,
    ) -> Image.Image:
        self._ensure_cam()
        targets = [ClassifierOutputTarget(class_index)]
        grayscale_cam = self._cam(input_tensor=input_tensor, targets=targets)[0]

        rgb_img = np.array(original_image).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return Image.fromarray(overlay)
