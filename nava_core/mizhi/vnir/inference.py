"""VNIR ONNX inference engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from nava_core.shared.utils.paths import models_dir
from nava_core.shared.utils.logging import get_logger

log = get_logger("mizhi.vnir")


def _default_model_path() -> Path:
    return models_dir() / "ThanalModel.onnx"


class VNIREngine:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or _default_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = ort.InferenceSession(
            str(self.model_path), sess_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        log.info("VNIR ONNX model loaded from %s", self.model_path.name)

    def predict(self, pil_image: Image.Image) -> Image.Image:
        img_resized = pil_image.resize((256, 256))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        input_tensor = np.expand_dims(img_np, axis=0)

        outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
        output_clipped = np.clip(outputs[0], 0, 1)
        output_array = np.squeeze(output_clipped)
        return Image.fromarray((output_array * 255).astype(np.uint8), mode="L")
