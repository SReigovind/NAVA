from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_model_path() -> Path:
    return _repo_root() / "models" / "thanal" / "ThanalModel.onnx"


class VNIREngine:
    def __init__(self, model_path: Path | None = None, device: str = "cpu") -> None:
        self.model_path = model_path or default_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]

        self.ort_session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers,
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def predict(self, pil_image: Image.Image) -> Image.Image:
        img_resized = pil_image.resize((256, 256))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        input_tensor = np.expand_dims(img_np, axis=0)

        outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
        output_tensor = outputs[0]

        output_clipped = np.clip(output_tensor, 0, 1)
        output_array = np.squeeze(output_clipped)
        vnir_image = Image.fromarray((output_array * 255).astype(np.uint8), mode="L")
        return vnir_image
