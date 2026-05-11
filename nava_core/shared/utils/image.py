"""Image conversion helpers."""

from __future__ import annotations

import base64
import io

from PIL import Image


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a data-URI base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Open raw bytes as a PIL RGB image."""
    return Image.open(io.BytesIO(data)).convert("RGB")
