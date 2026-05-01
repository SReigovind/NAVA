from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from .analyzer import VNIRStats


def _resample_filter() -> int:
    return getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)


def _measure_text_height(lines: List[str], font: ImageFont.ImageFont) -> int:
    padding = 10
    line_height = font.getbbox("Ag")[3] + 6
    return line_height * len(lines) + padding * 2


def _build_text_block(lines: List[str], width: int, font: ImageFont.ImageFont) -> Image.Image:
    padding = 10
    line_height = font.getbbox("Ag")[3] + 6
    text_height = line_height * len(lines) + padding * 2
    block = Image.new("RGB", (width, text_height), (255, 255, 255))
    draw = ImageDraw.Draw(block)
    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=(0, 0, 0))
        y += line_height
    return block


def _resize_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((max_width, max_height), _resample_filter())
    return resized


def build_vnir_panel(
    hsv_image: Image.Image,
    vnir_image: Image.Image,
    stats: VNIRStats,
    source_path: Path | None = None,
    max_output: int = 512,
) -> Image.Image:
    hsv_rgb = hsv_image.convert("RGB")
    vnir_rgb = vnir_image.convert("RGB")

    source_label = source_path.parent.name if source_path else "unknown"
    baseline = "n/a" if stats.baseline is None else f"{stats.baseline:.4f}"
    rolling_avg = "n/a" if stats.rolling_avg is None else f"{stats.rolling_avg:.4f}"
    prev_checkpoint = (
        "n/a" if stats.prev_checkpoint_avg is None else f"{stats.prev_checkpoint_avg:.4f}"
    )
    global_avg = "n/a" if stats.global_avg is None else f"{stats.global_avg:.4f}"
    vs_baseline = "n/a" if stats.vs_baseline is None else f"{stats.vs_baseline:+.1f}%"
    vs_global = "n/a" if stats.vs_global is None else f"{stats.vs_global:+.1f}%"
    vs_rolling = "n/a" if stats.vs_rolling is None else f"{stats.vs_rolling:+.1f}%"
    vs_prev_checkpoint = (
        "n/a" if stats.vs_prev_checkpoint is None else f"{stats.vs_prev_checkpoint:+.1f}%"
    )

    lines = [
        f"Source folder: {source_label}",
        "Left: HSV isolate | Right: VNIR output",
        f"Leaf state: {stats.leaf_state}",
        f"Status: {stats.status}",
        f"Avg Green: {stats.avg_g:.2f} | Avg VNIR: {stats.avg_vnir:.2f}",
        f"Ratio: {stats.ratio:.4f} | Baseline: {baseline} | Global Avg: {global_avg}",
        f"Rolling 5: {rolling_avg} | Prev Checkpoint: {prev_checkpoint}",
        f"Vs Baseline: {vs_baseline} | Vs Global: {vs_global}",
        f"Vs Rolling 5: {vs_rolling} | Vs Prev Checkpoint: {vs_prev_checkpoint}",
    ]

    font = ImageFont.load_default()
    text_height = _measure_text_height(lines, font)
    max_tile_width = max_output // 2
    max_tile_height = max(1, max_output - text_height)

    hsv_resized = _resize_to_fit(hsv_rgb, max_tile_width, max_tile_height)
    vnir_resized = vnir_rgb.resize(hsv_resized.size, _resample_filter())

    width = hsv_resized.width * 2
    image_row = Image.new("RGB", (width, hsv_resized.height), (255, 255, 255))
    image_row.paste(hsv_resized, (0, 0))
    image_row.paste(vnir_resized, (hsv_resized.width, 0))

    text_block = _build_text_block(lines, width, font)
    combined = Image.new(
        "RGB",
        (width, hsv_resized.height + text_block.height),
        (255, 255, 255),
    )
    combined.paste(image_row, (0, 0))
    combined.paste(text_block, (0, hsv_resized.height))
    return combined
