from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from nava_core.mizhi.detection import (
    EfficientNetB0Predictor,
    default_labels_path,
    default_model_path,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_sample_images(root: Path) -> List[Path]:
    matches: List[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        matches.extend(root.rglob(f"*{ext}"))
    return matches


def build_output_paths(output_arg: str, count: int) -> List[Path]:
    output_path = Path(output_arg)
    if output_path.suffix:
        base_dir = output_path.parent
        base_name = output_path.stem
    else:
        base_dir = output_path
        base_name = "mizhi_gradcam_preview"

    base_dir.mkdir(parents=True, exist_ok=True)
    return [base_dir / f"{base_name}_{idx + 1:02d}.png" for idx in range(count)]


def _resample_filter() -> int:
    return getattr(Image.Resampling, "LANCZOS", Image.LANCZOS)


def measure_text_block_height(lines: List[str], font: ImageFont.ImageFont) -> int:
    padding = 10
    line_height = font.getbbox("Ag")[3] + 6
    return line_height * len(lines) + padding * 2


def build_text_block(lines: List[str], width: int, font: ImageFont.ImageFont) -> Image.Image:
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


def resize_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((max_width, max_height), _resample_filter())
    return resized


def build_side_by_side(
    source: Image.Image,
    cam: Image.Image,
    label_lines: List[str],
) -> Image.Image:
    font = ImageFont.load_default()
    max_output = 512
    text_height = measure_text_block_height(label_lines, font)
    max_tile_width = max_output // 2
    max_tile_height = max(1, max_output - text_height)

    source_resized = resize_to_fit(source, max_tile_width, max_tile_height)
    cam_resized = resize_to_fit(cam, source_resized.width, source_resized.height)

    width = source_resized.width * 2
    image_row = Image.new("RGB", (width, source_resized.height), (255, 255, 255))
    image_row.paste(source_resized, (0, 0))
    image_row.paste(cam_resized, (source_resized.width, 0))

    text_block = build_text_block(label_lines, width, font)
    combined = Image.new(
        "RGB",
        (width, source_resized.height + text_block.height),
        (255, 255, 255),
    )
    combined.paste(image_row, (0, 0))
    combined.paste(text_block, (0, source_resized.height))
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Mizhi EfficientNet smoke test")
    parser.add_argument("--image", type=str, help="Path to a test image")
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root() / "logs" / "mizhi_gradcam_preview"),
        help="Output file path or directory for Grad-CAM overlays",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of random images to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for repeatable sampling",
    )
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else None
    if image_path is not None:
        image_paths = [image_path]
    else:
        sample_root = repo_root() / "data" / "processed" / "efficientnet" / "test"
        image_paths = find_sample_images(sample_root)
        if image_paths:
            count = min(len(image_paths), max(args.count, 1))
            if args.seed is not None:
                random.seed(args.seed)
            image_paths = random.sample(image_paths, count)

    if not image_paths or not all(path.exists() for path in image_paths):
        print("No sample images found. Provide --image path.")
        return 1
    predictor = EfficientNetB0Predictor(
        model_path=default_model_path(),
        labels_path=default_labels_path(),
        device="cpu",
    )

    output_paths = build_output_paths(args.output, len(image_paths))

    for image_path, output_path in zip(image_paths, output_paths, strict=False):
        image = Image.open(image_path).convert("RGB")
        result, cam_image = predictor.predict_with_cam(image)
        cam_image = cam_image.resize(image.size, _resample_filter())

        print(f"Image: {image_path}")
        print(f"Class: {result.class_label} (index {result.class_index})")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Reliability: {result.reliability}")

        label_lines = [
            f"Actual Label: {image_path.parent.name}",
            "Left: Source image | Right: Grad-CAM overlay",
            f"Predicted Label: {result.class_label}",
            f"Index: {result.class_index}",
            f"Confidence: {result.confidence:.4f}",
            f"Reliability: {result.reliability}",
        ]
        combined = build_side_by_side(image, cam_image, label_lines)
        combined.save(output_path)
        print(f"Combined output saved to: {output_path}")
        print("-")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
