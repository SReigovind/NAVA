from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image

from nava_core.mizhi.vnir import VNIRPipeline, build_vnir_panel


def _collect_images(root: Path) -> List[Path]:
    matches: List[Path] = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        matches.extend(root.rglob(f"*{ext}"))
    return sorted(matches)


def _group_by_parent(paths: List[Path], root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in paths:
        if path.parent == root:
            plant_id = root.name
        else:
            plant_id = path.parent.name
        groups.setdefault(plant_id, []).append(path)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description="VNIR monitoring smoke test")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "raw" / "thanal"),
        help="Input image folder or file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("logs") / "vnir"),
        help="Output directory for VNIR panels",
    )
    parser.add_argument(
        "--plant-id",
        type=str,
        default=None,
        help="Override plant id for all images",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Reset history for the selected plant id",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        return 1

    pipeline = VNIRPipeline()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        groups = {args.plant_id or input_path.stem: [input_path]}
    else:
        images = _collect_images(input_path)
        if not images:
            print("No images found.")
            return 1
        if args.plant_id:
            groups = {args.plant_id: images}
        else:
            groups = _group_by_parent(images, input_path)

    for plant_id, paths in groups.items():
        paths = sorted(paths)
        if args.clear_history:
            pipeline.clear_history(plant_id)

        plant_output = output_root / plant_id
        plant_output.mkdir(parents=True, exist_ok=True)

        for idx, image_path in enumerate(paths, start=1):
            image = Image.open(image_path).convert("RGB")
            stats, hsv_image, vnir_image = pipeline.process_image(image, plant_id)

            panel = build_vnir_panel(
                hsv_image,
                vnir_image,
                stats,
                source_path=image_path,
            )

            output_path = plant_output / f"{idx:03d}_{image_path.stem}.png"
            panel.save(output_path)

            print(
                f"{plant_id} | {image_path.name} | {stats.status} | "
                f"ratio={stats.ratio:.4f} | vs_base={stats.vs_baseline} | "
                f"vs_global={stats.vs_global} | vs_roll5={stats.vs_rolling} | "
                f"vs_prev_ckpt={stats.vs_prev_checkpoint}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
