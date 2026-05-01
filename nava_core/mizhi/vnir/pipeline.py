from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .analyzer import VNIRAnalyzer, VNIRStats
from .inference import VNIREngine


@dataclass
class LeafIsolationResult:
    leaf_state: str
    leaf_mask: np.ndarray
    hsv_visual: np.ndarray
    masked_rgb: np.ndarray


class VNIRPipeline:
    def __init__(
        self,
        model_path: Path | None = None,
        stress_threshold_pct: float = 15.0,
    ) -> None:
        self.engine = VNIREngine(model_path=model_path)
        self.stress_threshold_pct = stress_threshold_pct
        self.analyzers: dict[str, VNIRAnalyzer] = {}

    def _get_analyzer(self, plant_id: str) -> VNIRAnalyzer:
        if plant_id not in self.analyzers:
            self.analyzers[plant_id] = VNIRAnalyzer(
                plant_id=plant_id,
                stress_threshold_pct=self.stress_threshold_pct,
            )
        return self.analyzers[plant_id]

    def clear_history(self, plant_id: str) -> None:
        self._get_analyzer(plant_id).clear_history()

    def delete_history(self, plant_id: str) -> None:
        self.analyzers.pop(plant_id, None)
        history_path = Path(__file__).resolve().parents[3] / "logs" / "vnir" / f"{plant_id}_history.csv"
        if history_path.exists():
            history_path.unlink()

    def isolate_leaf(self, frame_bgr: np.ndarray) -> LeafIsolationResult:
        frame_256 = cv2.resize(frame_bgr, (256, 256))
        hsv_frame = cv2.cvtColor(frame_256, cv2.COLOR_BGR2HSV)
        total_pixels = 256 * 256

        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_hsv_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        green_hsv_mask = cv2.morphologyEx(green_hsv_mask, cv2.MORPH_CLOSE, kernel_large)
        green_hsv_mask = cv2.morphologyEx(green_hsv_mask, cv2.MORPH_OPEN, kernel_small)

        green_contours, _ = cv2.findContours(
            green_hsv_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        max_green_area = 0.0
        best_green_cnt = None
        if green_contours:
            best_green_cnt = max(green_contours, key=cv2.contourArea)
            max_green_area = cv2.contourArea(best_green_cnt)

        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([30, 255, 255])
        yellow_hsv_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        yellow_hsv_mask = cv2.morphologyEx(yellow_hsv_mask, cv2.MORPH_CLOSE, kernel_large)
        yellow_hsv_mask = cv2.morphologyEx(yellow_hsv_mask, cv2.MORPH_OPEN, kernel_small)

        yellow_contours, _ = cv2.findContours(
            yellow_hsv_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        max_yellow_area = 0.0
        best_yellow_cnt = None
        if yellow_contours:
            best_yellow_cnt = max(yellow_contours, key=cv2.contourArea)
            max_yellow_area = cv2.contourArea(best_yellow_cnt)

        leaf_state = "NONE"
        leaf_mask = np.zeros((256, 256), dtype=np.uint8)
        contour_bound = np.zeros((256, 256), dtype=np.uint8)
        min_area_required = total_pixels * 0.05

        if max_green_area >= max_yellow_area and max_green_area >= min_area_required:
            leaf_state = "GREEN"
            cv2.drawContours(contour_bound, [best_green_cnt], -1, 255, -1)
            leaf_mask = cv2.bitwise_and(green_hsv_mask, contour_bound)
        elif max_yellow_area > max_green_area and max_yellow_area >= min_area_required:
            leaf_state = "YELLOW_BROWN"
            cv2.drawContours(contour_bound, [best_yellow_cnt], -1, 255, -1)
            leaf_mask = cv2.bitwise_and(yellow_hsv_mask, contour_bound)

        hsv_visual = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
        hsv_visual = cv2.bitwise_and(hsv_visual, hsv_visual, mask=leaf_mask)

        masked_bgr = cv2.bitwise_and(frame_256, frame_256, mask=leaf_mask)
        masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

        return LeafIsolationResult(
            leaf_state=leaf_state,
            leaf_mask=leaf_mask,
            hsv_visual=hsv_visual,
            masked_rgb=masked_rgb,
        )

    def process_image(self, image: Image.Image, plant_id: str) -> tuple[VNIRStats, Image.Image, Image.Image]:
        frame_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        isolation = self.isolate_leaf(frame_bgr)

        hsv_image = Image.fromarray(cv2.cvtColor(isolation.hsv_visual, cv2.COLOR_BGR2RGB))
        vnir_image = Image.new("L", (256, 256), color=0)

        if isolation.leaf_state == "GREEN":
            vnir_image = self.engine.predict(Image.fromarray(isolation.masked_rgb))
            vnir_array = np.array(vnir_image).astype(np.float32)
            analyzer = self._get_analyzer(plant_id)
            stats = analyzer.analyze_and_log(
                isolation.masked_rgb,
                vnir_array,
                isolation.leaf_mask,
            )
        elif isolation.leaf_state == "YELLOW_BROWN":
            stats = VNIRStats(status="CRITICAL: Visual Stress")
        else:
            stats = VNIRStats(status="No Leaf Detected")

        stats.leaf_state = isolation.leaf_state
        return stats, hsv_image, vnir_image
