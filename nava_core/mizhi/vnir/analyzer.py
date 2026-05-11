"""VNIR analytics — DB-backed per-user history instead of global CSV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _safe_pct(new_val: float, old_val: float) -> float:
    denom = old_val if abs(old_val) > 1e-6 else 1e-6
    return ((new_val - old_val) / denom) * 100


@dataclass
class VNIRStats:
    status: str
    avg_g: float = 0.0
    avg_vnir: float = 0.0
    ratio: float = 0.0
    baseline: float | None = None
    rolling_avg: float | None = None
    prev_checkpoint_avg: float | None = None
    global_avg: float | None = None
    vs_baseline: float | None = None
    vs_global: float | None = None
    vs_rolling: float | None = None
    vs_prev_checkpoint: float | None = None
    ready: bool = False
    leaf_state: str = "NONE"
    scan_index: int = 0


class VNIRAnalyzer:
    """Compute VNIR stress statistics from a history of ratios.

    Unlike the original, this does NOT write CSV files — the caller
    (pipeline or API) is responsible for persisting via FieldStore.
    """

    def __init__(self, stress_threshold_pct: float = 15.0) -> None:
        self.stress_threshold_pct = stress_threshold_pct

    def analyze(
        self,
        rgb_image: np.ndarray,
        vnir_image: np.ndarray,
        leaf_mask: np.ndarray,
        history_ratios: list[float],
    ) -> VNIRStats:
        g_channel = rgb_image[:, :, 1].astype(np.float32)
        leaf_g = g_channel[leaf_mask > 0]
        leaf_vnir = vnir_image[leaf_mask > 0]

        if len(leaf_vnir) == 0:
            return VNIRStats(status="No Leaf Detected")

        avg_g = float(np.mean(leaf_g))
        avg_vnir = float(np.mean(leaf_vnir))
        current_ratio = float(avg_vnir / (avg_g + 1e-5))

        all_ratios = history_ratios + [current_ratio]
        total_scans = len(all_ratios)

        stats = VNIRStats(
            status="Calibrating",
            avg_g=avg_g,
            avg_vnir=avg_vnir,
            ratio=current_ratio,
            ready=False,
            scan_index=total_scans,
        )

        if total_scans < 5:
            stats.status = f"Calibrating ({total_scans}/5)"
        else:
            stats.ready = True
            baseline = float(np.mean(all_ratios[0:5]))
            global_avg = float(np.mean(all_ratios))
            current_5_avg = float(np.mean(all_ratios[-5:]))
            if total_scans >= 10:
                prev_checkpoint_avg = float(np.mean(all_ratios[-10:-5]))
            else:
                prev_checkpoint_avg = baseline

            stats.baseline = baseline
            stats.rolling_avg = current_5_avg
            stats.prev_checkpoint_avg = prev_checkpoint_avg
            stats.global_avg = global_avg

            stats.vs_baseline = _safe_pct(current_ratio, baseline)
            stats.vs_global = _safe_pct(current_ratio, global_avg)
            stats.vs_rolling = _safe_pct(current_ratio, current_5_avg)
            stats.vs_prev_checkpoint = _safe_pct(current_ratio, prev_checkpoint_avg)

            if stats.vs_baseline <= -self.stress_threshold_pct:
                stats.status = "WARNING: STRESS"
            else:
                stats.status = "OK"

        return stats
