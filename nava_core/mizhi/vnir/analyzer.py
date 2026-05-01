from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    def __init__(
        self,
        plant_id: str = "plant_default",
        history_dir: Path | None = None,
        stress_threshold_pct: float = 15.0,
    ) -> None:
        self.plant_id = plant_id
        self.stress_threshold_pct = stress_threshold_pct
        self.history_dir = history_dir or (_repo_root() / "logs" / "vnir")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.history_dir / f"{self.plant_id}_history.csv"

        if not self.csv_file.exists():
            self._create_csv()

    def _create_csv(self) -> None:
        with self.csv_file.open(mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Timestamp",
                    "Avg_Green",
                    "Avg_VNIR",
                    "Health_Ratio",
                    "Baseline",
                    "Rolling5_Avg",
                    "PrevCheckpoint_Avg",
                    "Global_Avg",
                    "Status",
                    "Vs_Baseline",
                    "Vs_Global",
                    "Vs_Rolling5",
                    "Vs_Prev_Checkpoint",
                ]
            )

    def clear_history(self) -> None:
        self._create_csv()

    def analyze_and_log(
        self,
        rgb_image: np.ndarray,
        vnir_image: np.ndarray,
        leaf_mask: np.ndarray,
    ) -> VNIRStats:
        g_channel = rgb_image[:, :, 1].astype(np.float32)
        leaf_g = g_channel[leaf_mask > 0]
        leaf_vnir = vnir_image[leaf_mask > 0]

        if len(leaf_vnir) == 0:
            return VNIRStats(status="No Leaf Detected")

        avg_g = float(np.mean(leaf_g))
        avg_vnir = float(np.mean(leaf_vnir))
        current_ratio = float(avg_vnir / (avg_g + 1e-5))

        history_ratios: list[float] = []
        if self.csv_file.exists():
            with self.csv_file.open(mode="r") as file:
                reader = list(csv.reader(file))
                if len(reader) > 1:
                    for row in reader[1:]:
                        try:
                            history_ratios.append(float(row[3]))
                        except (ValueError, IndexError):
                            continue

        history_ratios.append(current_ratio)
        total_scans = len(history_ratios)

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
            baseline = float(np.mean(history_ratios[0:5]))
            global_avg = float(np.mean(history_ratios))
            current_5_avg = float(np.mean(history_ratios[-5:]))
            if total_scans >= 10:
                prev_checkpoint_avg = float(np.mean(history_ratios[-10:-5]))
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

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.csv_file.open(mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    timestamp,
                    f"{avg_g:.2f}",
                    f"{avg_vnir:.2f}",
                    f"{current_ratio:.4f}",
                    "" if stats.baseline is None else f"{stats.baseline:.4f}",
                    "" if stats.rolling_avg is None else f"{stats.rolling_avg:.4f}",
                    "" if stats.prev_checkpoint_avg is None else f"{stats.prev_checkpoint_avg:.4f}",
                    "" if stats.global_avg is None else f"{stats.global_avg:.4f}",
                    stats.status,
                    "" if stats.vs_baseline is None else f"{stats.vs_baseline:.2f}",
                    "" if stats.vs_global is None else f"{stats.vs_global:.2f}",
                    "" if stats.vs_rolling is None else f"{stats.vs_rolling:.2f}",
                    "" if stats.vs_prev_checkpoint is None else f"{stats.vs_prev_checkpoint:.2f}",
                ]
            )

        return stats
