from __future__ import annotations

from pathlib import Path
from typing import List


def load_labels(path: Path) -> List[str]:
    """Load class labels from a text file.

    Supported formats:
    - "index: label" per line
    - one label per line (fallback)
    """
    raw = path.read_text(encoding="utf-8").splitlines()
    indexed = {}

    for line in raw:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped and stripped[0].isdigit():
            idx_str, label = stripped.split(":", 1)
            try:
                idx = int(idx_str.strip())
            except ValueError:
                continue
            label = label.strip()
            if label:
                indexed[idx] = label

    if indexed:
        return [indexed[i] for i in sorted(indexed)]

    return [line.strip() for line in raw if line.strip()]
