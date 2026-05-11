"""Centralized path helpers — single source of truth.

All paths resolve relative to the nava/ folder, which is the
self-contained project root. No references to the parent NAVA directory.

File location: nava/nava_core/shared/utils/paths.py
    parents[0] = utils/
    parents[1] = shared/
    parents[2] = nava_core/
    parents[3] = nava/          ← project root
"""

from pathlib import Path


def project_root() -> Path:
    """Return the nava folder — the self-contained project root."""
    return Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    """Return the models directory at nava/models/."""
    return project_root() / "models"


def logs_dir() -> Path:
    """Return the logs directory at nava/logs/."""
    d = project_root() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def static_dir() -> Path:
    """Return the project root for static assets like NAVA-Logo.png."""
    return project_root()
