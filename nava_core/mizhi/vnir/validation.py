"""Plant ID validation — lightweight, no ML dependencies."""

from __future__ import annotations

import re

_PLANT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_plant_id(plant_id: str) -> str:
    """Sanitize plant_id to prevent path traversal attacks."""
    if not _PLANT_ID_RE.match(plant_id):
        raise ValueError(
            f"Invalid plant_id '{plant_id}' — only alphanumeric, hyphens, underscores (max 64 chars)"
        )
    return plant_id
