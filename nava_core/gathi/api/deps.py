"""Shared FastAPI dependencies — singletons and auth.

ML models are lazily loaded on first use, not at import time,
so the server can start even if model files are missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Header, HTTPException

from nava_core.shared.config import get_settings
from nava_core.shared.storage.field_store import FieldStore
from nava_core.shared.storage.user_store import UserRecord, UserStore

if TYPE_CHECKING:
    from nava_core.mizhi.detection import EfficientNetB0Predictor
    from nava_core.mizhi.vnir import VNIRPipeline
    from nava_core.mozhi.chat import ChatService
    from nava_core.mozhi.memory import SessionStore


@lru_cache
def get_predictor() -> "EfficientNetB0Predictor":
    from nava_core.mizhi.detection.inference import EfficientNetB0Predictor
    s = get_settings()
    return EfficientNetB0Predictor(
        model_path=s.efficientnet_model_path,
        labels_path=s.efficientnet_labels_path,
        device=s.torch_device,
        confidence_threshold=s.confidence_threshold,
    )


@lru_cache
def get_vnir_pipeline() -> "VNIRPipeline":
    from nava_core.mizhi.vnir.pipeline import VNIRPipeline
    s = get_settings()
    return VNIRPipeline(
        model_path=s.vnir_model_path,
        stress_threshold_pct=s.vnir_stress_threshold_pct,
    )


@lru_cache
def get_user_store() -> UserStore:
    s = get_settings()
    return UserStore(db_path=s.users_db_path, session_ttl_hours=s.session_ttl_hours)


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization


def require_user(authorization: str | None = Header(None)) -> UserRecord:
    """FastAPI dependency that extracts and validates the auth token."""
    token = _extract_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing auth token")
    user = get_user_store().get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired auth token")
    return user


def field_store_for_user(user: UserRecord) -> FieldStore:
    return FieldStore(Path(user.db_path))


def session_store_for_user(user: UserRecord) -> "SessionStore":
    from nava_core.mozhi.memory import SessionStore
    return SessionStore(Path(user.db_path))


def chat_service_for_user(user: UserRecord) -> "ChatService":
    from nava_core.mozhi.chat import ChatService
    return ChatService.from_settings_with_store(
        store=session_store_for_user(user),
        field_store=field_store_for_user(user),
    )
