"""Shared FastAPI dependencies — singletons and auth.

Heavy ML singletons (EfficientNet, VNIR, ChromaDB, RAG retriever) are
preloaded at server startup via the lifespan hook in startup.py and stored
on app.state. Deps here fetch those singletons — they never instantiate them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Header, HTTPException, Request

from nava_core.shared.config import get_settings
from nava_core.shared.storage.field_store import FieldStore
from nava_core.shared.storage.user_store import UserRecord, UserStore

if TYPE_CHECKING:
    from nava_core.mizhi.detection import EfficientNetB0Predictor
    from nava_core.mizhi.vnir import VNIRPipeline
    from nava_core.mozhi.chat import ChatService
    from nava_core.mozhi.memory import SessionStore
    from nava_core.yukthi.retriever import RAGRetriever
    from nava_core.yukthi.store import YukthiStore


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


def get_rag_retriever(request: Request) -> "RAGRetriever | None":
    """Return the startup-preloaded RAG retriever from app.state (or None)."""
    return getattr(request.app.state, "rag_retriever", None)


def get_yukthi_store(request: Request) -> "YukthiStore | None":
    """Return the startup-preloaded YukthiStore from app.state (or None)."""
    return getattr(request.app.state, "yukthi_store", None)


def chat_service_for_user(user: UserRecord, request: Request) -> "ChatService":
    """Build a ChatService for this user, reusing the startup-preloaded singletons."""
    from nava_core.mozhi.chat import ChatService
    from nava_core.shared.config import get_settings
    s = get_settings()

    rag_retriever = get_rag_retriever(request)
    rag_router = None

    if rag_retriever is not None:
        try:
            from nava_core.yukthi.router import QueryRouter
            from nava_core.mozhi.chat.client import ChatClient
            rag_router = QueryRouter(
                client=ChatClient.from_settings(),
                model=s.hf_summary_model,
            )
        except Exception as e:
            import logging
            logging.getLogger("nava.deps").warning("QueryRouter init failed: %s", e)

    return ChatService.from_settings_with_store(
        store=session_store_for_user(user),
        field_store=field_store_for_user(user),
        rag_retriever=rag_retriever,
        rag_router=rag_router,
    )
