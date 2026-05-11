"""Auth router — register, login, logout, me."""

from __future__ import annotations

import sqlite3
import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from nava_core.shared.schemas import AuthLoginRequest, AuthRegisterRequest, AuthResponse, UserResponse
from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import get_user_store, require_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _to_user_response(user: UserRecord) -> UserResponse:
    return UserResponse(
        id=user.id, name=user.name, email=user.email,
        onboarded=user.onboarded, location=user.location,
        goals=user.goals, created_at=user.created_at,
    )


def _preload_models():
    from nava_core.gathi.api.deps import get_predictor, get_vnir_pipeline
    log = logging.getLogger("mizhi.preload")
    log.info("Preloading ML models in background...")
    try:
        get_predictor()
        get_vnir_pipeline()
        log.info("Models preloaded successfully.")
    except Exception as e:
        log.error("Failed to preload models: %s", e)


@router.post("/register", response_model=AuthResponse)
def register(request: AuthRegisterRequest, bg_tasks: BackgroundTasks) -> AuthResponse:
    store = get_user_store()
    try:
        user = store.create_user(request.name, request.email, request.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = store.create_session(user.id)
    bg_tasks.add_task(_preload_models)
    return AuthResponse(token=token, user=_to_user_response(user))


@router.post("/login", response_model=AuthResponse)
def login(request: AuthLoginRequest, bg_tasks: BackgroundTasks) -> AuthResponse:
    store = get_user_store()
    user = store.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = store.create_session(user.id)
    bg_tasks.add_task(_preload_models)
    return AuthResponse(token=token, user=_to_user_response(user))


@router.post("/logout")
def logout(user: UserRecord = Depends(require_user)) -> dict:
    return {"status": "logged_out"}


@router.get("/me", response_model=UserResponse)
def me(bg_tasks: BackgroundTasks, user: UserRecord = Depends(require_user)) -> UserResponse:
    bg_tasks.add_task(_preload_models)
    return _to_user_response(user)
