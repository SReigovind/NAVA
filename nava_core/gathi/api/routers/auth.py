"""Auth router — register, login, logout, me."""

from __future__ import annotations

import sqlite3
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, BackgroundTasks

from nava_core.shared.schemas import AuthLoginRequest, AuthRegisterRequest, AuthResponse, UserResponse, UpdateUserRequest, UpdatePasswordRequest
from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import get_user_store, require_user, _extract_token

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


def _models_loaded() -> bool:
    """Return True if both ML models are already in the lru_cache.
    Avoids spawning a redundant background thread on every /me request.
    """
    from nava_core.gathi.api.deps import get_predictor, get_vnir_pipeline
    return (
        get_predictor.cache_info().currsize > 0
        and get_vnir_pipeline.cache_info().currsize > 0
    )


def _refresh_user_weather(user_id: int) -> None:
    """Background task: refresh weather for every field of this user that has lat/lon stored.
    Called once on login. 1-second delay between each Open-Meteo call.
    """
    log = logging.getLogger("mizhi.weather_refresh")
    try:
        from nava_core.gathi.api.deps import field_store_for_user
        from nava_core.shared.utils.geo_context import refresh_user_weather
        from nava_core.shared.storage.field_store import FieldStore
        from pathlib import Path
        store = get_user_store()
        user = store.get_user(user_id)
        if not user:
            log.warning("[WEATHER-REFRESH] User %d not found — skipping", user_id)
            return
        fstore = FieldStore(Path(user.db_path))
        refresh_user_weather(fstore)
    except Exception as exc:
        log.warning("[WEATHER-REFRESH] Login refresh failed for user %d: %s", user_id, exc)


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
    bg_tasks.add_task(_refresh_user_weather, user.id)
    return AuthResponse(token=token, user=_to_user_response(user))


@router.post("/logout")
def logout(
    authorization: str | None = Header(None),
    user: UserRecord = Depends(require_user),
) -> dict:
    """Invalidate the session token so it cannot be reused after logout."""
    token = _extract_token(authorization)
    if token:
        get_user_store().delete_session(token)
    return {"status": "logged_out"}


@router.get("/me", response_model=UserResponse)
def me(bg_tasks: BackgroundTasks, user: UserRecord = Depends(require_user)) -> UserResponse:
    # Only spin up the preload task if models haven't been loaded yet.
    # Startup already handles this; the guard prevents a redundant thread
    # being spawned on every /me poll (e.g. auth keep-alive calls).
    if not _models_loaded():
        bg_tasks.add_task(_preload_models)
    return _to_user_response(user)


@router.put("/me", response_model=UserResponse)
def update_me(request: UpdateUserRequest, user: UserRecord = Depends(require_user)) -> UserResponse:
    store = get_user_store()
    updated = store.update_user(user.id, request.name)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_user_response(updated)


@router.put("/password")
def update_password(request: UpdatePasswordRequest, user: UserRecord = Depends(require_user)) -> dict:
    store = get_user_store()
    try:
        success = store.update_password(user.id, request.current_password, request.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not success:
        raise HTTPException(status_code=401, detail="Incorrect current password")
    return {"status": "password_updated"}


@router.delete("/me")
def delete_me(user: UserRecord = Depends(require_user)) -> dict:
    store = get_user_store()
    success = store.delete_user(user.id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete account")
    return {"status": "account_deleted"}
