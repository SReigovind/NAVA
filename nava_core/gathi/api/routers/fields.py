"""Fields, crops, plants, context, and events router."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from nava_core.shared.schemas import (
    CropContextRequest, CropContextResponse, CropCreateRequest,
    CropListResponse, CropResponse, CropUpdateRequest,
    EventListResponse, EventResponse,
    FieldContextRequest, FieldContextResponse, FieldCreateRequest,
    FieldListResponse, FieldResponse, FieldUpdateRequest,
    PlantCreateRequest, PlantListResponse, PlantResponse,
)
from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import field_store_for_user, require_user

router = APIRouter(prefix="/api", tags=["fields"])


def _refresh_field_context(store, field_id: int) -> None:
    ctx = store.auto_generate_field_context(field_id)
    store.update_field_context(field_id, ctx)


def _geocode_and_fetch_weather(db_path: str, field_id: int) -> None:
    """Background task: geocode the field's location string and fetch initial weather.

    Runs after field creation (and after location edits). Steps:
      1. Read the field's location string from DB.
      2. Resolve lat/lon via Nominatim (skipped if already decimal coords).
      3. Persist lat/lon to DB so Nominatim is never called again.
      4. Fetch weather from Open-Meteo and persist to DB.

    All failures are silent — the field is still usable without weather.
    """
    import logging
    from pathlib import Path
    from nava_core.shared.storage.field_store import FieldStore
    from nava_core.shared.utils.geo_context import resolve_coordinates, get_weather_context

    log = logging.getLogger("gathi.field_weather")
    try:
        store = FieldStore(Path(db_path))
        field = store.get_field(field_id)
        if not field:
            log.warning("[FIELD-WEATHER] Field %d not found in DB — skipping", field_id)
            return
        location = (field.get("location") or "").strip()
        if not location:
            log.info("[FIELD-WEATHER] Field %d has no location set — skipping", field_id)
            return

        # Step 1: Resolve coordinates (skip if already stored from a previous edit)
        lat = field.get("lat")
        lon = field.get("lon")
        if lat is None or lon is None:
            log.info("[FIELD-WEATHER] Resolving coordinates for field %d (%r)", field_id, location)
            coords = resolve_coordinates(location)
            if coords is None:
                log.warning("[FIELD-WEATHER] Could not resolve coordinates for field %d — no weather stored", field_id)
                return
            lat, lon = coords
            store.set_field_coordinates(field_id, lat, lon)
            log.info("[FIELD-WEATHER] Stored coordinates for field %d: lat=%.6f, lon=%.6f", field_id, lat, lon)
        else:
            log.info("[FIELD-WEATHER] Field %d already has coordinates: lat=%.6f, lon=%.6f", field_id, lat, lon)

        # Step 2: Fetch weather
        wx = get_weather_context(lat, lon)
        if wx is None:
            log.warning("[FIELD-WEATHER] Weather fetch failed for field %d — no weather stored", field_id)
            return
        store.update_field_weather(field_id, wx["temp"], wx["humidity"], wx["precipitation"], wx["wind_speed"])
        log.info(
            "[FIELD-WEATHER] ✓ Field %d (%s): %.1f°C, %s%% RH stored",
            field_id, field.get("name", ""), wx["temp"] or 0, wx["humidity"] or 0,
        )
    except Exception as exc:
        logging.getLogger("gathi.field_weather").warning(
            "[FIELD-WEATHER] Unexpected error for field %d: %s", field_id, exc
        )


# ── Fields ──────────────────────────────────────────────────────

@router.get("/fields", response_model=FieldListResponse)
def list_fields(user: UserRecord = Depends(require_user)) -> FieldListResponse:
    store = field_store_for_user(user)
    return FieldListResponse(fields=[FieldResponse(**f) for f in store.list_fields()])


@router.post("/fields", response_model=FieldResponse)
def create_field(
    request: FieldCreateRequest,
    bg_tasks: BackgroundTasks,
    user: UserRecord = Depends(require_user),
) -> FieldResponse:
    store = field_store_for_user(user)
    fid = store.create_field(request.name, request.location, request.area, request.soil_type, request.shared_context)
    _refresh_field_context(store, fid)
    field = store.get_field(fid)
    if not field:
        raise HTTPException(status_code=500, detail="Failed to create field")
    # Geocode + fetch initial weather in background so the response is instant
    if request.location and request.location.strip():
        bg_tasks.add_task(_geocode_and_fetch_weather, user.db_path, fid)
    return FieldResponse(**field)


@router.put("/fields", response_model=FieldResponse)
def update_field(
    request: FieldUpdateRequest,
    bg_tasks: BackgroundTasks,
    user: UserRecord = Depends(require_user),
) -> FieldResponse:
    store = field_store_for_user(user)
    # If the location changed, invalidate stored coordinates so geocoding re-runs
    if request.location is not None:
        old = store.get_field(request.field_id)
        if old and (request.location or "").strip() != (old.get("location") or "").strip():
            store.set_field_coordinates(request.field_id, None, None)  # type: ignore[arg-type]
    store.update_field(request.field_id, name=request.name, location=request.location,
                       area=request.area, soil_type=request.soil_type)
    _refresh_field_context(store, request.field_id)
    field = store.get_field(request.field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    # Re-geocode + refresh weather if location was updated
    if request.location and request.location.strip():
        bg_tasks.add_task(_geocode_and_fetch_weather, user.db_path, request.field_id)
    return FieldResponse(**field)


@router.delete("/fields/{field_id}")
def delete_field(field_id: int, user: UserRecord = Depends(require_user)) -> dict:
    """Permanently delete a field and all its crops, plants, events, and VNIR history."""
    store = field_store_for_user(user)
    field = store.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    store.delete_field(field_id)
    return {"status": "deleted", "field_id": field_id}


# ── Crops ────────────────────────────────────────────────────────

@router.get("/crops", response_model=CropListResponse)
def list_crops(field_id: int, user: UserRecord = Depends(require_user)) -> CropListResponse:
    store = field_store_for_user(user)
    return CropListResponse(crops=[CropResponse(**c) for c in store.list_crops(field_id)])


@router.post("/crops", response_model=CropResponse)
def create_crop(request: CropCreateRequest, user: UserRecord = Depends(require_user)) -> CropResponse:
    store = field_store_for_user(user)
    cid = store.create_crop(request.field_id, request.name, request.variety,
                            request.season, request.stage, request.notes)
    _refresh_field_context(store, request.field_id)
    crop = store.get_crop(cid)
    if not crop:
        raise HTTPException(status_code=500, detail="Failed to create crop")
    return CropResponse(**crop)


@router.put("/crops", response_model=CropResponse)
def update_crop(request: CropUpdateRequest, user: UserRecord = Depends(require_user)) -> CropResponse:
    store = field_store_for_user(user)
    crop = store.get_crop(request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    store.update_crop(request.crop_id, name=request.name, variety=request.variety,
                      season=request.season, stage=request.stage, notes=request.notes)
    _refresh_field_context(store, crop["field_id"])
    return CropResponse(**store.get_crop(request.crop_id))


@router.delete("/crops/{crop_id}")
def delete_crop(crop_id: int, user: UserRecord = Depends(require_user)) -> dict:
    store = field_store_for_user(user)
    crop = store.get_crop(crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    field_id = crop["field_id"]
    store.delete_crop(crop_id)
    _refresh_field_context(store, field_id)
    return {"status": "deleted", "crop_id": crop_id}


# ── Plants ───────────────────────────────────────────────────────

@router.get("/plants", response_model=PlantListResponse)
def list_plants(crop_id: int, user: UserRecord = Depends(require_user)) -> PlantListResponse:
    store = field_store_for_user(user)
    return PlantListResponse(plants=[PlantResponse(**p) for p in store.list_plants(crop_id)])


@router.post("/plants", response_model=PlantResponse)
def create_plant(request: PlantCreateRequest, user: UserRecord = Depends(require_user)) -> PlantResponse:
    store = field_store_for_user(user)
    pid = store.create_plant(request.crop_id, request.name, request.description)
    if pid < 0:
        raise HTTPException(status_code=400, detail="Failed to create plant (name may already exist)")
    plant = store.get_plant(pid)
    if not plant:
        raise HTTPException(status_code=500, detail="Plant not found after creation")
    return PlantResponse(**plant)


@router.delete("/plants/{plant_id}")
def delete_plant(plant_id: int, user: UserRecord = Depends(require_user)) -> dict:
    store = field_store_for_user(user)
    plant = store.get_plant(plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    store.delete_plant(plant_id)
    return {"status": "deleted", "plant_id": plant_id}


@router.delete("/plants/{plant_id}/history")
def clear_plant_history(plant_id: int, event_type: str | None = None,
                        user: UserRecord = Depends(require_user)) -> dict:
    """Clear detect/vnir history for a plant. Pass event_type=diagnose or vnir to clear only one."""
    store = field_store_for_user(user)
    plant = store.get_plant(plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    if event_type in ("vnir", None):
        store.clear_vnir_history(plant_id)
    store.delete_events_for_plant(plant_id, event_type=event_type)
    return {"status": "cleared", "plant_id": plant_id, "event_type": event_type or "all"}


# ── Context ──────────────────────────────────────────────────────

@router.post("/field-context", response_model=FieldContextResponse)
def update_field_context(request: FieldContextRequest, user: UserRecord = Depends(require_user)) -> FieldContextResponse:
    store = field_store_for_user(user)
    store.update_field_context(request.field_id, request.shared_context)
    field = store.get_field(request.field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return FieldContextResponse(field_id=request.field_id, shared_context=field.get("shared_context"))


@router.get("/field-context", response_model=FieldContextResponse)
def get_field_context(field_id: int, user: UserRecord = Depends(require_user)) -> FieldContextResponse:
    store = field_store_for_user(user)
    field = store.get_field(field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return FieldContextResponse(field_id=field_id, shared_context=field.get("shared_context"))


@router.get("/field-context/refresh", response_model=FieldContextResponse)
def refresh_field_context(field_id: int, user: UserRecord = Depends(require_user)) -> FieldContextResponse:
    """Re-generate the auto-populated field context from crops data."""
    store = field_store_for_user(user)
    if not store.get_field(field_id):
        raise HTTPException(status_code=404, detail="Field not found")
    _refresh_field_context(store, field_id)
    field = store.get_field(field_id)
    return FieldContextResponse(field_id=field_id, shared_context=field.get("shared_context"))


@router.post("/crop-context", response_model=CropContextResponse)
def update_crop_context(request: CropContextRequest, user: UserRecord = Depends(require_user)) -> CropContextResponse:
    store = field_store_for_user(user)
    store.update_crop_context(request.crop_id, request.notes)
    crop = store.get_crop(request.crop_id)
    if not crop:
        raise HTTPException(status_code=404, detail="Crop not found")
    return CropContextResponse(crop_id=request.crop_id, notes=crop.get("notes"))


# ── Events ───────────────────────────────────────────────────────

@router.get("/events", response_model=EventListResponse)
def list_events(
    field_id: int | None = None, crop_id: int | None = None,
    plant_id: int | None = None, limit: int = 50,
    user: UserRecord = Depends(require_user),
) -> EventListResponse:
    store = field_store_for_user(user)
    events = store.list_events(field_id=field_id, crop_id=crop_id, plant_id=plant_id, limit=limit)
    return EventListResponse(events=[EventResponse(**e) for e in events])


@router.delete("/events/{event_id}")
def delete_event(event_id: int, user: UserRecord = Depends(require_user)) -> dict:
    """Delete a single event by ID."""
    store = field_store_for_user(user)
    event = store.get_event(event_id)
    store.delete_event(event_id)
    if event and event.get("field_id"):
        try:
            _refresh_field_context(store, event["field_id"])
        except Exception:
            pass
    return {"status": "deleted", "event_id": event_id}


# ── Field Notes (manual, shown in UI) ────────────────────────────

class FieldNotesRequest(BaseModel):
    field_id: int
    notes: str


@router.post("/field-notes")
def update_field_notes(request: FieldNotesRequest, user: UserRecord = Depends(require_user)) -> dict:
    """Save manually written field notes (separate from auto-generated shared_context)."""
    store = field_store_for_user(user)
    store.update_field_notes(request.field_id, request.notes)
    return {"field_id": request.field_id, "notes": request.notes}

