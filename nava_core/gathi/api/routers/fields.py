"""Fields, crops, plants, context, and events router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
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


# ── Fields ──────────────────────────────────────────────────────

@router.get("/fields", response_model=FieldListResponse)
def list_fields(user: UserRecord = Depends(require_user)) -> FieldListResponse:
    store = field_store_for_user(user)
    return FieldListResponse(fields=[FieldResponse(**f) for f in store.list_fields()])


@router.post("/fields", response_model=FieldResponse)
def create_field(request: FieldCreateRequest, user: UserRecord = Depends(require_user)) -> FieldResponse:
    store = field_store_for_user(user)
    fid = store.create_field(request.name, request.location, request.area, request.soil_type, request.shared_context)
    _refresh_field_context(store, fid)
    field = store.get_field(fid)
    if not field:
        raise HTTPException(status_code=500, detail="Failed to create field")
    return FieldResponse(**field)


@router.put("/fields", response_model=FieldResponse)
def update_field(request: FieldUpdateRequest, user: UserRecord = Depends(require_user)) -> FieldResponse:
    store = field_store_for_user(user)
    store.update_field(request.field_id, name=request.name, location=request.location,
                       area=request.area, soil_type=request.soil_type)
    _refresh_field_context(store, request.field_id)
    field = store.get_field(request.field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return FieldResponse(**field)


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
    store.delete_event(event_id)
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

