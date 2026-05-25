"""Diagnose router — disease detection with Grad-CAM."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from nava_core.shared.schemas import DiagnoseResponse
from nava_core.shared.utils import image_to_base64, load_image_from_bytes
from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import field_store_for_user, get_predictor, require_user
from nava_core.gathi.api.routers.fields import _refresh_field_context

router = APIRouter(prefix="/api", tags=["diagnose"])


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(
    image: UploadFile = File(...),
    plant_id: int = Form(...),
    crop_id: int | None = Form(None),
    field_id: int | None = Form(None),
    user: UserRecord = Depends(require_user),
) -> DiagnoseResponse:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")

    store = field_store_for_user(user)
    plant = store.get_plant(plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")

    pil_image = load_image_from_bytes(data)
    predictor = get_predictor()

    result = predictor.predict(pil_image)

    event_payload = {
        "plant_name": plant["name"],
        "class_label": result.class_label,
        "class_index": result.class_index,
        "confidence": result.confidence,
        "reliability": result.reliability,
    }

    if result.reliability == "UNRELIABLE":
        store.add_event(
            event_type="diagnose",
            field_id=field_id,
            crop_id=crop_id or plant["crop_id"],
            plant_id=plant_id,
            payload=event_payload,
        )
        effective_field_id = field_id or (plant.get("field_id") if plant else None)
        if effective_field_id:
            try:
                _refresh_field_context(store, effective_field_id)
            except Exception:
                pass
        return DiagnoseResponse(
            class_label=result.class_label,
            class_index=result.class_index,
            confidence=result.confidence,
            reliability=result.reliability,
        )

    # Only run Grad-CAM for reliable predictions — single call, no double inference
    result, cam_image = predictor.predict_with_cam(pil_image)
    event_payload.update({
        "class_label": result.class_label,
        "class_index": result.class_index,
        "confidence": result.confidence,
        "reliability": result.reliability,
    })
    store.add_event(
        event_type="diagnose",
        field_id=field_id,
        crop_id=crop_id or plant["crop_id"],
        plant_id=plant_id,
        payload=event_payload,
    )
    effective_field_id = field_id or (plant.get("field_id") if plant else None)
    if effective_field_id:
        try:
            _refresh_field_context(store, effective_field_id)
        except Exception:
            pass
    return DiagnoseResponse(
        class_label=result.class_label,
        class_index=result.class_index,
        confidence=result.confidence,
        reliability=result.reliability,
        original_image_base64=image_to_base64(pil_image),
        gradcam_image_base64=image_to_base64(cam_image),
    )
