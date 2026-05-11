"""VNIR router — stress monitoring with per-plant isolation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from nava_core.mizhi.vnir import validate_plant_id
from nava_core.shared.schemas import VNIRResponse
from nava_core.shared.utils import image_to_base64, load_image_from_bytes
from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import field_store_for_user, get_vnir_pipeline, require_user

router = APIRouter(prefix="/api", tags=["vnir"])


@router.post("/vnir-clear")
def clear_vnir_history(
    plant_id: int = Form(...),
    user: UserRecord = Depends(require_user),
) -> dict:
    store = field_store_for_user(user)
    plant = store.get_plant(plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    store.clear_vnir_history(plant_id)
    return {"status": "cleared", "plant_id": plant_id}


@router.post("/vnir-upload", response_model=VNIRResponse)
async def vnir_upload(
    plant_id: int = Form(...),
    image: UploadFile = File(...),
    field_id: int | None = Form(None),
    crop_id: int | None = Form(None),
    user: UserRecord = Depends(require_user),
) -> VNIRResponse:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")

    store = field_store_for_user(user)
    plant = store.get_plant(plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")

    pil_image = load_image_from_bytes(data)
    pipeline = get_vnir_pipeline()

    history_ratios = store.get_vnir_ratios(plant_id)
    stats, hsv_image, vnir_image = pipeline.process_image(pil_image, plant["name"], history_ratios)

    store.add_vnir_reading(plant_id, stats.ratio, stats.avg_g, stats.avg_vnir, stats.status)

    store.add_event(
        event_type="vnir",
        field_id=field_id,
        crop_id=crop_id or plant["crop_id"],
        plant_id=plant_id,
        payload={
            "plant_name": plant["name"],
            "status": stats.status,
            "leaf_state": stats.leaf_state,
            "ratio": stats.ratio,
            "vs_baseline": stats.vs_baseline,
            "vs_global": stats.vs_global,
            "vs_rolling": stats.vs_rolling,
            "vs_prev_checkpoint": stats.vs_prev_checkpoint,
        },
    )

    return VNIRResponse(
        plant_id=str(plant_id),
        leaf_state=stats.leaf_state,
        status=stats.status,
        avg_green=stats.avg_g,
        avg_vnir=stats.avg_vnir,
        ratio=stats.ratio,
        baseline=stats.baseline,
        rolling_avg=stats.rolling_avg,
        prev_checkpoint_avg=stats.prev_checkpoint_avg,
        global_avg=stats.global_avg,
        vs_baseline=stats.vs_baseline,
        vs_global=stats.vs_global,
        vs_rolling=stats.vs_rolling,
        vs_prev_checkpoint=stats.vs_prev_checkpoint,
        hsv_image_base64=image_to_base64(hsv_image),
        vnir_image_base64=image_to_base64(vnir_image),
    )
