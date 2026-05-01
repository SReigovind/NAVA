from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from nava_core.mizhi.detection import EfficientNetB0Predictor
from nava_core.mizhi.vnir import VNIRPipeline
from nava_core.shared.config import get_settings
from nava_core.shared.schemas import DiagnoseResponse, VNIRPlantsResponse, VNIRResponse
from nava_core.shared.utils import image_to_base64, load_image_from_bytes

app = FastAPI(title="NAVA API", version="0.1.0")
UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"
VNIR_HISTORY_DIR = Path(__file__).resolve().parents[3] / "logs" / "vnir"


@lru_cache
def _predictor() -> EfficientNetB0Predictor:
    settings = get_settings()
    return EfficientNetB0Predictor(
        model_path=settings.efficientnet_model_path,
        labels_path=settings.efficientnet_labels_path,
        device=settings.torch_device,
        confidence_threshold=settings.confidence_threshold,
    )


@lru_cache
def _vnir_pipeline() -> VNIRPipeline:
    settings = get_settings()
    return VNIRPipeline(
        model_path=settings.vnir_model_path,
        stress_threshold_pct=settings.vnir_stress_threshold_pct,
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def diagnose(image: UploadFile = File(...)) -> DiagnoseResponse:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")

    pil_image = load_image_from_bytes(data)
    predictor = _predictor()
    result = predictor.predict(pil_image)

    if result.reliability == "UNRELIABLE":
        return DiagnoseResponse(
            class_label=result.class_label,
            class_index=result.class_index,
            confidence=result.confidence,
            reliability=result.reliability,
        )

    result, cam_image = predictor.predict_with_cam(pil_image)
    return DiagnoseResponse(
        class_label=result.class_label,
        class_index=result.class_index,
        confidence=result.confidence,
        reliability=result.reliability,
        original_image_base64=image_to_base64(pil_image),
        gradcam_image_base64=image_to_base64(cam_image),
    )


@app.get("/api/vnir-plants", response_model=VNIRPlantsResponse)
def list_vnir_plants() -> VNIRPlantsResponse:
    VNIR_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    plant_ids = []
    for file in VNIR_HISTORY_DIR.glob("*_history.csv"):
        name = file.stem.replace("_history", "")
        plant_ids.append(name)
    plant_ids = sorted(set(plant_ids))
    return VNIRPlantsResponse(plant_ids=plant_ids)


@app.post("/api/vnir-clear")
def clear_vnir_history(plant_id: str = Form(...)) -> dict:
    pipeline = _vnir_pipeline()
    pipeline.delete_history(plant_id)
    return {"status": "cleared", "plant_id": plant_id}


@app.post("/api/vnir-upload", response_model=VNIRResponse)
async def vnir_upload(
    plant_id: str = Form(...),
    image: UploadFile = File(...),
) -> VNIRResponse:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image payload")

    pil_image = load_image_from_bytes(data)
    pipeline = _vnir_pipeline()
    stats, hsv_image, vnir_image = pipeline.process_image(pil_image, plant_id)

    return VNIRResponse(
        plant_id=plant_id,
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
