from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DiagnoseResponse(BaseModel):
    class_label: str
    class_index: int
    confidence: float
    reliability: str
    original_image_base64: Optional[str] = None
    gradcam_image_base64: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    error: Optional[str] = None


class ChatClearRequest(BaseModel):
    session_id: str


class ChatClearResponse(BaseModel):
    session_id: str
    status: str


class ChatSummaryRequest(BaseModel):
    session_id: str


class ChatSummaryResponse(BaseModel):
    session_id: str
    summary: Optional[str] = None


class VNIRResponse(BaseModel):
    plant_id: str
    leaf_state: str
    status: str
    avg_green: float
    avg_vnir: float
    ratio: float
    baseline: Optional[float] = None
    rolling_avg: Optional[float] = None
    prev_checkpoint_avg: Optional[float] = None
    global_avg: Optional[float] = None
    vs_baseline: Optional[float] = None
    vs_global: Optional[float] = None
    vs_rolling: Optional[float] = None
    vs_prev_checkpoint: Optional[float] = None
    hsv_image_base64: str
    vnir_image_base64: str


class VNIRPlantsResponse(BaseModel):
    plant_ids: list[str]
