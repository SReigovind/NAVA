"""Pydantic schemas — split by domain for maintainability."""

from .auth import AuthLoginRequest, AuthRegisterRequest, AuthResponse, UserResponse
from .chat import (
    ChatClearRequest, ChatClearResponse, ChatHistoryMessage,
    ChatHistoryRequest, ChatHistoryResponse, ChatRequest,
    ChatResponse, ChatSummaryRequest, ChatSummaryResponse,
)
from .diagnose import DiagnoseResponse, VNIRPlantsResponse, VNIRResponse
from .events import EventListResponse, EventResponse
from .fields import (
    CropContextRequest, CropContextResponse, CropCreateRequest,
    CropListResponse, CropResponse, CropUpdateRequest,
    FieldContextRequest, FieldContextResponse, FieldCreateRequest,
    FieldListResponse, FieldResponse, FieldUpdateRequest,
    PlantCreateRequest, PlantListResponse, PlantResponse,
)

__all__ = [
    "AuthLoginRequest", "AuthRegisterRequest", "AuthResponse",
    "ChatClearRequest", "ChatClearResponse", "ChatHistoryMessage",
    "ChatHistoryRequest", "ChatHistoryResponse", "ChatRequest",
    "ChatResponse", "ChatSummaryRequest", "ChatSummaryResponse",
    "CropContextRequest", "CropContextResponse", "CropCreateRequest",
    "CropListResponse", "CropResponse", "CropUpdateRequest",
    "DiagnoseResponse", "EventListResponse", "EventResponse",
    "FieldContextRequest", "FieldContextResponse", "FieldCreateRequest",
    "FieldListResponse", "FieldResponse", "FieldUpdateRequest",
    "PlantCreateRequest", "PlantListResponse", "PlantResponse",
    "UserResponse", "VNIRPlantsResponse", "VNIRResponse",
]
