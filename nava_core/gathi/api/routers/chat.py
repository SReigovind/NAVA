"""Chat router."""

from __future__ import annotations
from fastapi import APIRouter, Depends, BackgroundTasks

from nava_core.shared.schemas import (
    ChatClearRequest, ChatClearResponse,
    ChatHistoryRequest, ChatHistoryResponse,
    ChatRequest, ChatResponse,
    ChatSummaryRequest, ChatSummaryResponse,
)
from nava_core.shared.storage.user_store import UserRecord
from nava_core.gathi.api.deps import chat_service_for_user, require_user

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, bg_tasks: BackgroundTasks, user: UserRecord = Depends(require_user)) -> ChatResponse:
    service = chat_service_for_user(user)
    result = service.chat(request.message, request.session_id, field_id=request.field_id, crop_id=request.crop_id)
    
    # Trigger background summary if threshold is met, preventing latency for the user
    bg_tasks.add_task(service._summarize_if_needed, result.session_id)
    
    return ChatResponse(session_id=result.session_id, reply=result.reply, error=result.error)


@router.post("/clear", response_model=ChatClearResponse)
def clear(request: ChatClearRequest, user: UserRecord = Depends(require_user)) -> ChatClearResponse:
    chat_service_for_user(user).clear_session(request.session_id)
    return ChatClearResponse(session_id=request.session_id, status="cleared")


@router.post("/history", response_model=ChatHistoryResponse)
def history(request: ChatHistoryRequest, user: UserRecord = Depends(require_user)) -> ChatHistoryResponse:
    messages = chat_service_for_user(user).get_history(request.session_id, limit=request.limit)
    return ChatHistoryResponse(session_id=request.session_id, messages=messages)


@router.post("/summary", response_model=ChatSummaryResponse)
def summary(request: ChatSummaryRequest, user: UserRecord = Depends(require_user)) -> ChatSummaryResponse:
    s = chat_service_for_user(user).get_summary_display(request.session_id)
    return ChatSummaryResponse(session_id=request.session_id, summary=s)
