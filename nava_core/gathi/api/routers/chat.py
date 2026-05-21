"""Chat router."""

from __future__ import annotations
from fastapi import APIRouter, Depends, BackgroundTasks, Request

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
def chat(request: Request, body: ChatRequest, bg_tasks: BackgroundTasks, user: UserRecord = Depends(require_user)) -> ChatResponse:
    service = chat_service_for_user(user, request)
    result = service.chat(body.message, body.session_id, field_id=body.field_id, crop_id=body.crop_id)

    # Trigger background summary if threshold is met, preventing latency for the user
    bg_tasks.add_task(service._summarize_if_needed, result.session_id)

    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        error=result.error,
        rag_used=result.rag_used,
        rag_chunk_count=result.rag_chunk_count,
        rag_chunks=result.rag_chunks,
    )


@router.post("/clear", response_model=ChatClearResponse)
def clear(request: Request, body: ChatClearRequest, user: UserRecord = Depends(require_user)) -> ChatClearResponse:
    chat_service_for_user(user, request).clear_session(body.session_id)
    return ChatClearResponse(session_id=body.session_id, status="cleared")


@router.post("/history", response_model=ChatHistoryResponse)
def history(request: Request, body: ChatHistoryRequest, user: UserRecord = Depends(require_user)) -> ChatHistoryResponse:
    messages = chat_service_for_user(user, request).get_history(body.session_id, limit=body.limit)
    return ChatHistoryResponse(session_id=body.session_id, messages=messages)


@router.post("/summary", response_model=ChatSummaryResponse)
def summary(request: Request, body: ChatSummaryRequest, user: UserRecord = Depends(require_user)) -> ChatSummaryResponse:
    s = chat_service_for_user(user, request).get_summary_display(body.session_id)
    return ChatSummaryResponse(session_id=body.session_id, summary=s)
