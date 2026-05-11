"""Chat schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    field_id: Optional[int] = None
    crop_id: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    error: Optional[str] = None


class ChatClearRequest(BaseModel):
    session_id: str


class ChatClearResponse(BaseModel):
    session_id: str
    status: str


class ChatHistoryRequest(BaseModel):
    session_id: str
    limit: Optional[int] = None


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatHistoryMessage]


class ChatSummaryRequest(BaseModel):
    session_id: str


class ChatSummaryResponse(BaseModel):
    session_id: str
    summary: Optional[str] = None
