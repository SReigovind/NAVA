"""Event schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class EventResponse(BaseModel):
    id: int
    event_type: str
    field_id: Optional[int] = None
    crop_id: Optional[int] = None
    plant_id: Optional[int] = None
    payload: Optional[dict] = None
    created_at: Optional[str] = None



class EventListResponse(BaseModel):
    events: list[EventResponse]
