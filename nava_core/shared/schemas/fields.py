"""Field and crop schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class FieldCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    location: Optional[str] = Field(None, max_length=300)
    area: Optional[str] = Field(None, max_length=100)
    soil_type: Optional[str] = Field(None, max_length=100)
    shared_context: Optional[str] = Field(None, max_length=2000)


class FieldUpdateRequest(BaseModel):
    field_id: int
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    location: Optional[str] = Field(None, max_length=300)
    area: Optional[str] = Field(None, max_length=100)
    soil_type: Optional[str] = Field(None, max_length=100)


class FieldResponse(BaseModel):
    id: int
    name: str
    location: Optional[str] = None
    area: Optional[str] = None
    soil_type: Optional[str] = None
    shared_context: Optional[str] = None
    field_notes: Optional[str] = None
    created_at: Optional[str] = None


class FieldListResponse(BaseModel):
    fields: list[FieldResponse]


class FieldContextRequest(BaseModel):
    field_id: int
    shared_context: str = Field(max_length=2000)


class FieldContextResponse(BaseModel):
    field_id: int
    shared_context: Optional[str] = None


class CropCreateRequest(BaseModel):
    field_id: int
    name: str = Field(min_length=1, max_length=200)
    variety: Optional[str] = Field(None, max_length=200)
    season: Optional[str] = Field(None, max_length=100)
    stage: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)


class CropUpdateRequest(BaseModel):
    crop_id: int
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    variety: Optional[str] = Field(None, max_length=200)
    season: Optional[str] = Field(None, max_length=100)
    stage: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)


class CropResponse(BaseModel):
    id: int
    field_id: int
    name: str
    variety: Optional[str] = None
    season: Optional[str] = None
    stage: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


class CropListResponse(BaseModel):
    crops: list[CropResponse]


class CropContextRequest(BaseModel):
    crop_id: int
    notes: str = Field(max_length=2000)


class CropContextResponse(BaseModel):
    crop_id: int
    notes: Optional[str] = None


class PlantCreateRequest(BaseModel):
    crop_id: int
    name: str = Field(min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class PlantResponse(BaseModel):
    id: int
    crop_id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[str] = None


class PlantListResponse(BaseModel):
    plants: list[PlantResponse]
