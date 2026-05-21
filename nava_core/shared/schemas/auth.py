"""Auth schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)


class AuthRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    onboarded: bool = False
    location: Optional[str] = None
    goals: Optional[str] = None
    created_at: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserResponse

class UpdateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)

class UpdatePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=128)
