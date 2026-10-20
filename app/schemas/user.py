"""
User and Profile schemas.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, ConfigDict


class UserBase(BaseModel):
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserProfileBase(BaseModel):
    full_name: Optional[str] = None
    display_name: Optional[str] = None
    title: Optional[str] = None
    timezone: str = "UTC"
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


class UserProfileRead(UserProfileBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime


class UserRead(UserBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime
    profile: Optional[UserProfileRead] = None
