"""
User Preferences and Settings schemas.
Supports proactivity configuration: 'low', 'medium', 'high'.
"""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


ProactivityLevel = Literal["low", "medium", "high"]


class UserPreferenceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    theme: str = "dark"
    language: str = "en"
    voice: str = "alloy"
    assistant_tone: str = "friendly"
    verbosity: str = "balanced"
    proactivity: ProactivityLevel = "medium"
    timezone: str = "UTC"
    created_at: datetime
    updated_at: datetime


class UserPreferenceUpdate(BaseModel):
    proactivity: Optional[ProactivityLevel] = Field(
        None,
        description="Assistant proactivity level: 'low' (reactive), 'medium' (balanced), 'high' (proactive)",
    )
    assistant_tone: Optional[str] = Field(None, max_length=100)
    verbosity: Optional[str] = Field(None, max_length=50)
    theme: Optional[str] = Field(None, max_length=50)
    language: Optional[str] = Field(None, max_length=50)
    voice: Optional[str] = Field(None, max_length=50)
    timezone: Optional[str] = Field(None, max_length=100)


class ProactivityUpdateRequest(BaseModel):
    proactivity: ProactivityLevel = Field(
        ...,
        description="Proactivity level: 'low', 'medium', or 'high'",
    )


class UserGoalCreate(BaseModel):
    category: str = Field("general", max_length=100)
    title: str = Field(..., max_length=255)
    description: Optional[str] = None
    priority: str = Field("medium", max_length=50)
    status: str = Field("in_progress", max_length=50)


class UserGoalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    category: str
    title: str
    description: Optional[str] = None
    priority: str
    status: str
    created_at: datetime
    updated_at: datetime


class UserHabitCreate(BaseModel):
    category: str = Field("general", max_length=100)
    name: str = Field(..., max_length=255)
    frequency: str = Field("daily", max_length=50)
    description: Optional[str] = None


class UserHabitRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    category: str
    name: str
    frequency: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
