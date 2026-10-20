"""
Pydantic schemas for Authentication and User state.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# Request Schemas
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
    name: Optional[str] = Field(None, max_length=100, description="Full or preferred name")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenRefresh(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


# Embedded Entity Schemas for User Profile, Preferences & Settings
class UserProfileSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: Optional[str] = None
    preferred_name: Optional[str] = None
    occupation: Optional[str] = None
    bio: Optional[str] = None
    communication_style: Optional[str] = None
    personality_preferences: Dict[str, Any] = Field(default_factory=dict)
    interests: list[str] = Field(default_factory=list)
    hobbies: list[str] = Field(default_factory=list)
    fitness_interests: list[str] = Field(default_factory=list)
    learning_interests: list[str] = Field(default_factory=list)
    work_interests: list[str] = Field(default_factory=list)
    daily_routine: Dict[str, Any] = Field(default_factory=dict)


class UserPreferencesSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    theme: str = "dark"
    language: str = "en"
    voice: str = "alloy"
    assistant_tone: str = "friendly"
    verbosity: str = "balanced"
    proactivity: str = "medium"
    timezone: str = "UTC"


class AssistantSettingsSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    personality: Optional[str] = None
    voice_settings: Dict[str, Any] = Field(default_factory=dict)
    proactivity_settings: Dict[str, Any] = Field(default_factory=dict)
    memory_settings: Dict[str, Any] = Field(default_factory=dict)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    email: EmailStr
    is_onboarded: bool = False
    onboarding_completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    profile: Optional[UserProfileSummary] = None
    preferences: Optional[UserPreferencesSummary] = None
    assistant_settings: Optional[AssistantSettingsSummary] = None


# Response Schemas
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserRead


class TokenRefreshResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None
