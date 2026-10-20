"""
Onboarding schemas for structured conversational profile ingestion.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.auth import UserRead


class OnboardingGoalInput(BaseModel):
    title: str = Field(..., max_length=255)
    category: str = Field("general", max_length=100)
    description: Optional[str] = None
    priority: str = Field("medium", max_length=50)


class OnboardingCompleteRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    preferred_name: Optional[str] = Field(None, max_length=100)
    occupation: Optional[str] = Field(None, max_length=255)
    interests: List[str] = Field(default_factory=list)
    hobbies: List[str] = Field(default_factory=list)
    goals: List[OnboardingGoalInput] = Field(default_factory=list)
    communication_style: Optional[str] = Field(None, max_length=100)
    assistant_tone: Optional[str] = Field(None, max_length=50)
    fitness_interests: List[str] = Field(default_factory=list)
    learning_interests: List[str] = Field(default_factory=list)
    work_interests: List[str] = Field(default_factory=list)
    daily_routine: Dict[str, Any] = Field(default_factory=dict)
    timezone: Optional[str] = Field(None, max_length=100)


class OnboardingCompleteResponse(BaseModel):
    success: bool = True
    message: str = "Onboarding completed successfully"
    user: UserRead
