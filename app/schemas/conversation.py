"""
Conversation and ConversationMessage Pydantic Schemas.
Enforces required storage fields, supported roles, and JSON serialization.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# Supported roles per requirements: user, assistant, tool, system
SupportedRole = Literal["user", "assistant", "tool", "system"]


class ConversationMessageBase(BaseModel):
    role: SupportedRole
    content: str
    message_type: str = "text"
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta_data"),
    )


class ConversationMessageCreate(ConversationMessageBase):
    """Payload for creating a new conversation message."""
    pass


class ConversationMessageResponse(BaseModel):
    """
    Stored message representation.
    Guarantees:
    - conversation_id
    - user_id
    - role
    - content
    - message_type
    - created_at
    - metadata
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    conversation_id: UUID
    user_id: UUID
    role: str
    content: str
    message_type: str = "text"
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta_data"),
    )
    created_at: datetime


class ConversationCreate(BaseModel):
    """Payload to create a new conversation."""
    title: Optional[str] = "New Conversation"


class ConversationUpdate(BaseModel):
    """Payload to update an existing conversation."""
    title: Optional[str] = None


class ConversationSummaryResponse(BaseModel):
    """Summary item for GET /conversations listing."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message_preview: Optional[str] = None
    last_message_at: Optional[datetime] = None


class ConversationDetailResponse(BaseModel):
    """Detailed conversation object for GET /conversations/:id."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    messages: List[ConversationMessageResponse] = []


class DeleteConversationResponse(BaseModel):
    """Confirmation payload for DELETE /conversations/:id."""
    success: bool = True
    message: str = "Conversation deleted successfully"
    id: UUID
