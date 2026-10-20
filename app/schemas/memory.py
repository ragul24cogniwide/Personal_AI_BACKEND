"""
Pydantic Schemas for Long-Term Personal Memory.
Supports the 12 required memory categories, vector embeddings, and CRUD validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

# 12 Supported memory categories
MemoryCategory = Literal[
    "identity",
    "preferences",
    "personality",
    "goals",
    "habits",
    "fitness",
    "learning",
    "work",
    "projects",
    "routine",
    "important_events",
    "other",
]


class MemoryBase(BaseModel):
    category: MemoryCategory = Field(
        default="other",
        description="Category classifying the long-term memory",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Factual content of the memory",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Importance score from 0.0 (trivial) to 1.0 (vital)",
    )
    source: str = Field(
        default="manual",
        max_length=100,
        description="Source of the memory (e.g. conversation, manual, user, onboarding)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta_data"),
        description="Optional structured metadata attributes",
    )


class MemoryCreate(MemoryBase):
    """Payload to create a new personal memory."""
    pass


class MemoryUpdate(BaseModel):
    """Payload to update an existing personal memory."""
    category: Optional[MemoryCategory] = None
    content: Optional[str] = Field(None, min_length=1, max_length=10000)
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = Field(None, max_length=100)
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        validation_alias=AliasChoices("metadata", "meta_data"),
    )


class MemoryResponse(BaseModel):
    """Memory representation returned to clients."""
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: UUID
    user_id: UUID
    category: str
    content: str
    importance: float
    source: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("metadata", "meta_data"),
    )
    has_embedding: bool = False
    similarity: Optional[float] = None
    created_at: datetime
    updated_at: datetime


class MemoryListResponse(BaseModel):
    """Paginated list of memories with category aggregations."""
    items: List[MemoryResponse]
    total: int
    category_counts: Dict[str, int] = Field(default_factory=dict)


class DeleteMemoryResponse(BaseModel):
    deleted: bool
    id: UUID
    message: str = "Memory deleted successfully"


class MemorySearchRequest(BaseModel):
    """Request for semantic vector search over personal memories."""
    query: str = Field(..., min_length=1, description="Natural language search query")
    category: Optional[MemoryCategory] = None
    limit: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.4, ge=0.0, le=1.0)


class ExtractedMemoryItem(BaseModel):
    """Single memory item extracted from conversation context."""
    category: MemoryCategory
    content: str
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class MemoryExtractRequest(BaseModel):
    """Request to extract memories from conversation context or raw text."""
    conversation_id: Optional[UUID] = None
    text: Optional[str] = None
    auto_save: bool = Field(
        default=True,
        description="Whether to automatically store extracted memories in pgvector",
    )


class MemoryExtractResponse(BaseModel):
    """Result of memory extraction process."""
    extracted: List[ExtractedMemoryItem]
    saved_memories: List[MemoryResponse] = Field(default_factory=list)


# Backwards compatibility alias
MemoryRead = MemoryResponse
