"""
Memories domain model with PostgreSQL + pgvector support.
Table: memories
"""

from typing import Any, Dict, List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

# 12 Supported memory categories required by the system
MEMORY_CATEGORIES = (
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
)


class Memory(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Long-term personal memory record stored in PostgreSQL with pgvector embeddings.
    Distinguished from temporary conversation context.
    """
    __tablename__ = "memories"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    category = Column(String(100), default="other", index=True, nullable=False)
    content = Column(Text, nullable=False)
    importance = Column(Float, default=0.5, nullable=False)
    source = Column(String(100), default="manual", nullable=False)
    meta_data = Column("metadata", JSONB, default=dict, nullable=False)

    # 1536-dimensional embedding vector stored in pgvector
    embedding = Column(Vector(1536), nullable=True)

    user = relationship("User", back_populates="memories")

    __table_args__ = (
        Index("ix_memories_user_id_category", "user_id", "category"),
        Index("ix_memories_user_id_created_at", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Memory id={self.id} user_id={self.user_id} category={self.category} importance={self.importance}>"
