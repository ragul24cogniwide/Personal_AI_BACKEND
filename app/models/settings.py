"""
Assistant System Settings domain model.
Table: assistant_settings
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AssistantSetting(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "assistant_settings"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    personality = Column(Text, default="empathetic, proactive, concise, and helpful", nullable=True)
    voice_settings = Column(JSONB, default=dict, nullable=False)
    proactivity_settings = Column(JSONB, default=dict, nullable=False)
    memory_settings = Column(JSONB, default=dict, nullable=False)

    user = relationship("User", back_populates="assistant_settings")
