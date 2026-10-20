"""
Conversations and ConversationMessages domain models.
Tables: conversations, conversation_messages
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Conversation(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "conversations"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title = Column(String(255), default="New Conversation", nullable=False)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )


class ConversationMessage(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "conversation_messages"

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role = Column(String(30), nullable=False)  # user, assistant, system, tool
    content = Column(Text, nullable=False)
    message_type = Column(String(50), default="text", nullable=False)
    meta_data = Column("metadata", JSONB, default=dict, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    conversation = relationship("Conversation", back_populates="messages")
    user = relationship("User")

    @property
    def metadata_dict(self):
        """Helper to access JSONB metadata dict."""
        return self.meta_data

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')",
            name="check_conversation_message_role",
        ),
        Index("ix_conversation_messages_conversation_id_created_at", "conversation_id", "created_at"),
        Index("ix_conversation_messages_user_id", "user_id"),
    )

