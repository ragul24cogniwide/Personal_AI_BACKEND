"""
Model Context Protocol (MCP) domain models.
Tables: mcp_servers, tool_permissions
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class MCPServer(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "mcp_servers"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name = Column(String(100), nullable=False)
    server_url = Column(String(500), nullable=True)
    configuration = Column(JSONB, default=dict, nullable=False)
    status = Column(String(50), default="active", nullable=False)

    user = relationship("User", back_populates="mcp_servers")
    tool_permissions = relationship(
        "ToolPermission",
        back_populates="mcp_server",
        cascade="all, delete-orphan",
    )


class ToolPermission(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "tool_permissions"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    mcp_server_id = Column(
        UUID(as_uuid=True),
        ForeignKey("mcp_servers.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    tool_name = Column(String(100), nullable=False)
    allowed = Column(Boolean, default=True, nullable=False)
    is_sensitive = Column(Boolean, default=False, nullable=False)
    last_used_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="tool_permissions")
    mcp_server = relationship("MCPServer", back_populates="tool_permissions")

    __table_args__ = (
        Index("ix_tool_permissions_user_tool", "user_id", "tool_name"),
    )
