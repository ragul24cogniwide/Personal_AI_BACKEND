"""
Pydantic schemas for Connected Tools management.
Exposes tool metadata, connection status, permissions, and last used timestamps.
Never exposes credentials, API tokens, or secrets.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class ToolItemResponse(BaseModel):
    id: str = Field(..., description="Unique tool identifier")
    tool_name: str = Field(..., description="Identifier name of the tool, e.g. 'get_calendar_events'")
    display_name: str = Field(..., description="Human-friendly label, e.g. 'Calendar'")
    description: str = Field(..., description="Short explanation of what this tool does")
    category: str = Field("utility", description="Category grouping, e.g. 'calendar', 'automation'")
    server_name: str = Field(..., description="Underlying MCP server name")
    connection_status: str = Field("connected", description="'connected', 'disconnected', or 'requires_auth'")
    permissions: List[str] = Field(default_factory=list, description="Scopes or capability strings, e.g. ['calendar.read']")
    is_sensitive: bool = Field(False, description="True if tool requires explicit user confirmation/authorization")
    allowed: bool = Field(True, description="True if enabled by user in PostgreSQL permissions table")
    last_used_at: Optional[datetime] = Field(None, description="Timestamp when assistant last invoked this tool")


class ToolListResponse(BaseModel):
    tools: List[ToolItemResponse]
    total: int
    connected_count: int
    enabled_count: int


class ToolToggleRequest(BaseModel):
    allowed: bool = Field(..., description="True to enable, False to disable")


class ToolActionResponse(BaseModel):
    success: bool
    message: str
    tool: ToolItemResponse
