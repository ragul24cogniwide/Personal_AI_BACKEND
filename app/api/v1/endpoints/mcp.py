"""
MCP Discovery and Tool Management Endpoints.
Provides REST APIs for:
- Tool discovery across connected MCP Servers
- Registered server listing
- Tool execution audit logs (sanitized)
- Permission management
"""

from typing import Any, Dict, List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.mcp import (
    ExecutionStatus,
    MCPTool,
    PermissionLevel,
    ToolExecution,
    ToolPermission,
    tool_router,
)

router = APIRouter()


class ToolDiscoveryResponse(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    required_permissions: List[str]
    is_read_only: bool


class ServerInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    tools_count: int


class PermissionCreateRequest(BaseModel):
    user_id: Optional[UUID] = None
    tool_name: str
    permission_level: PermissionLevel = PermissionLevel.ALLOW
    allowed_scopes: List[str] = Field(default_factory=list)


class ExecuteToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[UUID] = None


@router.get("/tools", response_model=List[ToolDiscoveryResponse])
async def list_mcp_tools():
    """
    Discovery Endpoint: Returns all MCP tools registered across connected MCP servers.
    Only exposes public tool definitions and parameter schemas. Credentials remain private.
    """
    tools = await tool_router.discover_all_tools()
    return [
        ToolDiscoveryResponse(
            name=t.name,
            description=t.description,
            input_schema=t.input_schema,
            server_name=t.server_name,
            required_permissions=t.required_permissions,
            is_read_only=t.is_read_only,
        )
        for t in tools
    ]


@router.get("/servers", response_model=List[ServerInfoResponse])
async def list_mcp_servers():
    """
    Lists all connected MCP Servers and their current status.
    """
    servers = tool_router._servers.values()
    return [
        ServerInfoResponse(
            name=s.name,
            version=s.version,
            description=s.description,
            tools_count=len(s.list_tools()),
        )
        for s in servers
    ]


@router.get("/executions", response_model=List[ToolExecution])
async def list_recent_executions(limit: int = 50):
    """
    Audit Endpoint: Returns sanitized execution history. All credentials and secrets are redacted.
    """
    return tool_router.get_recent_executions(limit=limit)


@router.post("/permissions", status_code=status.HTTP_201_CREATED)
async def create_or_update_permission(req: PermissionCreateRequest):
    """
    Configures security access control permissions for a specific tool.
    """
    permission = ToolPermission(
        user_id=req.user_id,
        tool_name=req.tool_name,
        permission_level=req.permission_level,
        allowed_scopes=req.allowed_scopes,
    )
    tool_router.set_permission(permission)
    return {"status": "success", "message": f"Permission updated for tool '{req.tool_name}'"}


@router.post("/execute", response_model=ToolExecution)
async def execute_tool_direct(req: ExecuteToolRequest):
    """
    Direct tool execution endpoint for testing discovery and execution.
    Enforces the same ToolPermission checks as the Gemini voice pipeline.
    """
    execution = await tool_router.route_tool_execution(
        tool_name=req.tool_name,
        arguments=req.arguments,
        user_id=req.user_id,
    )
    return execution
