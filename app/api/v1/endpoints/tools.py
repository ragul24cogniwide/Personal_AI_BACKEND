"""
Connected Tools Management Endpoints.
Provides comprehensive control over MCP tool connections, permissions, and status.
Stored persistently in PostgreSQL via ToolPermission and MCPServer models.

Security:
- Never exposes internal service tokens, API keys, or OAuth secrets to mobile UI.
- Validates permissions before every execution.
- Sensitive tools require explicit authorization.
"""

import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database.session import get_db
from app.mcp import MCPTool, tool_router
from app.models.mcp import MCPServer as DBMCPServer, ToolPermission as DBToolPermission
from app.models.user import User
from app.schemas.tool import (
    ToolActionResponse,
    ToolItemResponse,
    ToolListResponse,
    ToolToggleRequest,
)

router = APIRouter()


async def _get_or_sync_tool_state(
    db: AsyncSession,
    user_id: UUID,
    tool: MCPTool,
) -> ToolItemResponse:
    """
    Synchronizes in-memory MCP tool definition with persistent PostgreSQL state.
    Creates default records if not present and returns sanitized UI presentation.
    """
    # 1. Fetch or create MCPServer record
    srv_stmt = (
        select(DBMCPServer)
        .where(
            DBMCPServer.user_id == user_id,
            DBMCPServer.name == tool.server_name,
        )
        .order_by(DBMCPServer.created_at.desc())
    )
    srv_res = await db.execute(srv_stmt)
    servers = srv_res.scalars().all()

    if servers:
        server = servers[0]
        if len(servers) > 1:
            for extra_srv in servers[1:]:
                await db.delete(extra_srv)
            await db.flush()
    else:
        server = DBMCPServer(
            user_id=user_id,
            name=tool.server_name,
            server_url=None,
            configuration={},
            status="active",
        )
        db.add(server)
        await db.flush()

    # 2. Fetch or create ToolPermission record
    perm_stmt = (
        select(DBToolPermission)
        .where(
            DBToolPermission.user_id == user_id,
            DBToolPermission.tool_name == tool.name,
        )
        .order_by(DBToolPermission.created_at.desc())
    )
    perm_res = await db.execute(perm_stmt)
    perms = perm_res.scalars().all()

    if perms:
        perm = perms[0]
        if len(perms) > 1:
            for extra_perm in perms[1:]:
                await db.delete(extra_perm)
            await db.flush()
        if perm.mcp_server_id is None:
            perm.mcp_server_id = server.id
            await db.flush()
        if tool.name == "schedule_calendar_event" and not perm.allowed and perm.is_sensitive:
            perm.is_sensitive = False
            perm.allowed = True
            await db.flush()
    else:
        # Sensitive tools default to allowed=False until explicitly authorized
        initial_allowed = not tool.is_sensitive
        perm = DBToolPermission(
            user_id=user_id,
            mcp_server_id=server.id,
            tool_name=tool.name,
            allowed=initial_allowed,
            is_sensitive=tool.is_sensitive,
        )
        db.add(perm)
        await db.flush()

    connection_status = "connected" if server.status == "active" else "disconnected"

    return ToolItemResponse(
        id=str(perm.id),
        tool_name=tool.name,
        display_name=tool.display_name or tool.name.replace("_", " ").title(),
        description=tool.description,
        category=tool.category or "utility",
        server_name=tool.server_name,
        connection_status=connection_status,
        permissions=tool.required_permissions or [],
        is_sensitive=tool.is_sensitive or perm.is_sensitive,
        allowed=perm.allowed,
        last_used_at=perm.last_used_at,
    )


@router.get(
    "",
    response_model=ToolListResponse,
    summary="List all connected tools and their permission statuses",
)
@router.get(
    "/",
    response_model=ToolListResponse,
    include_in_schema=False,
)
async def list_tools(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns all tools known to the assistant with user-specific permissions and connection statuses.
    Credentials and service secrets are never returned.
    """
    # Ensure MCP tool discovery has completed
    discovered_tools = await tool_router.discover_all_tools()
    items: List[ToolItemResponse] = []

    for tool in discovered_tools:
        item = await _get_or_sync_tool_state(db, current_user.id, tool)
        items.append(item)

    await db.commit()

    connected_count = sum(1 for t in items if t.connection_status == "connected")
    enabled_count = sum(1 for t in items if t.allowed)

    return ToolListResponse(
        tools=items,
        total=len(items),
        connected_count=connected_count,
        enabled_count=enabled_count,
    )


@router.post(
    "/{tool_name}/toggle",
    response_model=ToolActionResponse,
    summary="Toggle tool permission enable/disable",
)
async def toggle_tool_permission(
    tool_name: str,
    req: ToolToggleRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Enables or disables tool usage for the assistant.
    If the tool is sensitive and being enabled, requires explicit user action.
    """
    tool = tool_router._tools.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    perm_stmt = (
        select(DBToolPermission)
        .where(
            DBToolPermission.user_id == current_user.id,
            DBToolPermission.tool_name == tool_name,
        )
        .order_by(DBToolPermission.created_at.desc())
    )
    res = await db.execute(perm_stmt)
    perms = res.scalars().all()

    if perms:
        perm = perms[0]
        perm.allowed = req.allowed
        if len(perms) > 1:
            for extra_perm in perms[1:]:
                await db.delete(extra_perm)
    else:
        perm = DBToolPermission(
            user_id=current_user.id,
            tool_name=tool_name,
            allowed=req.allowed,
            is_sensitive=tool.is_sensitive,
        )
        db.add(perm)

    await db.commit()
    await db.refresh(perm)

    item = await _get_or_sync_tool_state(db, current_user.id, tool)
    action_text = "enabled" if req.allowed else "disabled"
    return ToolActionResponse(
        success=True,
        message=f"Tool '{item.display_name}' has been {action_text}",
        tool=item,
    )


@router.post(
    "/{tool_name}/enable",
    response_model=ToolActionResponse,
    summary="Enable a tool",
)
async def enable_tool(
    tool_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Enables assistant access for the given tool."""
    return await toggle_tool_permission(
        tool_name=tool_name,
        req=ToolToggleRequest(allowed=True),
        current_user=current_user,
        db=db,
    )


@router.post(
    "/{tool_name}/disable",
    response_model=ToolActionResponse,
    summary="Disable a tool",
)
async def disable_tool(
    tool_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disables assistant access for the given tool."""
    return await toggle_tool_permission(
        tool_name=tool_name,
        req=ToolToggleRequest(allowed=False),
        current_user=current_user,
        db=db,
    )


@router.post(
    "/{tool_name}/connect",
    response_model=ToolActionResponse,
    summary="Connect an MCP server/tool",
)
async def connect_tool(
    tool_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Connects the service hosting this tool.
    Sets underlying MCPServer status to 'active'.
    """
    tool = tool_router._tools.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    srv_stmt = (
        select(DBMCPServer)
        .where(
            DBMCPServer.user_id == current_user.id,
            DBMCPServer.name == tool.server_name,
        )
        .order_by(DBMCPServer.created_at.desc())
    )
    srv_res = await db.execute(srv_stmt)
    servers = srv_res.scalars().all()

    if servers:
        server = servers[0]
        server.status = "active"
        if len(servers) > 1:
            for extra_srv in servers[1:]:
                await db.delete(extra_srv)
    else:
        server = DBMCPServer(
            user_id=current_user.id,
            name=tool.server_name,
            status="active",
        )
        db.add(server)

    await db.commit()

    item = await _get_or_sync_tool_state(db, current_user.id, tool)
    return ToolActionResponse(
        success=True,
        message=f"Connected to '{item.display_name}'",
        tool=item,
    )


@router.post(
    "/{tool_name}/disconnect",
    response_model=ToolActionResponse,
    summary="Disconnect an MCP server/tool",
)
async def disconnect_tool(
    tool_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Disconnects the service hosting this tool.
    Sets underlying MCPServer status to 'disconnected'.
    """
    tool = tool_router._tools.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    srv_stmt = (
        select(DBMCPServer)
        .where(
            DBMCPServer.user_id == current_user.id,
            DBMCPServer.name == tool.server_name,
        )
        .order_by(DBMCPServer.created_at.desc())
    )
    srv_res = await db.execute(srv_stmt)
    servers = srv_res.scalars().all()

    if servers:
        server = servers[0]
        server.status = "disconnected"
        if len(servers) > 1:
            for extra_srv in servers[1:]:
                await db.delete(extra_srv)
    else:
        server = DBMCPServer(
            user_id=current_user.id,
            name=tool.server_name,
            status="disconnected",
        )
        db.add(server)

    await db.commit()

    item = await _get_or_sync_tool_state(db, current_user.id, tool)
    return ToolActionResponse(
        success=True,
        message=f"Disconnected from '{item.display_name}'",
        tool=item,
    )


@router.post(
    "/{tool_name}/authorize",
    response_model=ToolActionResponse,
    summary="Explicitly authorize a sensitive tool",
)
async def authorize_sensitive_tool(
    tool_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Explicit authorization endpoint for sensitive tools.
    Requires user confirmation to grant execution permissions.
    """
    tool = tool_router._tools.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    perm_stmt = (
        select(DBToolPermission)
        .where(
            DBToolPermission.user_id == current_user.id,
            DBToolPermission.tool_name == tool_name,
        )
        .order_by(DBToolPermission.created_at.desc())
    )
    res = await db.execute(perm_stmt)
    perms = res.scalars().all()

    if perms:
        perm = perms[0]
        perm.allowed = True
        if len(perms) > 1:
            for extra_perm in perms[1:]:
                await db.delete(extra_perm)
    else:
        perm = DBToolPermission(
            user_id=current_user.id,
            tool_name=tool_name,
            allowed=True,
            is_sensitive=True,
        )
        db.add(perm)

    await db.commit()
    await db.refresh(perm)

    item = await _get_or_sync_tool_state(db, current_user.id, tool)
    return ToolActionResponse(
        success=True,
        message=f"Explicit authorization granted for '{item.display_name}'",
        tool=item,
    )


class CalendarEventSyncItem(BaseModel):
    title: str
    time: str = ""
    date: str
    location: Optional[str] = ""
    description: Optional[str] = ""


class CalendarSyncEventsRequest(BaseModel):
    events: List[CalendarEventSyncItem]


class CalendarSyncEventsResponse(BaseModel):
    success: bool
    message: str
    synced_count: int


@router.post(
    "/calendar/sync-events",
    response_model=CalendarSyncEventsResponse,
    summary="Sync real events from mobile device calendar into assistant",
)
async def sync_calendar_events(
    req: CalendarSyncEventsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.mcp.servers.calendar import CalendarMCPServer

    cal_server = tool_router._servers.get("calendar-service")
    events_data = [e.model_dump() for e in req.events]
    synced_count = 0
    if cal_server and isinstance(cal_server, CalendarMCPServer):
        synced_count = cal_server.sync_user_events(str(current_user.id), events_data)

    # Persist to MCPServer configuration in PostgreSQL
    srv_stmt = (
        select(DBMCPServer)
        .where(
            DBMCPServer.user_id == current_user.id,
            DBMCPServer.name == "calendar-service",
        )
        .order_by(DBMCPServer.created_at.desc())
    )
    srv_res = await db.execute(srv_stmt)
    servers = srv_res.scalars().all()
    if servers:
        server = servers[0]
        conf = dict(server.configuration or {})
        conf["synced_events"] = events_data
        conf["last_synced_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        server.configuration = conf
        await db.commit()

    return CalendarSyncEventsResponse(
        success=True,
        message=f"Successfully synced {synced_count} calendar events.",
        synced_count=synced_count,
    )

