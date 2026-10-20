"""
MCP Tool Router.
Acts as the central coordination layer between Gemini and the MCP ecosystem:
1. Discovers tools across all registered MCP Servers via MCP Clients.
2. Formats Gemini tool declarations without exposing credentials.
3. Enforces strict permission validation before tool execution.
4. Routes tool invocations to the designated MCP Client/Server.
5. Captures and safely logs ToolExecution audit records (zero secret leakage).
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.ai.schemas import ToolCall
from app.mcp.client import MCPClient
from app.mcp.models import (
    ExecutionStatus,
    MCPTool,
    PermissionLevel,
    ToolExecution,
    ToolPermission,
)
from app.mcp.exceptions import (
    MCPAPIFailureError,
    MCPAuthenticationError,
    MCPError,
    MCPPermissionDeniedError,
    MCPTimeoutError,
    MCPUnavailableError,
)
from app.mcp.server import MCPServer
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.session import AsyncSessionLocal

try:
    from google.genai import types
except ImportError:
    types = None  # type: ignore

logger = logging.getLogger("app.mcp.router")


class ToolRouter:
    """
    Central Tool Router and Access Control Controller for MCP.
    Guarantees backend-controlled execution and credential isolation.
    """

    def __init__(self):
        self._servers: Dict[str, MCPServer] = {}
        self._clients: Dict[str, MCPClient] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._tool_to_client: Dict[str, MCPClient] = {}
        # Permissions mapping: user_id -> List[ToolPermission], None -> Global permissions
        self._permissions: Dict[Optional[str], List[ToolPermission]] = {}
        self._execution_history: List[ToolExecution] = []
        self._lock = asyncio.Lock()

    def register_server(self, server: MCPServer) -> MCPClient:
        """
        Registers an MCP Server and attaches an MCP Client for routing.
        """
        self._servers[server.name] = server
        client = MCPClient(server=server)
        self._clients[server.name] = client
        logger.info(f"[ToolRouter] Registered MCP server: '{server.name}'")
        return client

    def register_client(self, client: MCPClient) -> None:
        """Registers a standalone MCP Client."""
        self._clients[client.server_name] = client
        logger.info(f"[ToolRouter] Registered MCP client: '{client.server_name}'")

    async def discover_all_tools(self) -> List[MCPTool]:
        """
        Discovery process: Iterates through all connected MCP Clients and gathers
        their published tool definitions into the central registry.
        """
        async with self._lock:
            self._tools.clear()
            self._tool_to_client.clear()

            all_tools: List[MCPTool] = []
            for client in self._clients.values():
                server_tools = await client.discover_tools()
                for tool in server_tools:
                    self._tools[tool.name] = tool
                    self._tool_to_client[tool.name] = client
                    all_tools.append(tool)

            logger.info(f"[ToolRouter] Discovered {len(all_tools)} tools across {len(self._clients)} MCP servers")
            return all_tools

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Returns the cached MCPTool definition by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[MCPTool]:
        """Returns all currently cached tools."""
        return list(self._tools.values())

    def get_gemini_tools(self) -> List[types.Tool]:
        """
        Translates all discovered MCP tools into Gemini types.Tool declarations.
        CRITICAL SECURITY: Strictly passes name, description, and parameter schemas.
        Credentials and backend URLs are NEVER included in these declarations.
        """
        if types is None:
            return []

        declarations = []
        for tool in self._tools.values():
            try:
                declarations.append(tool.to_gemini_function_declaration())
            except Exception as exc:
                logger.error(f"[ToolRouter] Failed to convert tool '{tool.name}' to Gemini declaration: {exc}")

        if not declarations:
            return []

        return [types.Tool(function_declarations=declarations)]

    def set_permission(self, permission: ToolPermission) -> None:
        """
        Adds or updates a security permission rule for tool execution.
        """
        user_key = str(permission.user_id) if permission.user_id else "__global__"
        if user_key not in self._permissions:
            self._permissions[user_key] = []
        
        # Replace existing rule for same tool if present
        self._permissions[user_key] = [
            p for p in self._permissions[user_key] if p.tool_name != permission.tool_name
        ]
        self._permissions[user_key].append(permission)
        logger.info(
            f"[ToolRouter] Updated permission for '{permission.tool_name}' "
            f"(user={user_key}, level={permission.permission_level.value})"
        )

    async def validate_permission(
        self,
        user_id: Optional[UUID],
        tool: MCPTool,
        arguments: Dict[str, Any],
        db: Optional[AsyncSession] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates permission rules before any tool execution takes place.
        Order of evaluation:
        1. PostgreSQL 'tool_permissions' table (user-specific persistent permissions)
        2. PostgreSQL 'mcp_servers' table (server active vs disconnected status)
        3. In-memory / Global permission rules (for testing & policy overrides)
        4. Sensitive tool check (sensitive tools require explicit authorization)
        5. Default safe policy (read-only tools ALLOW by default)
        """
        # 1. Check in-memory rules first (allows explicit test or runtime policy overrides like DENY)
        user_key = str(user_id) if user_id else "__global__"
        candidate_rules: List[ToolPermission] = []
        if user_key in self._permissions:
            candidate_rules.extend(self._permissions[user_key])
        if "__global__" in self._permissions:
            candidate_rules.extend(self._permissions["__global__"])

        for rule in candidate_rules:
            if rule.tool_name in (tool.name, "*"):
                if rule.permission_level == PermissionLevel.DENY:
                    return False, f"Tool '{tool.name}' is explicitly denied by security policy"
                elif rule.permission_level == PermissionLevel.REQUIRE_CONFIRMATION:
                    return False, f"Tool '{tool.name}' requires explicit user confirmation"
                elif rule.permission_level == PermissionLevel.ALLOW:
                    if tool.required_permissions:
                        for scope in tool.required_permissions:
                            if scope not in rule.allowed_scopes and "*" not in rule.allowed_scopes:
                                return False, f"Missing required permission scope '{scope}'"
                    return True, None

        # 2. Check PostgreSQL Persistent Permissions
        if user_id:
            try:
                from app.models.mcp import ToolPermission as DBToolPermission, MCPServer as DBMCPServer

                async def _query_db(sess: AsyncSession):
                    perm_stmt = (
                        select(DBToolPermission)
                        .where(
                            DBToolPermission.user_id == user_id,
                            DBToolPermission.tool_name == tool.name,
                        )
                        .order_by(DBToolPermission.created_at.desc())
                    )
                    perm_res = await sess.execute(perm_stmt)
                    perm_row = perm_res.scalars().first()

                    srv_status = "active"
                    if perm_row and perm_row.mcp_server_id:
                        srv_stmt = select(DBMCPServer.status).where(DBMCPServer.id == perm_row.mcp_server_id)
                        srv_res = await sess.execute(srv_stmt)
                        status_val = srv_res.scalars().first()
                        if status_val:
                            srv_status = status_val
                    else:
                        srv_stmt = (
                            select(DBMCPServer.status)
                            .where(
                                DBMCPServer.user_id == user_id,
                                DBMCPServer.name == tool.server_name,
                            )
                            .order_by(DBMCPServer.created_at.desc())
                        )
                        srv_res = await sess.execute(srv_stmt)
                        status_val = srv_res.scalars().first()
                        if status_val:
                            srv_status = status_val

                    return perm_row, srv_status

                if db:
                    db_perm, srv_status = await _query_db(db)
                else:
                    async with AsyncSessionLocal() as session:
                        db_perm, srv_status = await _query_db(session)

                # Check server connection status
                if srv_status == "disconnected":
                    return False, f"Connected service '{tool.server_name}' is currently disconnected"

                # Check persistent permission
                if db_perm is not None:
                    if tool.name == "schedule_calendar_event" and not tool.is_sensitive and db_perm.is_sensitive:
                        db_perm.is_sensitive = False
                        db_perm.allowed = True
                        if db:
                            db.add(db_perm)
                        return True, None
                    if (tool.is_sensitive or db_perm.is_sensitive) and not db_perm.allowed:
                        return False, f"Sensitive tool '{tool.name}' requires explicit authorization"
                    if not db_perm.allowed:
                        return False, f"Tool '{tool.name}' is disabled in Connected Tools"
                    return True, None
                else:
                    # No record yet in PostgreSQL:
                    if tool.name in ("schedule_calendar_event", "get_calendar_events") or tool.category == "calendar":
                        asyncio.create_task(self._create_default_db_permission(user_id, tool.name, is_sensitive=False))
                        return True, None
                    if tool.is_sensitive:
                        return False, f"Sensitive tool '{tool.name}' requires explicit authorization before use"
                    asyncio.create_task(self._create_default_db_permission(user_id, tool.name, is_sensitive=False))
            except Exception as exc:
                logger.warning(f"[ToolRouter] DB permission check error (falling back to memory): {exc}")

        # 3. Sensitive check
        if tool.is_sensitive:
            return False, f"Sensitive tool '{tool.name}' requires explicit authorization"

        # 4. Default policy: read-only tools and calendar tools allowed by default unless denied
        if tool.is_read_only or tool.name in ("schedule_calendar_event", "get_calendar_events") or tool.category == "calendar":
            return True, None

        # Write/destructive tools denied by default if no explicit permission
        return False, f"Destructive tool '{tool.name}' requires explicit authorization"

    async def _create_default_db_permission(self, user_id: UUID, tool_name: str, is_sensitive: bool = False) -> None:
        """Asynchronously seeds default permission record in PostgreSQL."""
        try:
            async with AsyncSessionLocal() as session:
                from app.models.mcp import ToolPermission as DBToolPermission
                stmt = select(DBToolPermission).where(
                    DBToolPermission.user_id == user_id,
                    DBToolPermission.tool_name == tool_name,
                )
                res = await session.execute(stmt)
                if not res.scalars().first():
                    new_perm = DBToolPermission(
                        user_id=user_id,
                        tool_name=tool_name,
                        allowed=not is_sensitive,
                        is_sensitive=is_sensitive,
                    )
                    session.add(new_perm)
                    await session.commit()
        except Exception as e:
            logger.debug(f"[ToolRouter] Could not seed default permission in DB: {e}")

    async def _update_db_last_used(self, user_id: UUID, tool_name: str) -> None:
        """Updates last_used_at timestamp in PostgreSQL upon tool execution."""
        try:
            async with AsyncSessionLocal() as session:
                from app.models.mcp import ToolPermission as DBToolPermission
                stmt = update(DBToolPermission).where(
                    DBToolPermission.user_id == user_id,
                    DBToolPermission.tool_name == tool_name,
                ).values(last_used_at=func.now())
                res = await session.execute(stmt)
                await session.commit()
                if res.rowcount == 0:
                    # Create record with last_used_at if didn't exist
                    new_perm = DBToolPermission(
                        user_id=user_id,
                        tool_name=tool_name,
                        allowed=True,
                        last_used_at=func.now(),
                    )
                    session.add(new_perm)
                    await session.commit()
        except Exception as e:
            logger.debug(f"[ToolRouter] Failed to update last_used_at in DB: {e}")

    async def route_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 5.0,
        db: Optional[AsyncSession] = None,
    ) -> ToolExecution:
        """
        Executes the full, secure MCP tool routing pipeline:
        1. Tool lookup in registry (detects MCP unavailable)
        2. PostgreSQL & In-Memory Permission validation (detects permission denied)
        3. Execution via designated MCP Client with timeout protection
        4. Structured error categorization (auth failure, unavailable, timeout, API failure)
        5. Safe audit logging (scrubbing secrets)
        6. Updates last_used_at in PostgreSQL on completion
        7. Return ToolExecution record
        """
        # Ensure tools are discovered if cache is empty
        if not self._tools:
            await self.discover_all_tools()

        tool = self._tools.get(tool_name)
        client = self._tool_to_client.get(tool_name)

        server_name = tool.server_name if tool else "unknown"
        execution = ToolExecution(
            user_id=user_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            status=ExecutionStatus.RUNNING,
        )

        # 1. Check Tool and Server Availability
        if not tool or not client or not client.is_connected:
            execution.fail(
                error=f"MCP server or tool '{tool_name}' is currently unavailable or disconnected",
                duration_ms=0.0,
                error_type="mcp_unavailable",
            )
            self._record_execution(execution)
            return execution

        # 2. Permission Check in PostgreSQL & Rules
        is_allowed, deny_reason = await self.validate_permission(
            user_id=user_id,
            tool=tool,
            arguments=arguments,
            db=db,
        )
        if not is_allowed:
            execution.deny(
                reason=deny_reason or "Unauthorized by security policy",
                error_type="permission_denied",
            )
            self._record_execution(execution)
            return execution

        # 3. Secure backend execution via MCP Client -> MCP Server -> External Service
        start_time = time.perf_counter()
        context_payload = dict(context or {})
        if user_id:
            context_payload["user_id"] = user_id
        if conversation_id:
            context_payload["conversation_id"] = conversation_id

        try:
            result = await asyncio.wait_for(
                client.call_tool(
                    name=tool_name,
                    arguments=arguments,
                    context=context_payload,
                ),
                timeout=timeout_seconds,
            )
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.complete(result=result, duration_ms=duration_ms)
            if user_id:
                asyncio.create_task(self._update_db_last_used(user_id, tool_name))

        except asyncio.TimeoutError:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.fail(
                error=f"Tool '{tool_name}' timed out after {timeout_seconds} seconds",
                duration_ms=duration_ms,
                error_type="timeout",
            )
        except MCPAuthenticationError as auth_err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.fail(
                error=str(auth_err),
                duration_ms=duration_ms,
                error_type="authentication_failure",
            )
        except MCPUnavailableError as unav_err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.fail(
                error=str(unav_err),
                duration_ms=duration_ms,
                error_type="mcp_unavailable",
            )
        except MCPTimeoutError as to_err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.fail(
                error=str(to_err),
                duration_ms=duration_ms,
                error_type="timeout",
            )
        except MCPAPIFailureError as api_err:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.fail(
                error=str(api_err),
                duration_ms=duration_ms,
                error_type="api_failure",
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            execution.fail(
                error=f"Tool execution failed: {str(exc)}",
                duration_ms=duration_ms,
                error_type="api_failure",
            )

        self._record_execution(execution)
        return execution

    def _record_execution(self, execution: ToolExecution) -> None:
        """Stores execution in audit log."""
        self._execution_history.append(execution)
        if len(self._execution_history) > 200:
            self._execution_history.pop(0)

    def get_recent_executions(self, limit: int = 50) -> List[ToolExecution]:
        """Returns recent tool executions."""
        return list(reversed(self._execution_history[-limit:]))


# Global singleton instance of ToolRouter
tool_router = ToolRouter()
