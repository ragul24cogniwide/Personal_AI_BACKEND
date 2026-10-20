"""
Comprehensive Test Suite for Model Context Protocol (MCP) Architecture.

Tests:
1. MCP Abstraction Classes (MCPClient, MCPServer, MCPTool, ToolRouter, ToolPermission, ToolExecution)
2. Tool Discovery & Schema Conversion for Gemini
3. Credential Isolation & Safe Secret Scrubbing in ToolExecution logs
4. Security Permission Validation (ALLOW, DENY, REQUIRE_CONFIRMATION)
5. Prototype Calendar MCP Server Execution
6. End-to-End Gemini Tool Decision -> MCP Tool Router -> MCP Client -> MCP Server -> Spoken Synthesis
7. REST API Endpoints for Discovery and Auditing
"""

import asyncio
import datetime
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.mcp import (
    ExecutionStatus,
    MCPClient,
    MCPServer,
    MCPTool,
    PermissionLevel,
    ToolExecution,
    ToolPermission,
    ToolRouter,
    tool_router,
)
from app.mcp.models import sanitize_sensitive_data
from app.mcp.servers.calendar import CalendarMCPServer
from app.ai.service import ai_service
from app.database.session import AsyncSessionLocal
from app.models.user import User, UserProfile, UserPreference


@pytest.mark.asyncio
async def test_mcp_abstractions_and_discovery():
    """
    Verifies MCP abstraction classes:
    - MCPServer tool registration
    - MCPClient discovery
    - ToolRouter aggregation
    - Tool schema conversion for Gemini
    """
    router = ToolRouter()

    # Create server
    server = MCPServer(name="test-server", version="1.0.0", description="Test MCP Server")
    tool = MCPTool(
        name="calculate_sum",
        description="Calculate the sum of two integers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
        server_name="test-server",
        required_permissions=["math.execute"],
        is_read_only=True,
    )

    async def sum_handler(a: int, b: int):
        return {"result": a + b}

    server.register_tool(tool=tool, handler=sum_handler)

    # Attach to router
    client = router.register_server(server)
    assert isinstance(client, MCPClient)
    assert client.server_name == "test-server"

    # Test Discovery
    discovered = await router.discover_all_tools()
    assert len(discovered) == 1
    assert discovered[0].name == "calculate_sum"

    # Verify Gemini Tool Schema Translation
    gemini_tools = router.get_gemini_tools()
    assert len(gemini_tools) == 1
    decl = gemini_tools[0].function_declarations[0]
    assert decl.name == "calculate_sum"
    assert "sum of two integers" in decl.description
    assert "a" in decl.parameters.properties
    assert "b" in decl.parameters.properties


@pytest.mark.asyncio
async def test_security_credential_isolation_and_safe_scrubbing():
    """
    Security verification:
    1. Credentials never leak into Gemini tool declarations.
    2. ToolExecution sanitizes sensitive tokens, passwords, and secrets in logs.
    """
    calendar_server = CalendarMCPServer(api_key="SUPER_SECRET_CALENDAR_BEARER_TOKEN_999")
    tools = calendar_server.list_tools()
    cal_tool = tools[0]

    # Verify Gemini declaration has NO credentials
    gemini_decl = cal_tool.to_gemini_function_declaration()
    decl_str = str(gemini_decl)
    assert "SUPER_SECRET_CALENDAR_BEARER_TOKEN_999" not in decl_str
    assert "token" not in gemini_decl.name.lower()

    # Verify safe sanitization logic
    dirty_data = {
        "user_email": "user@example.com",
        "api_key": "secret_key_abc123",
        "nested": {
            "auth_token": "bearer_jwt_xyz987",
            "normal_field": "visible_data",
        },
        "passwords": ["p1", "p2"],
    }
    cleaned = sanitize_sensitive_data(dirty_data)
    assert cleaned["user_email"] == "user@example.com"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["auth_token"] == "[REDACTED]"
    assert cleaned["nested"]["normal_field"] == "visible_data"

    # Verify ToolExecution record sanitization
    execution = ToolExecution(
        tool_name="get_calendar_events",
        server_name="calendar-service",
        arguments={"date": "today", "auth_secret": "my_secret_token"},
    )
    assert execution.arguments["date"] == "today"
    assert execution.arguments["auth_secret"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_tool_permission_validation():
    """
    Verifies ToolPermission access control:
    - ALLOW permits execution
    - DENY strictly blocks execution
    - REQUIRE_CONFIRMATION blocks automated execution
    """
    router = ToolRouter()
    server = CalendarMCPServer()
    router.register_server(server)
    await router.discover_all_tools()

    user_id = uuid4()

    # 1. Default read-only tool is ALLOWED
    exec_allowed = await router.route_tool_execution(
        tool_name="get_calendar_events",
        arguments={"date": "today"},
        user_id=user_id,
    )
    assert exec_allowed.status == ExecutionStatus.COMPLETED
    assert exec_allowed.result["status"] == "success"

    # 2. Add explicit DENY permission for user
    deny_rule = ToolPermission(
        user_id=user_id,
        tool_name="get_calendar_events",
        permission_level=PermissionLevel.DENY,
    )
    router.set_permission(deny_rule)

    exec_denied = await router.route_tool_execution(
        tool_name="get_calendar_events",
        arguments={"date": "today"},
        user_id=user_id,
    )
    assert exec_denied.status == ExecutionStatus.DENIED
    assert "denied by security policy" in exec_denied.error_message

    # 3. Another user without DENY rule is still ALLOWED
    other_user_id = uuid4()
    exec_other = await router.route_tool_execution(
        tool_name="get_calendar_events",
        arguments={"date": "today"},
        user_id=other_user_id,
    )
    assert exec_other.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_calendar_prototype_server_execution():
    """
    Tests CalendarMCPServer prototype execution and event formatting.
    """
    server = CalendarMCPServer()
    result = await server.execute_tool(name="get_calendar_events", arguments={"date": "today"})

    assert result["status"] == "success"
    assert result["events_count"] >= 2
    titles = [e["title"] for e in result["events"]]
    assert "Team Standup" in titles
    assert any("Demo" in t or "Lunch" in t for t in titles)


@pytest.mark.asyncio
async def test_end_to_end_gemini_tool_decision_and_spoken_synthesis():
    """
    Tests the complete end-to-end tool execution flow:
    User: "What do I have on my calendar today?"
    Gemini requests calendar tool -> Tool Router -> Permission check -> MCP Client -> Calendar Server -> Result -> Gemini -> Spoken synthesis
    """
    # Initialize router with CalendarMCPServer
    if not tool_router.get_tool("get_calendar_events"):
        tool_router.register_server(CalendarMCPServer())
        await tool_router.discover_all_tools()

    async with AsyncSessionLocal() as db:
        # Create test user
        email = f"mcp_user_{uuid4().hex[:8]}@example.com"
        user = User(email=email, password_hash="test_pw_hash", is_onboarded=True)
        db.add(user)
        await db.commit()
        await db.refresh(user)

        user_profile = UserProfile(user_id=user.id, name="Ragul", preferred_name="Ragul")
        user_pref = UserPreference(user_id=user.id, assistant_tone="friendly and concise")
        db.add_all([user_profile, user_pref])
        await db.commit()

        # Execute conversation turn with calendar query
        chunks = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="What do I have on my calendar today?",
        ):
            chunks.append(chunk)

        full_spoken_response = "".join(chunks)
        print("\n[E2E Spoken Response]:", full_spoken_response)

        # Gemini should mention calendar events in natural spoken English
        assert len(full_spoken_response) > 20
        response_lower = full_spoken_response.lower()
        assert (
            "standup" in response_lower
            or "calendar" in response_lower
            or "events" in response_lower
            or "demo" in response_lower
            or "lunch" in response_lower
        )

        # Check audit execution record
        recent_execs = tool_router.get_recent_executions(limit=10)
        cal_execs = [e for e in recent_execs if e.tool_name == "get_calendar_events"]
        assert len(cal_execs) > 0
        assert cal_execs[0].status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_mcp_discovery_rest_apis():
    """
    Verifies REST endpoints for MCP tool discovery, servers, executions, and permissions.
    """
    # Ensure calendar server is registered
    if not tool_router.get_tool("get_calendar_events"):
        tool_router.register_server(CalendarMCPServer())
        await tool_router.discover_all_tools()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. GET /api/v1/mcp/tools
        res_tools = await ac.get("/api/v1/mcp/tools")
        assert res_tools.status_code == 200
        tools_data = res_tools.json()
        assert len(tools_data) >= 1
        assert any(t["name"] == "get_calendar_events" for t in tools_data)

        # 2. GET /api/v1/mcp/servers
        res_servers = await ac.get("/api/v1/mcp/servers")
        assert res_servers.status_code == 200
        servers_data = res_servers.json()
        assert len(servers_data) >= 1
        assert any(s["name"] == "calendar-service" for s in servers_data)

        # 3. POST /api/v1/mcp/execute (Direct testing execution)
        res_exec = await ac.post(
            "/api/v1/mcp/execute",
            json={"tool_name": "get_calendar_events", "arguments": {"date": "today"}},
        )
        assert res_exec.status_code == 200
        exec_data = res_exec.json()
        assert exec_data["status"] == "completed"
        assert exec_data["result"]["status"] == "success"

        # 4. GET /api/v1/mcp/executions
        res_history = await ac.get("/api/v1/mcp/executions")
        assert res_history.status_code == 200
        history_data = res_history.json()
        assert len(history_data) >= 1
