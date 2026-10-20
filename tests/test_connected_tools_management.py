"""
Integration and End-to-End Test Suite for Connected Tools Management System.

Tests:
1. Tool Discovery & Status API (GET /tools & GET /api/v1/tools):
   - Returns tool name, connection status, permissions, last used timestamp.
   - ZERO credentials exposed in API responses (service tokens, API keys, private OAuth secrets).
2. Permission Enable / Disable / Toggle:
   - Persists state changes in PostgreSQL `tool_permissions` table.
3. Service Connect / Disconnect:
   - Updates `mcp_servers.status` in PostgreSQL ('active' vs 'disconnected').
4. Sensitive Tool Authorization:
   - Sensitive tools require explicit authorization before execution.
   - Explicit authorization endpoint unlocks sensitive tool.
5. Assistant Permission Check Enforcement:
   - When tool is disabled in PostgreSQL, assistant denies execution.
   - Assistant never claims to have accessed the tool or data on permission denial.
6. Successful Execution Flow & Last Used Timestamp:
   - When tool is allowed, assistant completes execution and speaks results.
   - Automatically updates `last_used_at` in PostgreSQL `tool_permissions`.
"""

from datetime import datetime, timezone
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.main import app
from app.ai.service import ai_service
from app.core.security import create_access_token
from app.database.session import AsyncSessionLocal
from app.mcp import (
    ExecutionStatus,
    tool_router,
)
from app.mcp.servers.calendar import CalendarMCPServer
from app.models.mcp import MCPServer as DBMCPServer, ToolPermission as DBToolPermission
from app.models.user import User, UserPreference, UserProfile


@pytest.fixture(autouse=True)
async def ensure_calendar_server():
    """Ensures calendar server is registered and resets modes."""
    server = tool_router._servers.get("calendar-service")
    if not server:
        server = CalendarMCPServer()
        tool_router.register_server(server)
        await tool_router.discover_all_tools()

    if isinstance(server, CalendarMCPServer):
        server.reset_modes()

    yield server

    if isinstance(server, CalendarMCPServer):
        server.reset_modes()


async def create_test_user(db) -> User:
    """Helper to create an authenticated test user."""
    email = f"tools_test_{uuid4().hex[:8]}@example.com"
    user = User(email=email, password_hash="test_secret_hash", is_onboarded=True)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    profile = UserProfile(user_id=user.id, name="TestUser", preferred_name="Tester")
    pref = UserPreference(user_id=user.id, assistant_tone="helpful, concise")
    db.add_all([profile, pref])
    await db.commit()
    return user


@pytest.mark.asyncio
async def test_get_connected_tools_no_credentials_exposed():
    """
    Validates GET /tools endpoint:
    - Lists tools with tool_name, connection_status, permissions, last_used_at, allowed.
    - Strictly verifies that NO service tokens or credentials leak in the JSON response.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res = await client.get(
            "/tools",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert "tools" in data
        assert data["total"] >= 1

        tools = data["tools"]
        tool_names = [t["tool_name"] for t in tools]
        assert "get_calendar_events" in tool_names

        calendar_tool = next(t for t in tools if t["tool_name"] == "get_calendar_events")
        assert calendar_tool["display_name"] == "Calendar"
        assert calendar_tool["connection_status"] in ("connected", "disconnected")
        assert "calendar.read" in calendar_tool["permissions"]
        assert "allowed" in calendar_tool
        assert "is_sensitive" in calendar_tool
        assert calendar_tool["is_sensitive"] is False

        # STRICT CREDENTIAL LEAKAGE TEST:
        # None of the backend secrets or private tokens may be exposed in payload
        res_str = res.text.lower()
        forbidden_strings = [
            "sec_backend_calendar_token",
            "password_hash",
            "private_key",
            "bearer",
            "client_secret",
            "access_token",
        ]
        for secret in forbidden_strings:
            assert secret not in res_str, f"Credential leaked in tools API: {secret}"


@pytest.mark.asyncio
async def test_toggle_and_enable_disable_permissions():
    """
    Validates enable, disable, and toggle endpoints update PostgreSQL `tool_permissions`.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # 1. Disable tool
        res_dis = await client.post(
            "/tools/get_calendar_events/disable",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_dis.status_code == 200
        dis_data = res_dis.json()
        assert dis_data["tool"]["allowed"] is False

        # Check DB directly
        async with AsyncSessionLocal() as db:
            stmt = select(DBToolPermission).where(
                DBToolPermission.user_id == user.id,
                DBToolPermission.tool_name == "get_calendar_events",
            )
            perm = (await db.execute(stmt)).scalar_one_or_none()
            assert perm is not None
            assert perm.allowed is False

        # 2. Enable tool
        res_en = await client.post(
            "/tools/get_calendar_events/enable",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_en.status_code == 200
        en_data = res_en.json()
        assert en_data["tool"]["allowed"] is True

        # Check DB directly
        async with AsyncSessionLocal() as db:
            stmt = select(DBToolPermission).where(
                DBToolPermission.user_id == user.id,
                DBToolPermission.tool_name == "get_calendar_events",
            )
            perm = (await db.execute(stmt)).scalar_one_or_none()
            assert perm is not None
            assert perm.allowed is True


@pytest.mark.asyncio
async def test_connect_and_disconnect_service():
    """
    Validates connect and disconnect endpoints update PostgreSQL `mcp_servers`.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # 1. Disconnect tool service
        res_disc = await client.post(
            "/tools/get_calendar_events/disconnect",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_disc.status_code == 200
        disc_data = res_disc.json()
        assert disc_data["tool"]["connection_status"] == "disconnected"

        # Check PostgreSQL mcp_servers table
        async with AsyncSessionLocal() as db:
            stmt = select(DBMCPServer).where(
                DBMCPServer.user_id == user.id,
                DBMCPServer.name == "calendar-service",
            )
            srv = (await db.execute(stmt)).scalar_one_or_none()
            assert srv is not None
            assert srv.status == "disconnected"

        # 2. Re-connect tool service
        res_conn = await client.post(
            "/tools/get_calendar_events/connect",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res_conn.status_code == 200
        conn_data = res_conn.json()
        assert conn_data["tool"]["connection_status"] == "connected"

        # Check PostgreSQL
        async with AsyncSessionLocal() as db:
            stmt = select(DBMCPServer).where(
                DBMCPServer.user_id == user.id,
                DBMCPServer.name == "calendar-service",
            )
            srv = (await db.execute(stmt)).scalar_one_or_none()
            assert srv is not None
            assert srv.status == "active"


@pytest.mark.asyncio
async def test_sensitive_tool_requires_explicit_authorization():
    """
    Validates sensitive tool behavior:
    - Sensitive tool is NOT allowed by default until explicitly authorized.
    - POST /tools/{tool_name}/authorize grants explicit authorization in PostgreSQL.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # List tools to trigger initial state creation
        res = await client.get("/tools", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        tools = res.json()["tools"]
        sched_tool = next(t for t in tools if t["tool_name"] == "schedule_calendar_event")
        assert sched_tool["is_sensitive"] is True
        # Initial default for sensitive tool is allowed=False
        assert sched_tool["allowed"] is False

        # Attempting execution before authorization must be denied by router
        execution = await tool_router.route_tool_execution(
            tool_name="schedule_calendar_event",
            arguments={"title": "Team Standup", "date": "today"},
            user_id=user.id,
        )
        assert execution.status == ExecutionStatus.DENIED
        assert "authorization" in execution.error_message.lower()

        # Explicitly authorize the sensitive tool via API
        auth_res = await client.post(
            "/tools/schedule_calendar_event/authorize",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert auth_res.status_code == 200
        auth_data = auth_res.json()
        assert auth_data["tool"]["allowed"] is True

        # Check PostgreSQL
        async with AsyncSessionLocal() as db:
            stmt = select(DBToolPermission).where(
                DBToolPermission.user_id == user.id,
                DBToolPermission.tool_name == "schedule_calendar_event",
            )
            perm = (await db.execute(stmt)).scalar_one_or_none()
            assert perm is not None
            assert perm.allowed is True

        # Now execution must succeed!
        execution2 = await tool_router.route_tool_execution(
            tool_name="schedule_calendar_event",
            arguments={"title": "Team Standup", "date": "today"},
            user_id=user.id,
        )
        assert execution2.status == ExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_assistant_checks_permission_denial(ensure_calendar_server):
    """
    Tests that the assistant checks PostgreSQL permissions before tool execution:
    When user has disabled the tool in Connected Tools, assistant handles the denial
    gracefully and never claims to have accessed the calendar.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

    # Disable tool via API
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res = await client.post(
            "/tools/get_calendar_events/disable",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 200

    # User speaks to assistant asking for calendar
    async with AsyncSessionLocal() as db:
        chunks = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="What do I have on my calendar today?",
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        resp_lower = full_response.lower()

        # Assistant must explain that permission is disabled or tool is unavailable
        assert (
            "permission" in resp_lower
            or "disabled" in resp_lower
            or "access" in resp_lower
            or "unable" in resp_lower
            or "cannot" in resp_lower
            or "can't" in resp_lower
            or "connected tools" in resp_lower
        )

        # Assistant MUST NOT hallucinate the mock events
        assert "quarterly roadmap sync" not in resp_lower


@pytest.mark.asyncio
async def test_assistant_successful_execution_updates_last_used(ensure_calendar_server):
    """
    Tests that successful execution:
    1. Delivers calendar information in spoken response.
    2. Updates last_used_at timestamp in PostgreSQL tool_permissions.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

    # Explicitly ensure tool is enabled
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        await client.post(
            "/tools/get_calendar_events/enable",
            headers={"Authorization": f"Bearer {token}"},
        )

    # Execute assistant voice turn
    async with AsyncSessionLocal() as db:
        chunks = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="What do I have today?",
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        assert len(full_response) > 0
        resp_lower = full_response.lower()

        # Should mention calendar event
        assert any(keyword in resp_lower for keyword in ["standup", "lunch", "demo", "event", "scheduled"])

        # Verify last_used_at in PostgreSQL
        stmt = select(DBToolPermission).where(
            DBToolPermission.user_id == user.id,
            DBToolPermission.tool_name == "get_calendar_events",
        )
        perm = (await db.execute(stmt)).scalar_one_or_none()
        assert perm is not None
        assert perm.last_used_at is not None
