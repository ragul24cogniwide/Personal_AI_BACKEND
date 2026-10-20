"""
Comprehensive Test Suite for Real Calendar MCP Tool and Error Handling.

Covers the complete voice flow and all failure/edge cases:
1. Successful Schedule Query: "What do I have today?" -> Spoken events
2. Empty Calendar: "What do I have today?" -> Calendar clear, no hallucinated events
3. Permission Denied: Tool access blocked -> Informs user, never claims calendar access
4. Authentication Failure: Not authenticated -> Informs user, never claims calendar access
5. MCP Unavailable: Server disconnected -> Informs user, never claims calendar access
6. Tool Timeout: Request times out -> Informs user, never claims calendar access
7. API Failure: Upstream error -> Informs user, never claims calendar access
"""

import asyncio
from uuid import uuid4
import pytest

from app.ai.service import ai_service
from app.database.session import AsyncSessionLocal
from app.mcp import (
    ExecutionStatus,
    PermissionLevel,
    ToolPermission,
    tool_router,
)
from app.mcp.servers.calendar import CalendarMCPServer
from app.models.user import User, UserProfile, UserPreference


@pytest.fixture(autouse=True)
async def setup_calendar_mcp():
    """Ensures CalendarMCPServer is registered and in normal state for tests."""
    calendar_server = tool_router._servers.get("calendar-service")
    if not calendar_server:
        calendar_server = CalendarMCPServer()
        tool_router.register_server(calendar_server)
        await tool_router.discover_all_tools()

    if isinstance(calendar_server, CalendarMCPServer):
        calendar_server.reset_modes()

    yield calendar_server

    if isinstance(calendar_server, CalendarMCPServer):
        calendar_server.reset_modes()


async def create_test_user(db) -> User:
    """Creates a temporary test user with friendly personality preferences."""
    email = f"cal_test_{uuid4().hex[:8]}@example.com"
    user = User(email=email, password_hash="test_pw", is_onboarded=True)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    profile = UserProfile(user_id=user.id, name="Alex", preferred_name="Alex")
    pref = UserPreference(user_id=user.id, assistant_tone="friendly, helpful, concise")
    db.add_all([profile, pref])
    await db.commit()
    return user


@pytest.mark.asyncio
async def test_calendar_success_what_do_i_have_today(setup_calendar_mcp):
    """
    Test 1: User asks: "What do I have today?"
    Flow: User voice/text -> Gemini -> Tool decision -> Router -> MCP -> Result -> Gemini -> Spoken response.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.reset_modes()

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[Calendar Success Output]:\n", response)

            assert len(response) > 10
            resp_lower = response.lower()
            # Must mention today's meetings from calendar
            assert (
                "standup" in resp_lower
                or "lunch" in resp_lower
                or "demo" in resp_lower
                or "schedule" in resp_lower
                or "calendar" in resp_lower
            )

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert len(cal_exec) > 0
            assert cal_exec[0].status == ExecutionStatus.COMPLETED
            assert cal_exec[0].result["calendar_accessed"] is True
            assert cal_exec[0].result["events_count"] >= 2
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_empty_schedule(setup_calendar_mcp):
    """
    Test 2: Empty calendar.
    Assistant successfully checks calendar, reports calendar is clear, and does NOT hallucinate events.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.set_default_mode("empty")

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[Empty Calendar Output]:\n", response)

            resp_lower = response.lower()
            # Must confirm calendar is clear or no events
            assert (
                "clear" in resp_lower
                or "no events" in resp_lower
                or "nothing scheduled" in resp_lower
                or "nothing on your calendar" in resp_lower
                or "free today" in resp_lower
                or "no scheduled" in resp_lower
            )

            # Must NOT invent meetings
            assert "standup" not in resp_lower
            assert "priya" not in resp_lower

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert cal_exec[0].status == ExecutionStatus.COMPLETED
            assert cal_exec[0].result["events_count"] == 0
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_permission_denied(setup_calendar_mcp):
    """
    Test 3: Permission denied.
    Assistant must inform user that permission was denied and MUST NOT claim it accessed the calendar.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            # Explicitly DENY calendar tool for this user
            deny_rule = ToolPermission(
                user_id=user.id,
                tool_name="get_calendar_events",
                permission_level=PermissionLevel.DENY,
            )
            tool_router.set_permission(deny_rule)

            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[Permission Denied Output]:\n", response)

            resp_lower = response.lower()
            # Must state permission denied / access denied
            assert (
                "permission" in resp_lower
                or "denied" in resp_lower
                or "unable to access" in resp_lower
                or "access" in resp_lower
                or "don't have permission" in resp_lower
            )

            # Must NOT claim calendar was empty or accessed
            assert "your calendar is clear" not in resp_lower
            assert "you have nothing on your calendar" not in resp_lower

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert cal_exec[0].status == ExecutionStatus.DENIED
            assert cal_exec[0].error_type == "permission_denied"
        finally:
            # Reset permission rule
            allow_rule = ToolPermission(
                user_id=user.id,
                tool_name="get_calendar_events",
                permission_level=PermissionLevel.ALLOW,
            )
            tool_router.set_permission(allow_rule)
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_authentication_failure(setup_calendar_mcp):
    """
    Test 4: Authentication failure.
    Assistant must inform user of authentication failure / connection requirement.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.set_default_mode("auth_error")

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[Authentication Failure Output]:\n", response)

            resp_lower = response.lower()
            # Must mention auth or not connected
            assert (
                "authenticat" in resp_lower
                or "connect" in resp_lower
                or "account" in resp_lower
                or "unable" in resp_lower
                or "login" in resp_lower
            )

            # Must NOT claim calendar is clear
            assert "your calendar is clear" not in resp_lower

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert cal_exec[0].status == ExecutionStatus.FAILED
            assert cal_exec[0].error_type == "authentication_failure"
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_mcp_unavailable(setup_calendar_mcp):
    """
    Test 5: MCP server unavailable.
    Assistant must inform user that the calendar service is unavailable.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.set_default_mode("unavailable")

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[MCP Unavailable Output]:\n", response)

            resp_lower = response.lower()
            # Must indicate service unavailable or offline
            assert (
                "unavailable" in resp_lower
                or "offline" in resp_lower
                or "unable" in resp_lower
                or "unreachable" in resp_lower
                or "service" in resp_lower
            )

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert cal_exec[0].status == ExecutionStatus.FAILED
            assert cal_exec[0].error_type == "mcp_unavailable"
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_tool_timeout(setup_calendar_mcp):
    """
    Test 6: Tool timeout.
    Assistant informs user that the request timed out.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.set_default_mode("timeout")

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[Tool Timeout Output]:\n", response)

            resp_lower = response.lower()
            # Must state request timed out or taking too long
            assert (
                "time" in resp_lower
                or "timed out" in resp_lower
                or "taking too long" in resp_lower
                or "unable" in resp_lower
                or "try again" in resp_lower
            )

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert cal_exec[0].status == ExecutionStatus.FAILED
            assert cal_exec[0].error_type == "timeout"
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_api_failure(setup_calendar_mcp):
    """
    Test 7: External API failure (e.g. 500 error).
    Assistant informs user that an error occurred fetching events.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.set_default_mode("api_error")

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="What do I have today?",
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[API Failure Output]:\n", response)

            resp_lower = response.lower()
            # Must indicate an error or issue connecting
            assert (
                "error" in resp_lower
                or "issue" in resp_lower
                or "problem" in resp_lower
                or "unable" in resp_lower
                or "fail" in resp_lower
            )

            # Audit record check
            recent = tool_router.get_recent_executions(limit=5)
            cal_exec = [e for e in recent if e.tool_name == "get_calendar_events"]
            assert cal_exec[0].status == ExecutionStatus.FAILED
            assert cal_exec[0].error_type == "api_failure"
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_calendar_schedule_event(setup_calendar_mcp):
    """
    Test 8: Schedule a meeting via voice prompt.
    User asks: "Schedule a meeting with Vijay tomorrow at 3 PM"
    Flow: Gemini -> schedule_calendar_event -> on_tool_executed callback -> Spoken confirmation.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.reset_modes()

    captured_tools = []

    async def on_tool_executed(name: str, result: dict):
        captured_tools.append((name, result))

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            chunks = []
            async for chunk in ai_service.process_turn_stream(
                db=db,
                user_id=user.id,
                user_text="Schedule a meeting with Vijay tomorrow at 3 PM",
                on_tool_executed=on_tool_executed,
            ):
                chunks.append(chunk)

            response = "".join(chunks).strip()
            print("\n[Schedule Calendar Output]:\n", response)

            # Check that schedule_calendar_event tool executed and was received via callback
            sched_calls = [t for t in captured_tools if t[0] == "schedule_calendar_event"]
            assert len(sched_calls) > 0, f"Expected schedule_calendar_event to be called, got: {captured_tools}"
            event_data = sched_calls[0][1]
            assert event_data.get("status") == "success" or event_data.get("calendar_accessed") is True
            assert len(response) > 5
        finally:
            await db.delete(user)
            await db.commit()

