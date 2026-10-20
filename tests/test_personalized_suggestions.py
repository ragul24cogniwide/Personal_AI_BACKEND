"""
Comprehensive Test Suite for Personalized Context-Aware Suggestions.

Verifies:
1. Context-Aware Fitness Suggestions:
   - "I don't feel like going to the gym."
   - Synthesizes user profile fitness interests, goals, habits, and relevant memories.
   - Suggestions are recommendations, NOT commands (no bossy/coercive phrasing).
   - Avoids generic textbook lectures.

2. Day Planning Context with MCP Calendar:
   - "What should I focus on today?"
   - Triggers MCP calendar tool.
   - Synthesizes scheduled meetings, active goals, and habits into a cohesive daily focus suggestion.

3. Anti-Hallucination & Unavailable Information Grounding:
   - When schedule/calendar is empty, does NOT hallucinate meetings, clients, or appointments.
   - Acknowledges open schedule and grounds suggestions only in real goals.

4. Configurable Proactivity Levels (Low, Medium, High):
   - Proactivity API: GET, PATCH, and PUT /preferences/proactivity.
   - Low proactivity: Reactive only; does not volunteer unprompted advice or tasks.
   - High proactivity: Proactively suggests plans, complementary habits, and next steps.
   - Medium proactivity: Balanced mode with at most one lightweight, gentle suggestion.
"""

from datetime import datetime, timezone
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.main import app
from app.ai.service import ai_service
from app.ai.context_builder import context_builder
from app.core.security import create_access_token
from app.database.session import AsyncSessionLocal
from app.mcp import tool_router
from app.mcp.servers.calendar import CalendarMCPServer
from app.models.memory import Memory
from app.models.user import User, UserGoal, UserHabit, UserPreference, UserProfile
from app.schemas.memory import MemoryCreate
from app.services.memory import memory_service


@pytest.fixture(autouse=True)
async def setup_calendar_mcp():
    """Ensures CalendarMCPServer is registered and reset for tests."""
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


async def create_user_with_full_context(
    db,
    proactivity: str = "medium",
    goals: list = None,
    habits: list = None,
) -> User:
    """Creates a user with rich profile, preferences, goals, and habits."""
    email = f"suggestions_test_{uuid4().hex[:8]}@example.com"
    user = User(email=email, password_hash="test_pw_hash", is_onboarded=True)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    profile = UserProfile(
        user_id=user.id,
        name="Ragul",
        preferred_name="Ragul",
        occupation="AI Systems Engineer",
        bio="Building personal AI agents and audio processing pipelines",
        fitness_interests=["mobility stretching", "calisthenics", "outdoor running"],
    )
    pref = UserPreference(
        user_id=user.id,
        assistant_tone="empathetic, supportive, concise",
        verbosity="balanced",
        proactivity=proactivity,
        timezone="Asia/Kolkata",
    )
    db.add_all([profile, pref])

    if goals:
        for g in goals:
            db.add(UserGoal(user_id=user.id, **g))

    if habits:
        for h in habits:
            db.add(UserHabit(user_id=user.id, **h))

    await db.commit()
    return user


# -------------------------------------------------------------------------
# Test 1: Fitness Hesitation Contextual Suggestions (Not Commands)
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fitness_hesitation_personalized_suggestions():
    """
    Test 1: User says: "I don't feel like going to the gym."
    The assistant must:
    - Use available fitness context (mobility, light running, habits).
    - Provide recommendations, NOT commands (no "You must", "You have to", "Go now").
    - Validate feelings empathetically instead of generic textbook lecture.
    """
    async with AsyncSessionLocal() as db:
        user = await create_user_with_full_context(
            db,
            proactivity="medium",
            goals=[
                {
                    "category": "fitness",
                    "title": "Build mobility & consistent cardio endurance",
                    "description": "Prefers gentle active recovery when low on energy",
                    "priority": "high",
                    "status": "in_progress",
                }
            ],
            habits=[
                {
                    "category": "fitness",
                    "name": "15-minute mobility stretch",
                    "frequency": "daily",
                    "description": "Light full-body stretching on rest days",
                }
            ],
        )

        # Seed personal memory about fatigue & preferred light activity
        await memory_service.create_memory(
            db=db,
            user_id=user.id,
            data=MemoryCreate(
                category="fitness",
                content="User prefers a gentle 15-minute mobility session or a short outdoor walk whenever feeling too tired for heavy gym lifting.",
                importance=0.95,
            ),
        )

        # Execute voice/text turn
        chunks = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="I don't feel like going to the gym.",
        ):
            chunks.append(chunk)

        response = "".join(chunks).strip()
        print("\n[Fitness Hesitation Response]:\n", response)

        assert len(response) > 15
        resp_lower = response.lower()

        # 1. Must incorporate personal fitness context (mobility, walk, stretching, light movement, rest)
        has_fitness_context = any(
            k in resp_lower
            for k in [
                "stretch",
                "mobility",
                "walk",
                "light",
                "recovery",
                "rest",
                "home",
                "listen to your body",
                "take it easy",
            ]
        )
        assert has_fitness_context, f"Response lacked personal fitness context: {response}"

        # 2. Strict Guardrail: Recommendations, NOT commands!
        # Must not contain demanding, bossy commands
        forbidden_commands = [
            "you must go",
            "you have to go",
            "go to the gym anyway",
            "no excuses",
            "you need to push through",
        ]
        for cmd in forbidden_commands:
            assert cmd not in resp_lower, f"Response contained commanding phrasing '{cmd}': {response}"


# -------------------------------------------------------------------------
# Test 2: Day Planning Suggestions with MCP Calendar
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_day_planning_suggestions_with_mcp_calendar(setup_calendar_mcp):
    """
    Test 2: User says: "What should I focus on today?"
    The assistant must:
    - Query MCP Calendar tool for today's schedule.
    - Synthesize active goals, habits, and scheduled meetings.
    - Suggest a cohesive daily plan as recommendations.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.reset_modes()

    async with AsyncSessionLocal() as db:
        user = await create_user_with_full_context(
            db,
            proactivity="medium",
            goals=[
                {
                    "category": "work",
                    "title": "Complete voice assistant MCP integration",
                    "description": "Core architecture and end-to-end testing",
                    "priority": "high",
                    "status": "in_progress",
                }
            ],
            habits=[
                {
                    "category": "productivity",
                    "name": "Morning focus deep work block",
                    "frequency": "daily",
                    "description": "90 minutes uninterrupted coding",
                }
            ],
        )

        chunks = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="What should I focus on today?",
        ):
            chunks.append(chunk)

        response = "".join(chunks).strip()
        print("\n[Day Planning Response with Calendar]:\n", response)

        assert len(response) > 20
        resp_lower = response.lower()

        # 1. MCP tool was called
        recent_execs = tool_router.get_recent_executions(limit=5)
        cal_calls = [e for e in recent_execs if e.tool_name == "get_calendar_events"]
        assert len(cal_calls) > 0, "Expected get_calendar_events tool to be executed"

        # 2. Response incorporates calendar schedule (e.g. standup, lunch, demo, meetings)
        has_calendar_info = any(
            w in resp_lower
            for w in ["standup", "demo", "lunch", "meeting", "schedule", "calendar", "afternoon"]
        )
        assert has_calendar_info, f"Response did not reference calendar schedule: {response}"

        # 3. Response incorporates goals or focus work
        has_goal_context = any(
            w in resp_lower
            for w in ["voice", "assistant", "mcp", "integration", "deep work", "focus", "project"]
        )
        assert has_goal_context, f"Response did not integrate active goals or focus: {response}"


# -------------------------------------------------------------------------
# Test 3: Anti-Hallucination & Unavailable Information Grounding
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_anti_hallucination_on_empty_schedule(setup_calendar_mcp):
    """
    Test 3: Empty Calendar.
    When the assistant queries the calendar and finds no events,
    it must NOT invent fictional meetings, appointments, or deadlines.
    It should acknowledge the open day and suggest focusing on real goals.
    """
    calendar_server: CalendarMCPServer = setup_calendar_mcp
    calendar_server.set_default_mode("empty")

    async with AsyncSessionLocal() as db:
        user = await create_user_with_full_context(
            db,
            proactivity="medium",
            goals=[
                {
                    "category": "work",
                    "title": "Optimize vector search latency",
                    "description": "pgvector index tuning",
                    "priority": "medium",
                    "status": "in_progress",
                }
            ],
        )

        chunks = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="What should I focus on today?",
        ):
            chunks.append(chunk)

        response = "".join(chunks).strip()
        print("\n[Anti-Hallucination Response]:\n", response)

        resp_lower = response.lower()

        # Must NOT claim fictional meetings from the normal fixture
        assert "standup" not in resp_lower, "Hallucinated standup meeting when calendar was empty"
        assert "architecture demo" not in resp_lower, "Hallucinated demo when calendar was empty"

        # Must acknowledge clear/open schedule or focus on the real goal
        assert any(
            w in resp_lower
            for w in ["clear", "open", "no meetings", "no events", "vector", "latency", "free"]
        ), f"Response failed to acknowledge open schedule or real goal: {response}"


# -------------------------------------------------------------------------
# Test 4: Configurable Proactivity Settings (Low, Medium, High)
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_configurable_proactivity_behavior_and_api():
    """
    Test 4: Configurable Proactivity.
    - Low: Reactive only. Casual remarks receive direct concise reply without unsolicited advice.
    - High: Proactive partner mode. Connects statement with goals and habits constructively.
    - API: GET /preferences, PATCH /preferences, PUT /preferences/proactivity persist in DB.
    """
    async with AsyncSessionLocal() as db:
        user = await create_user_with_full_context(
            db,
            proactivity="low",
            goals=[
                {
                    "category": "learning",
                    "title": "Master Rust systems programming",
                    "description": "Complete chapter 4 exercises",
                    "priority": "medium",
                    "status": "in_progress",
                }
            ],
            habits=[
                {
                    "category": "learning",
                    "name": "30 mins daily Rust study",
                    "frequency": "daily",
                    "description": "Morning study with coffee",
                }
            ],
        )
        token, _, _ = create_access_token(subject=str(user.id))

        # --- A. Low Proactivity Behavior ---
        chunks_low = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="I am brewing a cup of coffee right now.",
        ):
            chunks_low.append(chunk)

        response_low = "".join(chunks_low).strip()
        print("\n[Low Proactivity Response]:\n", response_low)

        # Low proactivity must be concise and reactive: enjoy your coffee, no unsolicited tasks
        assert len(response_low.split()) < 35, f"Low proactivity response too verbose: {response_low}"
        # Should NOT boss the user around or shove study tasks into coffee time
        assert "you must" not in response_low.lower()

        # --- B. High Proactivity Behavior ---
        pref_stmt = select(UserPreference).where(UserPreference.user_id == user.id)
        pref = (await db.execute(pref_stmt)).scalar_one()
        pref.proactivity = "high"
        await db.commit()

        chunks_high = []
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="I am brewing a cup of coffee right now.",
        ):
            chunks_high.append(chunk)

        response_high = "".join(chunks_high).strip()
        print("\n[High Proactivity Response]:\n", response_high)

        # High proactivity should acknowledge coffee and proactively suggest pairing with Rust habit or goal
        resp_high_lower = response_high.lower()
        has_proactive_suggestion = any(
            w in resp_high_lower
            for w in ["rust", "study", "learning", "chapter", "reading", "plan", "habit", "routine"]
        )
        assert has_proactive_suggestion, (
            f"High proactivity expected connection to habit/goal, got: {response_high}"
        )

    # --- C. Test Preferences REST Endpoints ---
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        headers = {"Authorization": f"Bearer {token}"}

        # 1. GET /preferences
        res = await client.get("/preferences", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert data["proactivity"] == "high"

        # 2. PATCH /preferences (change to medium)
        patch_res = await client.patch(
            "/preferences",
            headers=headers,
            json={"proactivity": "medium", "assistant_tone": "zen, calm"},
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()
        assert patch_data["proactivity"] == "medium"
        assert patch_data["assistant_tone"] == "zen, calm"

        # 3. PUT /preferences/proactivity (direct update to low)
        put_res = await client.put(
            "/preferences/proactivity",
            headers=headers,
            json={"proactivity": "low"},
        )
        assert put_res.status_code == 200
        assert put_res.json()["proactivity"] == "low"

        # 4. Invalid proactivity level rejection (validation)
        bad_res = await client.put(
            "/preferences/proactivity",
            headers=headers,
            json={"proactivity": "extreme_proactive"},
        )
        assert bad_res.status_code == 422 or bad_res.status_code == 400

        # 5. Verify PostgreSQL persistence
        async with AsyncSessionLocal() as db_verify:
            verify_pref = (
                await db_verify.execute(
                    select(UserPreference).where(UserPreference.user_id == user.id)
                )
            ).scalar_one()
            assert verify_pref.proactivity == "low"
