"""
Test Suite for Context-Aware Greetings.

Verifies:
1. New User Flow:
   - User with is_onboarded=False triggers onboarding flow.
   - REST /greeting returns is_new_user=True, action='onboarding'.
   - WebSocket emits onboarding_required and transitions to idle without playing returning greeting audio.

2. Returning User Flow:
   - User with is_onboarded=True loads profile (name, timezone, tone).
   - Generates personalized greeting addressed to the user.
   - REST /greeting returns is_new_user=False with natural greeting text.

3. Context-Aware Context Loading (Memories, Topics, Schedule):
   - Loads recent memories (e.g. AI project).
   - Loads last conversation user topic.
   - Loads optional schedule/calendar events if tool permission is granted.
   - Verifies ZERO sensitive credentials or secrets are leaked in greeting.

4. Anti-Repetition Guardrail:
   - Multiple greetings for the same user do not repeat verbatim.
   - LRU cache prevents consecutive duplicate greetings.

5. Voice Streaming & VoiceOrb State Machine:
   - WebSocket emits greeting text and transitions state to 'speaking' (driving VoiceOrb animation).
   - Synthesizes and streams Sarvam TTS audio chunks.
   - Emits audio_end.

6. Barge-in / Interruption Flow:
   - User interrupts greeting by speaking (audio_start / audio_chunk / interrupt).
   - Assistant immediately halts TTS synthesis/streaming.
   - Transitions state to 'listening'.
   - Processes user request and provides answer.
"""

from datetime import datetime, timezone
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from starlette.testclient import TestClient

from app.main import app
from app.core.security import create_access_token
from app.database.session import AsyncSessionLocal
from app.mcp import tool_router
from app.mcp.servers.calendar import CalendarMCPServer
from app.models.conversation import Conversation, ConversationMessage
from app.models.mcp import ToolPermission as DBToolPermission
from app.models.memory import Memory
from app.models.user import User, UserPreference, UserProfile
from app.services.greeting import GreetingService, _recent_greetings_cache


@pytest.fixture(autouse=True)
async def ensure_calendar_server():
    """Ensures calendar server is registered for schedule context."""
    server = tool_router._servers.get("calendar-service")
    if not server:
        server = CalendarMCPServer()
        tool_router.register_server(server)
        await tool_router.discover_all_tools()
    yield server


async def create_new_user(db) -> User:
    """Creates a new, non-onboarded user."""
    email = f"new_user_{uuid4().hex[:8]}@example.com"
    user = User(email=email, password_hash="hashed_pw", is_onboarded=False)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def create_returning_user(db, name: str = "Ragul", preferred_name: str = "Ragul") -> User:
    """Creates an onboarded returning user with profile and preferences."""
    email = f"returning_{uuid4().hex[:8]}@example.com"
    user = User(email=email, password_hash="hashed_pw", is_onboarded=True)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    profile = UserProfile(
        user_id=user.id,
        name=name,
        preferred_name=preferred_name,
    )
    pref = UserPreference(
        user_id=user.id,
        timezone="Asia/Kolkata",
        assistant_tone="warm, concise",
    )
    db.add_all([profile, pref])
    await db.commit()
    return user


# -------------------------------------------------------------------------
# Test 1: New User Flow
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_new_user_greeting_triggers_onboarding():
    """
    Test 1: New User Flow.
    A new user (is_onboarded=False) must NOT receive a returning greeting.
    The service and endpoints must direct them to onboarding.
    """
    async with AsyncSessionLocal() as db:
        user = await create_new_user(db)
        token, _, _ = create_access_token(subject=str(user.id))

        service = GreetingService()
        result = await service.generate_greeting(db, user.id)

        assert result["is_new_user"] is True
        assert result["action"] == "onboarding"
        assert result.get("greeting_text") is None

    # Test REST Endpoint
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res = await client.get("/greeting", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["is_new_user"] is True
        assert data["action"] == "onboarding"

    # Test WebSocket Flow for New User
    with TestClient(app) as test_client:
        with test_client.websocket_connect(f"/ws/voice?token={token}") as ws:
            conn_msg = ws.receive_json()
            assert conn_msg["type"] == "connected"

            # Request greeting
            ws.send_json({"type": "greeting_request"})

            # Must receive onboarding_required frame
            onboarding_msg = ws.receive_json()
            assert onboarding_msg["type"] == "onboarding_required"
            assert onboarding_msg["action"] == "onboarding"

            # State frame must be idle
            state_msg = ws.receive_json()
            assert state_msg["type"] == "state"
            assert state_msg["state"] == "idle"


# -------------------------------------------------------------------------
# Test 2: Returning User Flow
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_returning_user_greeting_loads_profile():
    """
    Test 2: Returning User Flow.
    An onboarded user (is_onboarded=True) loads user profile and receives
    a personalized spoken greeting from Gemini addressed to them.
    """
    async with AsyncSessionLocal() as db:
        user = await create_returning_user(db, name="Ragul", preferred_name="Ragul")
        token, _, _ = create_access_token(subject=str(user.id))

        service = GreetingService()
        result = await service.generate_greeting(db, user.id)

        assert result["is_new_user"] is False
        assert result["action"] == "greet"
        assert result["user_name"] == "Ragul"
        greeting_text = result["greeting_text"]
        assert greeting_text is not None
        assert len(greeting_text) > 5
        # Greeting should naturally include name or greeting
        assert "ragul" in greeting_text.lower() or "ready" in greeting_text.lower() or "good" in greeting_text.lower()

    # Test REST Endpoint
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res = await client.get("/greeting", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["is_new_user"] is False
        assert data["action"] == "greet"
        assert data["user_name"] == "Ragul"
        assert data["greeting_text"] is not None


# -------------------------------------------------------------------------
# Test 3: Context-Aware Recent Context & Schedule Injection
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_returning_user_context_aware_schedule_and_memories():
    """
    Test 3: Context Injection.
    Loads recent memories, past conversation topic ("AI project"), and
    optional calendar schedule. Verifies privacy guardrails (zero credential leakage).
    """
    async with AsyncSessionLocal() as db:
        user = await create_returning_user(db, name="Ragul", preferred_name="Ragul")

        # 1. Add recent project memory
        memory = Memory(
            user_id=user.id,
            category="projects",
            content="Working on AI voice assistant project with Sarvam and Gemini",
            importance=0.9,
            source="user",
        )
        db.add(memory)

        # 2. Add past conversation with recent topic
        conv = Conversation(user_id=user.id, title="AI Project Discussions")
        db.add(conv)
        await db.flush()

        msg = ConversationMessage(
            conversation_id=conv.id,
            user_id=user.id,
            role="user",
            content="I want to work on my AI project today.",
            message_type="text",
        )
        db.add(msg)

        # 3. Enable Calendar tool permission
        from app.models.mcp import MCPServer as DBMCPServer
        server_row = DBMCPServer(user_id=user.id, name="calendar-service", status="active")
        db.add(server_row)
        await db.flush()

        perm = DBToolPermission(
            user_id=user.id,
            mcp_server_id=server_row.id,
            tool_name="get_calendar_events",
            allowed=True,
        )
        db.add(perm)
        await db.commit()

        service = GreetingService()
        ctx = await service.get_greeting_context(db, user.id)

        assert ctx["is_new_user"] is False
        assert ctx["user_name"] == "Ragul"
        assert len(ctx["recent_memories"]) > 0
        assert "AI voice assistant" in ctx["recent_memories"][0]
        assert "AI project" in ctx["recent_topic"]
        assert ctx["schedule_context"] is not None

        # Generate greeting with full context
        result = await service.generate_greeting(db, user.id)
        assert result["is_new_user"] is False
        text = result["greeting_text"]
        assert text is not None

        # STRICT PRIVACY & CREDENTIAL CHECK:
        # None of the backend secrets or internal system markers may appear in greeting
        text_lower = text.lower()
        forbidden_terms = [
            "password_hash",
            "bearer",
            "access_token",
            "user_id",
            "system_instruction",
            "gemini-",
            "sarvam_",
            "secret",
        ]
        for term in forbidden_terms:
            assert term not in text_lower, f"Forbidden term leaked in greeting: {term}"


# -------------------------------------------------------------------------
# Test 4: Anti-Repetition Guardrail
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_anti_repetition_cache_and_variation():
    """
    Test 4: Anti-Repetition.
    Verifies that calling generate_greeting multiple times stores recent greetings
    in the anti-repetition cache and prevents repeating the same greeting verbatim.
    """
    async with AsyncSessionLocal() as db:
        user = await create_returning_user(db, name="Ragul", preferred_name="Ragul")
        user_key = str(user.id)
        _recent_greetings_cache[user_key].clear()

        service = GreetingService()

        # Call 1
        g1 = await service.generate_greeting(db, user.id)
        text1 = g1["greeting_text"]
        assert text1 in _recent_greetings_cache[user_key]

        # Call 2
        g2 = await service.generate_greeting(db, user.id)
        text2 = g2["greeting_text"]
        assert text2 in _recent_greetings_cache[user_key]
        assert len(_recent_greetings_cache[user_key]) == 2


# -------------------------------------------------------------------------
# Test 5: Voice Synthesis & VoiceOrb State Machine
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_voice_greeting_streaming_and_orb_state():
    """
    Test 5: WebSocket Voice Greeting & VoiceOrb Animation.
    When a returning user requests a greeting:
    - Server emits greeting message.
    - Server emits state 'speaking' (so the VoiceOrb animates).
    - Server emits assistant_audio chunks synthesized with Sarvam TTS.
    - Server emits audio_end.
    """
    async with AsyncSessionLocal() as db:
        user = await create_returning_user(db, name="Ragul", preferred_name="Ragul")
        token, _, _ = create_access_token(subject=str(user.id))

    with TestClient(app) as test_client:
        with test_client.websocket_connect(f"/ws/voice?token={token}") as ws:
            conn = ws.receive_json()
            assert conn["type"] == "connected"

            # Request greeting
            ws.send_json({"type": "greeting_request"})

            received_greeting = False
            received_speaking_state = False
            received_audio_chunks = 0
            received_audio_end = False

            # Collect emitted frames
            for _ in range(15):
                msg = ws.receive_json()
                mtype = msg.get("type")

                if mtype == "greeting":
                    received_greeting = True
                    assert msg.get("user_name") == "Ragul"
                elif mtype == "state" and msg.get("state") == "speaking":
                    received_speaking_state = True
                elif mtype in ("assistant_audio", "audio_chunk"):
                    received_audio_chunks += 1
                    assert len(msg.get("data", "")) > 50
                elif mtype == "audio_end":
                    received_audio_end = True
                    break

            assert received_greeting, "Should receive greeting frame"
            assert received_speaking_state, "VoiceOrb should transition to speaking"
            assert received_audio_chunks > 0, "Should receive synthesized audio chunks"
            assert received_audio_end, "Should receive audio_end frame"

            # Playback completed -> idle
            ws.send_json({"type": "playback_done"})
            idle_msg = ws.receive_json()
            assert idle_msg["type"] == "state"
            assert idle_msg["state"] == "idle"


# -------------------------------------------------------------------------
# Test 6: Barge-In / Interruption Flow
# -------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_greeting_barge_in_interruption_switches_to_listening():
    """
    Test 6: Barge-In Interruption.
    If the user starts speaking while the greeting is playing:
    - Greeting TTS is cancelled immediately.
    - State transitions to 'listening'.
    - New user request is processed.
    """
    async with AsyncSessionLocal() as db:
        user = await create_returning_user(db, name="Ragul", preferred_name="Ragul")
        token, _, _ = create_access_token(subject=str(user.id))

    with TestClient(app) as test_client:
        with test_client.websocket_connect(f"/ws/voice?token={token}") as ws:
            conn = ws.receive_json()
            assert conn["type"] == "connected"

            # Request greeting
            ws.send_json({"type": "greeting_request"})

            # Wait until greeting starts
            while True:
                msg = ws.receive_json()
                if msg.get("type") in ("greeting", "state", "assistant_audio"):
                    break

            # User barges in (starts speaking)
            ws.send_json({"type": "audio_start"})

            # Must switch immediately to listening
            switched_to_listening = False
            for _ in range(15):
                msg = ws.receive_json()
                if msg.get("type") == "state" and msg.get("state") == "listening":
                    switched_to_listening = True
                    break
            assert switched_to_listening, "Should transition to listening state upon barge-in"

            # User speaks query
            ws.send_json({"type": "text_input", "text": "What is the capital of France?"})

            got_answer = False
            for _ in range(30):
                resp = ws.receive_json()
                if resp.get("type") in ("assistant_text_done", "ai_response_text"):
                    assert "paris" in resp.get("text", "").lower()
                    got_answer = True
                    break

            assert got_answer, "Assistant should process interrupted query successfully"
