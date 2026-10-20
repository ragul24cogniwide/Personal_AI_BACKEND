"""
Comprehensive Test Suite for STEP 8: Gemini as the Primary LLM Brain.
Tests all required scenarios:
1. "Hello"
2. "Who are you?"
3. "What can you help me with?"
4. Multi-turn conversation (context retention & DB persistence)
5. Long response (streaming chunk handling)
6. Gemini API failure (invalid key / graceful error)
7. Timeout handling
8. Invalid response handling
9. Full WebSocket streaming turn test
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from app.core.config import settings
from app.ai import (
    ai_service,
    ChatMessage,
    GeminiProvider,
    llm_manager,
    LLMAPIError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMTimeoutError,
)
from app.database.session import AsyncSessionLocal
from app.models.conversation import Conversation, ConversationMessage
from app.models.user import User


@pytest.mark.asyncio
async def test_1_hello():
    """Test 1: 'Hello' prompt generates friendly, conversational greeting."""
    provider = llm_manager.get_provider()
    messages = [ChatMessage(role="user", content="Hello")]
    
    response = await provider.generate(messages=messages)
    assert response.content, "Response content should not be empty"
    assert response.provider == "gemini"
    assert response.model == settings.GEMINI_MODEL
    print(f"\n[Test 1] 'Hello' -> Response: {response.content}")


@pytest.mark.asyncio
async def test_2_who_are_you():
    """Test 2: 'Who are you?' prompt identifies as Aura assistant."""
    provider = llm_manager.get_provider()
    messages = [ChatMessage(role="user", content="Who are you?")]
    
    response = await provider.generate(
        messages=messages,
        system_instruction="You are Aura, an intelligent, calm, and friendly personal AI voice assistant.",
    )
    assert response.content, "Response content should not be empty"
    content_lower = response.content.lower()
    assert "aura" in content_lower or "assistant" in content_lower or "ai" in content_lower
    print(f"\n[Test 2] 'Who are you?' -> Response: {response.content}")


@pytest.mark.asyncio
async def test_3_what_can_you_help_me_with():
    """Test 3: 'What can you help me with?' explains assistant capabilities."""
    provider = llm_manager.get_provider()
    messages = [ChatMessage(role="user", content="What can you help me with?")]
    
    response = await provider.generate(messages=messages)
    assert response.content, "Response content should not be empty"
    assert len(response.content) > 20
    print(f"\n[Test 3] 'What can you help me with?' -> Response: {response.content}")


@pytest.mark.asyncio
async def test_4_multiturn_conversation():
    """Test 4: Multi-turn conversation retains context across turns and persists in PostgreSQL."""
    async with AsyncSessionLocal() as db:
        # Create a dedicated test conversation
        from sqlalchemy import select
        res = await db.execute(select(User).limit(1))
        user = res.scalar_one_or_none()
        assert user is not None, "Need at least one user in DB for testing"

        test_conv = Conversation(
            user_id=user.id,
            title="Multi-turn Test Session",
        )
        db.add(test_conv)
        await db.commit()
        await db.refresh(test_conv)

        # Turn 1: Tell assistant favorite color
        turn1_response = ""
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="My favorite color is emerald green.",
            conversation_id=test_conv.id,
        ):
            turn1_response += chunk

        assert turn1_response, "Turn 1 response should not be empty"
        print(f"\n[Test 4 - Turn 1] User: 'My favorite color is emerald green' -> Assistant: {turn1_response}")

        # Turn 2: Ask assistant what the favorite color is
        turn2_response = ""
        async for chunk in ai_service.process_turn_stream(
            db=db,
            user_id=user.id,
            user_text="What is my favorite color?",
            conversation_id=test_conv.id,
        ):
            turn2_response += chunk

        assert turn2_response, "Turn 2 response should not be empty"
        print(f"[Test 4 - Turn 2] User: 'What is my favorite color?' -> Assistant: {turn2_response}")
        assert "emerald" in turn2_response.lower() or "green" in turn2_response.lower(), (
            f"Expected 'emerald' or 'green' in context response, got: {turn2_response}"
        )

        # Verify both turns persisted in DB
        res_msgs = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == test_conv.id)
            .order_by(ConversationMessage.created_at.asc())
        )
        saved_messages = res_msgs.scalars().all()
        assert len(saved_messages) >= 4, f"Expected 4 messages (2 user + 2 assistant), found {len(saved_messages)}"
        assert saved_messages[0].role == "user"
        assert saved_messages[1].role == "assistant"
        assert saved_messages[2].role == "user"
        assert saved_messages[3].role == "assistant"
        print(f"[Test 4] Database verification: All {len(saved_messages)} messages properly stored in PostgreSQL!")


@pytest.mark.asyncio
async def test_5_long_response_streaming():
    """Test 5: Long response streaming verifies multiple chunks arrive in order."""
    provider = llm_manager.get_provider()
    messages = [
        ChatMessage(
            role="user",
            content="Write a detailed explanation of quantum computing in two paragraphs.",
        )
    ]
    
    chunks = []
    async for chunk in provider.stream(messages=messages):
        chunks.append(chunk)

    full_text = "".join(chunks)
    assert len(chunks) >= 2, f"Expected streaming to yield multiple chunks, got {len(chunks)}"
    assert len(full_text) > 100, "Long response should produce substantial text"
    print(f"\n[Test 5] Long response streamed in {len(chunks)} chunks, total length: {len(full_text)} characters.")


@pytest.mark.asyncio
async def test_6_gemini_api_failure():
    """Test 6: Gemini API failure raises LLMAPIError gracefully."""
    bad_provider = GeminiProvider(api_key="AIzaSy_INVALID_KEY_FOR_TESTING_1234567890")
    messages = [ChatMessage(role="user", content="Hello")]

    with pytest.raises(LLMAPIError) as exc_info:
        await bad_provider.generate(messages=messages)

    print(f"\n[Test 6] Caught expected LLMAPIError: {exc_info.value}")
    assert "Gemini" in str(exc_info.value) or "API" in str(exc_info.value) or "key" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_7_gemini_timeout():
    """Test 7: Timeout handling raises LLMTimeoutError when deadline exceeded."""
    provider = llm_manager.get_provider()
    messages = [ChatMessage(role="user", content="Tell me a long story")]

    # Set an impossibly small timeout (0.0001s)
    with pytest.raises(LLMTimeoutError) as exc_info:
        await provider.generate(messages=messages, timeout=0.0001)

    print(f"\n[Test 7] Caught expected LLMTimeoutError: {exc_info.value}")
    assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_8_invalid_response_handling():
    """Test 8: Invalid or empty response raises LLMInvalidResponseError."""
    provider = GeminiProvider()
    
    # Mock client returning candidate with None text and None content
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "SAFETY"
    mock_candidate.content = None
    mock_response = MagicMock(spec=["text", "candidates", "usage_metadata"])
    mock_response.text = None
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    provider._client = mock_client
    provider._loop = asyncio.get_running_loop()

    messages = [ChatMessage(role="user", content="Test prompt")]

    with pytest.raises(LLMInvalidResponseError) as exc_info:
        await provider.generate(messages=messages)

    print(f"\n[Test 8] Caught expected LLMInvalidResponseError: {exc_info.value}")
    assert "empty" in str(exc_info.value).lower() or "blocked" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_9_websocket_gemini_turn():
    """Test 9: End-to-end WebSocket test verifying thinking, speaking, deltas, and completion."""
    from starlette.testclient import TestClient
    from app.main import app
    from app.core.security import create_access_token
    from app.database.session import AsyncSessionLocal
    from app.models.user import User

    async with AsyncSessionLocal() as db:
        user = User(email=f"ws_test_{uuid4().hex[:8]}@example.com", password_hash="test_pw", is_onboarded=True)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        user_id = user.id

    token, _, _ = create_access_token(subject=str(user_id))
    with TestClient(app) as test_client:
        with test_client.websocket_connect(f"/ws/voice?token={token}") as ws:
            # 1. Connected acknowledgement
            conn_msg = ws.receive_json()
            assert conn_msg["type"] == "connected"
            assert conn_msg.get("llm_provider") == "gemini"

            # 2. Send text input
            ws.send_json({"type": "text_input", "text": "What is the speed of light? Answer in one sentence."})

            received_types = []
            deltas = []
            full_response = ""

            # Receive streamed turn events
            for _ in range(40):
                msg = ws.receive_json()
                m_type = msg.get("type")
                received_types.append(m_type)

                if m_type == "assistant_text_delta":
                    deltas.append(msg.get("text", ""))
                elif m_type in ("assistant_text_done", "ai_response_text"):
                    full_response = msg.get("text", "")
                elif m_type == "latency_metrics":
                    break

            assert len(full_response) > 0 or len(deltas) > 0, "Expected non-empty response"
            assert any(term in full_response.lower() for term in ["299", "300", "speed", "light", "meter", "second"])
            print("\n[Test 9] Full WebSocket turn test PASSED!")
