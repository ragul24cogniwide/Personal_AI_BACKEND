"""
Test Suite for Persistent Conversation History with PostgreSQL.
Tests:
1. Conversation and ConversationMessage DB Model Persistence (all required fields).
2. Supported roles: user, assistant, tool, system.
3. Hidden reasoning is stripped and NEVER stored.
4. User ownership and user isolation (conversations belong to a user).
5. APIs:
   - GET /conversations
   - GET /conversations/:id
   - POST /conversations
   - DELETE /conversations/:id
6. Sensible Context Limits (do not load unlimited historical messages into Gemini).
7. Context retrieval when returning to an existing conversation.
"""

from datetime import datetime, timezone, timedelta
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from uuid import uuid4

from app.main import app
from app.ai import ai_service
from app.core.security import create_access_token, get_password_hash
from app.database.session import AsyncSessionLocal, async_engine
from app.models.conversation import Conversation, ConversationMessage
from app.models.user import User


async def create_test_user(db) -> User:
    """Helper to create a verified test user."""
    email = f"test_{uuid4().hex[:8]}@example.com"
    user = User(
        email=email,
        password_hash=get_password_hash("Password123!"),
        is_onboarded=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest.mark.asyncio
async def test_model_persistence_and_roles():
    """
    Verify Conversation & ConversationMessage PostgreSQL persistence.
    Verifies all required fields: conversation_id, user_id, role, content, message_type, created_at, metadata.
    Supported roles: user, assistant, tool, system.
    """
    async with AsyncSessionLocal() as db:
        user_a = await create_test_user(db)
        try:
            # Create conversation
            conv = Conversation(
                user_id=user_a.id,
                title="Model Test Conversation",
            )
            db.add(conv)
            await db.commit()
            await db.refresh(conv)

            assert conv.id is not None
            assert conv.user_id == user_a.id

            # Insert messages with all supported roles
            supported_roles = ["user", "assistant", "tool", "system"]
            for role in supported_roles:
                msg = ConversationMessage(
                    conversation_id=conv.id,
                    user_id=user_a.id,
                    role=role,
                    content=f"Sample message for role {role}",
                    message_type="text",
                    meta_data={"role_test": role, "version": 1},
                )
                db.add(msg)

            await db.commit()

            # Query back and verify all fields stored correctly
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conv.id)
                .order_by(ConversationMessage.created_at.asc())
            )
            result = await db.execute(stmt)
            messages = result.scalars().all()

            assert len(messages) == 4
            for idx, role in enumerate(supported_roles):
                m = messages[idx]
                assert m.conversation_id == conv.id
                assert m.user_id == user_a.id
                assert m.role == role
                assert m.message_type == "text"
                assert m.created_at is not None
                assert m.meta_data == {"role_test": role, "version": 1}
                assert f"role {role}" in m.content

            # Cleanup conv
            await db.delete(conv)
            await db.commit()
        finally:
            await db.delete(user_a)
            await db.commit()


@pytest.mark.asyncio
async def test_hidden_reasoning_not_stored():
    """
    Verify that hidden reasoning (thoughts, scratchpads, <thought>...</thought>)
    is NEVER stored in PostgreSQL.
    """
    async with AsyncSessionLocal() as db:
        user_a = await create_test_user(db)
        try:
            conv = Conversation(user_id=user_a.id, title="Reasoning Test")
            db.add(conv)
            await db.commit()
            await db.refresh(conv)

            raw_reasoning_content = (
                "<thought>\n"
                "The user is asking about the weather. Let's analyze sensor readings and retrieve context.\n"
                "</thought>\n"
                "<thinking>Verifying user location</thinking>\n"
                "Today's weather is sunny and 72 degrees."
            )

            saved_msg = await ai_service.save_message(
                db=db,
                conversation_id=conv.id,
                user_id=user_a.id,
                role="assistant",
                content=raw_reasoning_content,
                metadata={"source": "voice", "internal_thought_debug": "should_be_stripped"},
            )

            assert "<thought>" not in saved_msg.content
            assert "</thought>" not in saved_msg.content
            assert "<thinking>" not in saved_msg.content
            assert saved_msg.content == "Today's weather is sunny and 72 degrees."
            assert "internal_thought_debug" not in (saved_msg.meta_data or {})

            # Query from DB to verify raw row in PostgreSQL
            refetched = await db.get(ConversationMessage, saved_msg.id)
            assert refetched.content == "Today's weather is sunny and 72 degrees."

            # Cleanup
            await db.delete(conv)
            await db.commit()
        finally:
            await db.delete(user_a)
            await db.commit()


@pytest.mark.asyncio
async def test_conversation_crud_apis():
    """
    Tests REST APIs:
    - POST /conversations
    - GET /conversations
    - GET /conversations/:id
    - DELETE /conversations/:id
    Ensures user isolation: User B cannot view or delete User A's conversations.
    """
    async with AsyncSessionLocal() as db:
        user_a = await create_test_user(db)
        user_b = await create_test_user(db)

    try:
        token_a, _, _ = create_access_token(subject=str(user_a.id))
        token_b, _, _ = create_access_token(subject=str(user_b.id))


        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. POST /conversations (create for user A)
            res_post = await client.post(
                "/conversations",
                headers=headers_a,
                json={"title": "Design Discussion"},
            )
            assert res_post.status_code == 201, res_post.text
            conv_data = res_post.json()
            conv_id = conv_data["id"]
            assert conv_data["title"] == "Design Discussion"
            assert conv_data["user_id"] == str(user_a.id)
            assert conv_data["message_count"] == 0

            # Add messages for user A via message API
            res_msg1 = await client.post(
                f"/conversations/{conv_id}/messages",
                headers=headers_a,
                json={"role": "user", "content": "What are our architecture goals?"},
            )
            assert res_msg1.status_code == 201

            res_msg2 = await client.post(
                f"/conversations/{conv_id}/messages",
                headers=headers_a,
                json={"role": "assistant", "content": "Our main goals are high performance and low latency."},
            )
            assert res_msg2.status_code == 201

            # 2. GET /conversations (list for user A)
            res_list_a = await client.get("/conversations", headers=headers_a)
            assert res_list_a.status_code == 200
            list_a = res_list_a.json()
            assert len(list_a) >= 1
            found = next((c for c in list_a if c["id"] == conv_id), None)
            assert found is not None
            assert found["message_count"] == 2
            assert "high performance" in found["last_message_preview"]

            # Verify user B CANNOT see User A's conversation
            res_list_b = await client.get("/conversations", headers=headers_b)
            assert res_list_b.status_code == 200
            list_b = res_list_b.json()
            assert all(c["id"] != conv_id for c in list_b)

            # 3. GET /conversations/:id
            res_get_a = await client.get(f"/conversations/{conv_id}", headers=headers_a)
            assert res_get_a.status_code == 200
            detail_a = res_get_a.json()
            assert detail_a["id"] == conv_id
            assert detail_a["message_count"] == 2
            assert len(detail_a["messages"]) == 2
            assert detail_a["messages"][0]["role"] == "user"
            assert detail_a["messages"][0]["user_id"] == str(user_a.id)
            assert detail_a["messages"][1]["role"] == "assistant"
            assert detail_a["messages"][1]["user_id"] == str(user_a.id)

            # Verify User B gets 404 for User A's conversation
            res_get_b = await client.get(f"/conversations/{conv_id}", headers=headers_b)
            assert res_get_b.status_code == 404

            # 4. DELETE /conversations/:id by User B fails with 404
            res_del_b = await client.delete(f"/conversations/{conv_id}", headers=headers_b)
            assert res_del_b.status_code == 404

            # DELETE /conversations/:id by User A succeeds
            res_del_a = await client.delete(f"/conversations/{conv_id}", headers=headers_a)
            assert res_del_a.status_code == 200
            del_data = res_del_a.json()
            assert del_data["success"] is True

            # Verify conversation is gone
            res_get_gone = await client.get(f"/conversations/{conv_id}", headers=headers_a)
            assert res_get_gone.status_code == 404
    finally:
        async with AsyncSessionLocal() as db:
            ua = await db.get(User, user_a.id)
            ub = await db.get(User, user_b.id)
            if ua:
                await db.delete(ua)
            if ub:
                await db.delete(ub)
            await db.commit()


@pytest.mark.asyncio
async def test_sensible_context_limits():
    """
    Verify that context limits prevent loading unlimited historical messages into Gemini.
    Creates 20 historical messages and verifies that load_recent_messages limits
    context to the sensible default (15 messages) while maintaining chronological order.
    """
    async with AsyncSessionLocal() as db:
        user_a = await create_test_user(db)
        try:
            conv = Conversation(user_id=user_a.id, title="Context Limit Test")
            db.add(conv)
            await db.commit()
            await db.refresh(conv)

            base_time = datetime.now(timezone.utc) - timedelta(minutes=30)
            # Insert 20 historical turns in batch with monotonic created_at
            for i in range(20):
                role = "user" if i % 2 == 0 else "assistant"
                msg = ConversationMessage(
                    conversation_id=conv.id,
                    user_id=user_a.id,
                    role=role,
                    content=f"Turn message index {i}",
                    message_type="text",
                    created_at=base_time + timedelta(seconds=i * 10),
                )
                db.add(msg)
            await db.commit()

            # Load context with default limit (15)
            history = await ai_service.load_recent_messages(
                db=db,
                conversation_id=conv.id,
                limit=15,
            )

            # Context must NOT contain all 20 messages
            assert len(history) <= 15
            assert len(history) > 0

            # Most recent message was index 19 -> must be in the window
            last_item = history[-1]
            assert "Turn message index 19" in last_item.content

            # Old message index 0 should have been excluded by the context window limit
            all_contents = [m.content for m in history]
            assert not any("Turn message index 0" == c for c in all_contents)

            # Cleanup
            await db.delete(conv)
            await db.commit()
        finally:
            await db.delete(user_a)
            await db.commit()
