"""
Integration & Unit Test Suite for Long-Term Personal Memory with PostgreSQL + pgvector.

Tests:
1. Memory model persistence in PostgreSQL with pgvector embeddings (1536 dimensions).
2. All 12 supported memory categories (identity, preferences, personality, goals, habits,
   fitness, learning, work, projects, routine, important_events, other).
3. Memory management APIs:
   - GET /memories (listing, category filtering, search)
   - POST /memories (creation with automatic pgvector embedding)
   - PATCH /memories/:id (updating fields and re-embedding)
   - DELETE /memories/:id (deletion)
4. User isolation and privacy: User B cannot access User A's personal memories.
5. pgvector Semantic Similarity Search using cosine distance.
6. Memory extraction: Distinguishes durable personal facts from transient conversation
   chatter (weather, time, math).
"""

from datetime import datetime, timezone
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from uuid import uuid4

from app.main import app
from app.ai.embedding import embedding_service
from app.core.security import create_access_token
from app.database.session import AsyncSessionLocal
from app.models.memory import Memory, MEMORY_CATEGORIES
from app.models.user import User
from app.schemas.memory import MemoryCreate, MemoryUpdate
from app.services.memory import memory_service


async def create_test_user(db) -> User:
    """Helper to create a test user."""
    email = f"mem_test_{uuid4().hex[:8]}@example.com"
    user = User(
        email=email,
        password_hash="test_hash_pw",
        is_onboarded=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest.mark.asyncio
async def test_memory_model_pgvector_persistence():
    """
    1. Verifies that Memory model persists in PostgreSQL with all required fields:
       id, user_id, category, content, importance, source, embedding, created_at, updated_at, metadata.
    2. Validates pgvector 1536-dimensional embedding storage and retrieval.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            # Generate valid 1536-dim embedding vector
            embedding_vec = await embedding_service.generate_embedding("User prefers dark mode interfaces")
            assert len(embedding_vec) == 1536

            memory = Memory(
                user_id=user.id,
                category="preferences",
                content="User prefers dark mode interfaces",
                importance=0.85,
                source="manual",
                meta_data={"theme": "dark", "platform": "mobile"},
                embedding=embedding_vec,
            )
            db.add(memory)
            await db.commit()
            await db.refresh(memory)

            # Query row directly from PostgreSQL
            stmt = select(Memory).where(Memory.id == memory.id)
            fetched = (await db.execute(stmt)).scalars().first()

            assert fetched is not None
            assert fetched.id == memory.id
            assert fetched.user_id == user.id
            assert fetched.category == "preferences"
            assert fetched.content == "User prefers dark mode interfaces"
            assert fetched.importance == 0.85
            assert fetched.source == "manual"
            assert fetched.meta_data == {"theme": "dark", "platform": "mobile"}
            assert fetched.embedding is not None
            assert len(fetched.embedding) == 1536
            assert isinstance(fetched.created_at, datetime)
            assert isinstance(fetched.updated_at, datetime)

            # Cleanup
            await db.delete(memory)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_all_twelve_categories_support():
    """
    Verifies that all 12 required categories are valid and persist correctly:
    identity, preferences, personality, goals, habits, fitness, learning,
    work, projects, routine, important_events, other.
    """
    expected_categories = [
        "identity",
        "preferences",
        "personality",
        "goals",
        "habits",
        "fitness",
        "learning",
        "work",
        "projects",
        "routine",
        "important_events",
        "other",
    ]

    for cat in expected_categories:
        assert cat in MEMORY_CATEGORIES

    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            created_memories = []
            for cat in expected_categories:
                mem = await memory_service.create_memory(
                    db=db,
                    user_id=user.id,
                    data=MemoryCreate(
                        category=cat,  # type: ignore
                        content=f"Fact about user {cat}",
                        importance=0.6,
                        source="test",
                    ),
                )
                assert mem.category == cat
                created_memories.append(mem)

            assert len(created_memories) == 12

            # Fetch list and check category aggregations
            memories_list, total, counts = await memory_service.get_memories(
                db=db,
                user_id=user.id,
            )
            assert total == 12
            for cat in expected_categories:
                assert counts.get(cat, 0) == 1

            # Cleanup
            for m in created_memories:
                await db.delete(m)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_memory_crud_apis_and_user_isolation():
    """
    Tests REST APIs:
    - POST /memories
    - GET /memories
    - GET /memories/:id
    - PATCH /memories/:id
    - DELETE /memories/:id
    Verifies User Isolation: User B cannot view, modify, or delete User A's memories.
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
            # 1. POST /memories (User A creates a memory)
            res_post = await client.post(
                "/memories",
                headers=headers_a,
                json={
                    "category": "goals",
                    "content": "User wants to run a marathon by November 2026",
                    "importance": 0.9,
                    "source": "manual",
                },
            )
            assert res_post.status_code == 201, res_post.text
            mem_data = res_post.json()
            mem_id = mem_data["id"]
            assert mem_data["category"] == "goals"
            assert mem_data["content"] == "User wants to run a marathon by November 2026"
            assert mem_data["importance"] == 0.9
            assert mem_data["has_embedding"] is True
            assert mem_data["user_id"] == str(user_a.id)

            # 2. GET /memories (User A lists memories)
            res_list_a = await client.get("/memories", headers=headers_a)
            assert res_list_a.status_code == 200
            list_data_a = res_list_a.json()
            assert list_data_a["total"] >= 1
            assert any(item["id"] == mem_id for item in list_data_a["items"])
            assert list_data_a["category_counts"].get("goals", 0) >= 1

            # User B lists memories -> Must NOT see User A's memory
            res_list_b = await client.get("/memories", headers=headers_b)
            assert res_list_b.status_code == 200
            list_data_b = res_list_b.json()
            assert all(item["id"] != mem_id for item in list_data_b["items"])

            # 3. GET /memories/:id (User A retrieves memory)
            res_get_a = await client.get(f"/memories/{mem_id}", headers=headers_a)
            assert res_get_a.status_code == 200
            assert res_get_a.json()["id"] == mem_id

            # User B attempts to access User A's memory -> 404 Not Found
            res_get_b = await client.get(f"/memories/{mem_id}", headers=headers_b)
            assert res_get_b.status_code == 404

            # 4. PATCH /memories/:id (User A updates memory)
            res_patch_a = await client.patch(
                f"/memories/{mem_id}",
                headers=headers_a,
                json={
                    "importance": 0.95,
                    "content": "User aims to complete a marathon in under 4 hours",
                },
            )
            assert res_patch_a.status_code == 200
            updated_data = res_patch_a.json()
            assert updated_data["importance"] == 0.95
            assert "under 4 hours" in updated_data["content"]
            assert updated_data["has_embedding"] is True

            # User B attempts to patch User A's memory -> 404 Not Found
            res_patch_b = await client.patch(
                f"/memories/{mem_id}",
                headers=headers_b,
                json={"importance": 0.1},
            )
            assert res_patch_b.status_code == 404

            # 5. DELETE /memories/:id (User B attempts to delete User A's memory -> 404)
            res_del_b = await client.delete(f"/memories/{mem_id}", headers=headers_b)
            assert res_del_b.status_code == 404

            # User A deletes memory -> 200 OK
            res_del_a = await client.delete(f"/memories/{mem_id}", headers=headers_a)
            assert res_del_a.status_code == 200
            assert res_del_a.json()["deleted"] is True

            # Refetch to confirm deletion
            res_get_deleted = await client.get(f"/memories/{mem_id}", headers=headers_a)
            assert res_get_deleted.status_code == 404

    finally:
        async with AsyncSessionLocal() as db:
            await db.delete(user_a)
            await db.delete(user_b)
            await db.commit()


@pytest.mark.asyncio
async def test_pgvector_similarity_search():
    """
    Tests pgvector cosine similarity search over personal memories.
    Inserts distinct memories and verifies that a semantic query ranks the most relevant memory first.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            m1 = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="preferences",
                    content="User prefers concise answers without fluff.",
                    importance=0.9,
                ),
            )
            m2 = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="projects",
                    content="User is building an advanced voice assistant app.",
                    importance=0.8,
                ),
            )
            m3 = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="fitness",
                    content="User runs 5 kilometers every morning.",
                    importance=0.6,
                ),
            )

            # Query semantically: "How long should your responses be?"
            # Memory 1 (concise answers) must be ranked first
            results = await memory_service.search_similar_memories(
                db=db,
                user_id=user.id,
                query="How long should your responses be?",
                limit=3,
                min_similarity=0.3,
            )

            assert len(results) > 0
            top_memory, top_sim = results[0]
            assert top_memory.id == m1.id
            assert "concise" in top_memory.content.lower()
            assert top_sim > 0.0

            # Query semantically: "Tell me about my coding software project"
            # Memory 2 (voice assistant project) must be ranked first
            results_proj = await memory_service.search_similar_memories(
                db=db,
                user_id=user.id,
                query="Tell me about my coding software project",
                limit=3,
                min_similarity=0.3,
            )
            assert len(results_proj) > 0
            assert results_proj[0][0].id == m2.id

            # Cleanup
            for m in [m1, m2, m3]:
                await db.delete(m)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_distinguish_temporary_context_vs_durable_memory():
    """
    Tests memory extraction logic:
    - Durable statements (e.g. "I prefer concise answers", "I am working on a voice assistant")
      MUST be recognized as long-term memories.
    - Transient queries (e.g. "What is the weather?", "What time is it?", "Calculate 4 * 12")
      MUST NOT be saved as memories.
    """
    async with AsyncSessionLocal() as db:
        user = await create_test_user(db)
        try:
            # Case 1: Purely transient conversation
            transient_conversation = (
                "User: What is the weather today?\n"
                "Assistant: It is sunny and 72 degrees.\n"
                "User: Great, what time is it right now?\n"
                "Assistant: It is 2:30 PM."
            )

            extracted_transient, saved_transient = await memory_service.extract_and_save_memories(
                db=db,
                user_id=user.id,
                text=transient_conversation,
                auto_save=True,
            )

            # Transient conversation should yield ZERO memories
            assert len(extracted_transient) == 0
            assert len(saved_transient) == 0

            # Case 2: Conversation containing durable personal facts
            durable_conversation = (
                "User: What is the weather?\n"
                "Assistant: It is 65 degrees and cloudy.\n"
                "User: Thanks. By the way, I prefer concise answers and I am currently working on a voice assistant project."
            )

            extracted_durable, saved_durable = await memory_service.extract_and_save_memories(
                db=db,
                user_id=user.id,
                text=durable_conversation,
                auto_save=True,
            )

            # Durable conversation MUST extract personal facts
            assert len(extracted_durable) >= 1
            extracted_categories = [item.category for item in extracted_durable]
            # Should contain preferences or projects/work
            assert any(cat in ["preferences", "projects", "work"] for cat in extracted_categories)

            # Must be saved to PostgreSQL
            assert len(saved_durable) >= 1
            saved_ids = [m.id for m in saved_durable]

            # Verify in DB
            stmt = select(Memory).where(Memory.id.in_(saved_ids))
            db_memories = (await db.execute(stmt)).scalars().all()
            assert len(db_memories) == len(saved_durable)

            # Cleanup
            for m in db_memories:
                await db.delete(m)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()
