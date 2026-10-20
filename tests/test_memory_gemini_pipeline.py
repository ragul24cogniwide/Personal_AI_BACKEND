"""
Integration & E2E Test Suite for Memory-Connected Gemini Pipeline.
Tests:
1. Relevance Determination:
   - "What should I do for my workout today?" -> retrieves fitness memories, excludes unrelated memories.
   - "What is 25 times 4?" / "What is the capital of France?" -> generic query -> retrieves 0 memories.
2. ContextBuilder Priority Order:
   - Order: Current user request > Safety > Current preferences > Relevant memories > User profile > History > Personality.
   - Current user request overrides older memories.
3. Unknown Information Handling:
   - When personal facts are unrecorded, assistant admits it does not know and never invents memories.
4. End-to-End Pipeline:
   - User request -> determine context -> pgvector search -> relevant memories -> user profile -> conversation history -> Gemini -> response.
"""

import pytest
from sqlalchemy import select
from uuid import uuid4

from app.ai import ai_service
from app.ai.context_builder import context_builder
from app.database.session import AsyncSessionLocal
from app.models.conversation import Conversation
from app.models.memory import Memory
from app.models.user import User, UserProfile, UserPreference
from app.schemas.memory import MemoryCreate
from app.services.memory import memory_service


async def create_pipeline_test_user(db) -> User:
    """Creates a test user with profile and preferences."""
    email = f"pipeline_test_{uuid4().hex[:8]}@example.com"
    user = User(
        email=email,
        password_hash="test_pw_hash",
        is_onboarded=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    profile = UserProfile(
        user_id=user.id,
        name="Ragul",
        preferred_name="Ragul",
        occupation="AI Engineer",
        bio="Passionate about AI voice technology",
        communication_style="Direct and concise",
    )
    db.add(profile)

    preferences = UserPreference(
        user_id=user.id,
        assistant_tone="calm",
        verbosity="concise",
        language="en",
    )
    db.add(preferences)
    await db.commit()
    return user


@pytest.mark.asyncio
async def test_relevance_determination_and_selective_retrieval():
    """
    Verifies that:
    - Domain queries (e.g. workout) retrieve relevant memories (fitness).
    - Unrelated memories (e.g. work, food) are NOT retrieved.
    - Purely generic queries (e.g. math, capital of France) retrieve ZERO memories.
    """
    async with AsyncSessionLocal() as db:
        user = await create_pipeline_test_user(db)
        try:
            # 1. Seed diverse memories
            m_fitness = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="fitness",
                    content="User runs 5 kilometers three times a week and focuses on cardio endurance.",
                    importance=0.9,
                ),
            )
            m_work = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="work",
                    content="User is developing a voice assistant using FastAPI and React Native.",
                    importance=0.8,
                ),
            )
            m_habits = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="habits",
                    content="User drinks black coffee every morning at 7:30 AM.",
                    importance=0.5,
                ),
            )

            # Case A: Workout query -> MUST retrieve fitness memory
            retrieved_workout = await context_builder.retrieve_relevant_memories(
                db=db,
                user_id=user.id,
                user_request="What should I do for my workout today?",
            )
            assert len(retrieved_workout) >= 1
            retrieved_ids = [m.id for m in retrieved_workout]
            assert m_fitness.id in retrieved_ids
            # Must NOT retrieve unrelated work memory
            assert m_work.id not in retrieved_ids

            # Case B: Work/project query -> MUST retrieve work memory, NOT fitness
            retrieved_work = await context_builder.retrieve_relevant_memories(
                db=db,
                user_id=user.id,
                user_request="Can you give me an update on my current software project?",
            )
            assert len(retrieved_work) >= 1
            work_retrieved_ids = [m.id for m in retrieved_work]
            assert m_work.id in work_retrieved_ids
            assert m_fitness.id not in work_retrieved_ids

            # Case C: Generic Math query -> MUST retrieve ZERO memories
            retrieved_math = await context_builder.retrieve_relevant_memories(
                db=db,
                user_id=user.id,
                user_request="What is 25 multiplied by 4?",
            )
            assert len(retrieved_math) == 0

            # Case D: Generic Trivia query -> MUST retrieve ZERO memories
            retrieved_trivia = await context_builder.retrieve_relevant_memories(
                db=db,
                user_id=user.id,
                user_request="What is the capital of France?",
            )
            assert len(retrieved_trivia) == 0

            # Cleanup
            for m in [m_fitness, m_work, m_habits]:
                await db.delete(m)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_current_user_request_overrides_old_memories():
    """
    Verifies the priority order:
    1. Current user request
    2. Safety
    3. Explicit current preferences
    4. Relevant memories
    ...
    If the user's current request explicitly updates or contradicts older memories
    (e.g., switched from running to swimming), the current instruction MUST override.
    """
    async with AsyncSessionLocal() as db:
        user = await create_pipeline_test_user(db)
        try:
            m_run = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="fitness",
                    content="User runs 5k every Monday and Wednesday morning.",
                    importance=0.8,
                ),
            )

            # Query where user explicitly overrides the running routine with swimming
            user_override_request = (
                "Actually, I have completely stopped running and now only do swimming laps. "
                "What should I do for my workout today?"
            )

            conv = await ai_service.get_or_create_conversation(db, user.id)

            response = await ai_service.generate_response(
                db=db,
                user_id=user.id,
                user_text=user_override_request,
                conversation_id=conv.id,
            )

            response_lower = response.lower()
            # The model MUST focus on swimming / laps
            assert "swim" in response_lower or "pool" in response_lower or "laps" in response_lower
            # The model MUST NOT insist that the user go for a run
            assert "you should go for your 5k run" not in response_lower

            # Cleanup
            await db.delete(m_run)
            await db.delete(conv)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_unknown_information_never_invents_memories():
    """
    Verifies that when asked about personal information that is NOT recorded,
    the assistant admits it does not know and does NOT hallucinate or fabricate facts.
    """
    async with AsyncSessionLocal() as db:
        user = await create_pipeline_test_user(db)
        try:
            # Memory has NO information about pets or dogs
            m_pref = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="preferences",
                    content="User prefers concise answers.",
                    importance=0.7,
                ),
            )

            conv = await ai_service.get_or_create_conversation(db, user.id)

            # User asks for unrecorded pet name
            response = await ai_service.generate_response(
                db=db,
                user_id=user.id,
                user_text="What is my pet dog's name and favorite toy?",
                conversation_id=conv.id,
            )

            response_lower = response.lower()
            # Must admit lack of knowledge
            unknown_phrases = [
                "don't know",
                "do not know",
                "don't have",
                "do not have",
                "haven't shared",
                "have not shared",
                "no record",
                "not recorded",
                "haven't mentioned",
                "have not mentioned",
            ]
            assert any(phrase in response_lower for phrase in unknown_phrases)

            # Cleanup
            await db.delete(m_pref)
            await db.delete(conv)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()


@pytest.mark.asyncio
async def test_e2e_memory_retrieval_and_gemini_personalization():
    """
    End-to-End Test:
    User request -> pgvector memory search -> relevant memory -> ContextBuilder -> Gemini -> Personalized response.
    """
    async with AsyncSessionLocal() as db:
        user = await create_pipeline_test_user(db)
        try:
            # Seed personal goal memory
            m_goal = await memory_service.create_memory(
                db=db,
                user_id=user.id,
                data=MemoryCreate(
                    category="goals",
                    content="User's primary 2026 goal is to launch their AI voice assistant on Product Hunt by December.",
                    importance=0.95,
                ),
            )

            conv = await ai_service.get_or_create_conversation(db, user.id)

            response = await ai_service.generate_response(
                db=db,
                user_id=user.id,
                user_text="What is my top launch goal for this year?",
                conversation_id=conv.id,
            )

            response_lower = response.lower()
            # Must cite the launch goal
            assert "product hunt" in response_lower or "voice assistant" in response_lower or "december" in response_lower

            # Cleanup
            await db.delete(m_goal)
            await db.delete(conv)
            await db.commit()
        finally:
            await db.delete(user)
            await db.commit()
