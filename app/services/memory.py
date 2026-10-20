"""
Memory Service implementing storage, pgvector similarity search, and extraction.
Guarantees user isolation and distinguishes temporary conversation from long-term memory.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embedding import embedding_service
from app.ai.memory_extractor import memory_extractor
from app.models.conversation import ConversationMessage
from app.models.memory import Memory, MEMORY_CATEGORIES
from app.schemas.memory import (
    ExtractedMemoryItem,
    MemoryCreate,
    MemoryResponse,
    MemoryUpdate,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Core service for managing long-term personal memories in PostgreSQL with pgvector.
    """

    async def create_memory(
        self,
        db: AsyncSession,
        user_id: UUID,
        data: MemoryCreate,
    ) -> Memory:
        """
        Creates a new personal memory with a 1536-dimensional pgvector embedding.
        """
        # Validate category
        category = data.category.lower() if data.category else "other"
        if category not in MEMORY_CATEGORIES:
            category = "other"

        # Generate embedding for content
        embedding_vec = await embedding_service.generate_embedding(data.content)

        memory = Memory(
            user_id=user_id,
            category=category,
            content=data.content.strip(),
            importance=data.importance,
            source=data.source,
            meta_data=data.metadata or {},
            embedding=embedding_vec,
        )

        db.add(memory)
        await db.commit()
        await db.refresh(memory)
        return memory

    async def get_memories(
        self,
        db: AsyncSession,
        user_id: UUID,
        category: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Tuple[Memory, Optional[float]]], int, Dict[str, int]]:
        """
        Retrieves user's memories with optional category filtering or semantic vector search.
        Returns:
            - List of (Memory, similarity_or_none)
            - Total matching count
            - Category counts aggregation for UI filter pills
        """
        # 1. Base query for counts by category for this user
        cat_counts_stmt = (
            select(Memory.category, func.count(Memory.id))
            .where(Memory.user_id == user_id)
            .group_by(Memory.category)
        )
        cat_counts_res = await db.execute(cat_counts_stmt)
        category_counts: Dict[str, int] = {row[0]: row[1] for row in cat_counts_res.all()}

        # 2. Main query
        if query and query.strip():
            # Semantic search via pgvector cosine distance
            query_vec = await embedding_service.generate_embedding(query.strip())
            cosine_dist = Memory.embedding.cosine_distance(query_vec).label("distance")

            stmt = (
                select(Memory, cosine_dist)
                .where(Memory.user_id == user_id)
                .where(Memory.embedding.isnot(None))
            )

            if category and category.lower() != "all":
                stmt = stmt.where(Memory.category == category.lower())

            # Count matching
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total_count = (await db.execute(count_stmt)).scalar() or 0

            # Order by smallest distance (highest similarity)
            stmt = stmt.order_by(cosine_dist.asc()).offset(offset).limit(limit)
            result = await db.execute(stmt)
            rows = result.all()

            memories_with_sim: List[Tuple[Memory, Optional[float]]] = []
            for mem, dist in rows:
                sim = round(max(0.0, 1.0 - float(dist)), 4) if dist is not None else None
                memories_with_sim.append((mem, sim))

            return memories_with_sim, total_count, category_counts

        else:
            # Standard listing sorted by recency
            stmt = select(Memory).where(Memory.user_id == user_id)
            if category and category.lower() != "all":
                stmt = stmt.where(Memory.category == category.lower())

            # Count total
            count_stmt = select(func.count(Memory.id)).where(Memory.user_id == user_id)
            if category and category.lower() != "all":
                count_stmt = count_stmt.where(Memory.category == category.lower())
            total_count = (await db.execute(count_stmt)).scalar() or 0

            stmt = stmt.order_by(Memory.created_at.desc()).offset(offset).limit(limit)
            result = await db.execute(stmt)
            memories = result.scalars().all()

            memories_with_sim = [(mem, None) for mem in memories]
            return memories_with_sim, total_count, category_counts

    async def get_memory_by_id(
        self,
        db: AsyncSession,
        user_id: UUID,
        memory_id: UUID,
    ) -> Optional[Memory]:
        """Fetches a specific memory ensuring strict user ownership."""
        stmt = (
            select(Memory)
            .where(Memory.id == memory_id)
            .where(Memory.user_id == user_id)
        )
        result = await db.execute(stmt)
        return result.scalars().first()

    async def update_memory(
        self,
        db: AsyncSession,
        user_id: UUID,
        memory_id: UUID,
        data: MemoryUpdate,
    ) -> Optional[Memory]:
        """
        Updates an existing memory. Regenerates vector embedding if content was modified.
        """
        memory = await self.get_memory_by_id(db, user_id, memory_id)
        if not memory:
            return None

        content_changed = False
        if data.content is not None and data.content.strip() != memory.content:
            memory.content = data.content.strip()
            content_changed = True

        if data.category is not None:
            cat = data.category.lower()
            if cat in MEMORY_CATEGORIES:
                memory.category = cat

        if data.importance is not None:
            memory.importance = data.importance

        if data.source is not None:
            memory.source = data.source

        if data.metadata is not None:
            memory.meta_data = data.metadata

        # Re-generate embedding if content changed
        if content_changed:
            memory.embedding = await embedding_service.generate_embedding(memory.content)

        memory.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(memory)
        return memory

    async def delete_memory(
        self,
        db: AsyncSession,
        user_id: UUID,
        memory_id: UUID,
    ) -> bool:
        """Deletes a memory ensuring user ownership."""
        memory = await self.get_memory_by_id(db, user_id, memory_id)
        if not memory:
            return False

        await db.delete(memory)
        await db.commit()
        return True

    async def search_similar_memories(
        self,
        db: AsyncSession,
        user_id: UUID,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.4,
        category: Optional[str] = None,
    ) -> List[Tuple[Memory, float]]:
        """
        Performs semantic similarity search over user memories using pgvector cosine distance.
        Returns relevant memories strictly bounded by similarity threshold and limit.
        """
        query_clean = (query or "").strip()
        if not query_clean:
            return []

        query_vec = await embedding_service.generate_embedding(query_clean)
        cosine_dist = Memory.embedding.cosine_distance(query_vec).label("distance")

        # In pgvector cosine distance: distance = 1 - cosine_similarity
        # Therefore: similarity >= min_similarity <=> distance <= (1 - min_similarity)
        max_dist = max(0.0, min(1.0, 1.0 - min_similarity))

        stmt = (
            select(Memory, cosine_dist)
            .where(Memory.user_id == user_id)
            .where(Memory.embedding.isnot(None))
            .where(cosine_dist <= max_dist)
        )

        if category and category.lower() != "all" and category.lower() in MEMORY_CATEGORIES:
            stmt = stmt.where(Memory.category == category.lower())

        stmt = stmt.order_by(cosine_dist.asc()).limit(limit)

        result = await db.execute(stmt)
        rows = result.all()

        results: List[Tuple[Memory, float]] = []
        for mem, dist in rows:
            sim = round(max(0.0, 1.0 - float(dist)), 4)
            results.append((mem, sim))

        return results

    async def extract_and_save_memories(
        self,
        db: AsyncSession,
        user_id: UUID,
        text: Optional[str] = None,
        conversation_id: Optional[UUID] = None,
        auto_save: bool = True,
    ) -> Tuple[List[ExtractedMemoryItem], List[Memory]]:
        """
        Analyzes conversation or text, determines durable personal facts, and stores them.
        Distinguishes temporary chatter from long-term memory.
        """
        content_to_analyze = (text or "").strip()

        # If conversation_id is provided, fetch recent turns from PostgreSQL
        if not content_to_analyze and conversation_id:
            msg_stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.created_at.asc())
                .limit(20)
            )
            msg_result = await db.execute(msg_stmt)
            messages = msg_result.scalars().all()
            lines = [f"{m.role.capitalize()}: {m.content}" for m in messages if m.content]
            content_to_analyze = "\n".join(lines)

        if not content_to_analyze:
            return [], []

        # 1. Run memory extraction via Gemini
        extracted = await memory_extractor.extract_memories(content_to_analyze)
        if not extracted:
            return [], []

        saved: List[Memory] = []
        if auto_save:
            for item in extracted:
                # Deduplication check: check if user already has an identical memory
                existing_stmt = (
                    select(Memory)
                    .where(Memory.user_id == user_id)
                    .where(func.lower(Memory.content) == item.content.lower().strip())
                )
                existing = (await db.execute(existing_stmt)).scalars().first()
                if existing:
                    # Update importance if newly extracted value is higher
                    if item.importance > existing.importance:
                        existing.importance = item.importance
                        existing.updated_at = datetime.now(timezone.utc)
                        await db.commit()
                    saved.append(existing)
                    continue

                # Generate embedding and store new memory
                embedding_vec = await embedding_service.generate_embedding(item.content)
                new_mem = Memory(
                    user_id=user_id,
                    category=item.category,
                    content=item.content,
                    importance=item.importance,
                    source="conversation" if conversation_id else "extraction",
                    meta_data={"confidence": item.confidence, "reasoning": item.reasoning},
                    embedding=embedding_vec,
                )
                db.add(new_mem)
                await db.commit()
                await db.refresh(new_mem)
                saved.append(new_mem)

        return extracted, saved


# Singleton instance
memory_service = MemoryService()
