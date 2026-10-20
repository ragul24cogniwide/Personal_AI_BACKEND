"""
Personal Memories API Endpoints.
Implements:
- GET /memories (Listing, category filter, semantic search)
- POST /memories (Manual & structured memory creation with pgvector embeddings)
- PATCH /memories/:id (Memory updates with auto re-embedding)
- DELETE /memories/:id (Memory deletion)
- POST /memories/extract (Extraction distinguishing temporary chatter from long-term memory)
- GET /memories/categories (List supported categories)
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database.session import get_db
from app.models.memory import MEMORY_CATEGORIES, Memory
from app.models.user import User
from app.schemas.memory import (
    DeleteMemoryResponse,
    MemoryCreate,
    MemoryExtractRequest,
    MemoryExtractResponse,
    MemoryListResponse,
    MemoryResponse,
    MemorySearchRequest,
    MemoryUpdate,
)
from app.services.memory import memory_service

router = APIRouter()


def _format_memory_response(mem: Memory, similarity: Optional[float] = None) -> MemoryResponse:
    """Helper to convert ORM Memory model to MemoryResponse schema."""
    return MemoryResponse(
        id=mem.id,
        user_id=mem.user_id,
        category=mem.category,
        content=mem.content,
        importance=mem.importance,
        source=mem.source,
        metadata=mem.meta_data or {},
        has_embedding=mem.embedding is not None,
        similarity=similarity,
        created_at=mem.created_at,
        updated_at=mem.updated_at,
    )


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List personal memories",
    description="Retrieve long-term personal memories with optional category filtering and semantic vector search.",
)
async def list_memories(
    category: Optional[str] = Query(None, description="Filter by category (or 'all')"),
    query: Optional[str] = Query(None, description="Semantic search query using pgvector embeddings"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    memories_with_sim, total, category_counts = await memory_service.get_memories(
        db=db,
        user_id=current_user.id,
        category=category,
        query=query,
        limit=limit,
        offset=offset,
    )

    items = [_format_memory_response(m, sim) for m, sim in memories_with_sim]
    return MemoryListResponse(
        items=items,
        total=total,
        category_counts=category_counts,
    )


@router.get(
    "/categories",
    response_model=List[str],
    summary="Get supported memory categories",
)
async def get_categories(
    current_user: User = Depends(get_current_user),
):
    return list(MEMORY_CATEGORIES)


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a personal memory",
    description="Creates a new memory record and stores its 1536-dim vector embedding in pgvector.",
)
async def create_memory(
    data: MemoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    memory = await memory_service.create_memory(
        db=db,
        user_id=current_user.id,
        data=data,
    )
    return _format_memory_response(memory)


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Get memory by ID",
)
async def get_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    memory = await memory_service.get_memory_by_id(
        db=db,
        user_id=current_user.id,
        memory_id=memory_id,
    )
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    return _format_memory_response(memory)


@router.patch(
    "/{memory_id}",
    response_model=MemoryResponse,
    summary="Update a personal memory",
    description="Updates content or metadata. Re-generates vector embedding if content is modified.",
)
async def update_memory(
    memory_id: UUID,
    data: MemoryUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    updated = await memory_service.update_memory(
        db=db,
        user_id=current_user.id,
        memory_id=memory_id,
        data=data,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    return _format_memory_response(updated)


@router.delete(
    "/{memory_id}",
    response_model=DeleteMemoryResponse,
    summary="Delete a personal memory",
)
async def delete_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    deleted = await memory_service.delete_memory(
        db=db,
        user_id=current_user.id,
        memory_id=memory_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory not found",
        )
    return DeleteMemoryResponse(
        deleted=True,
        id=memory_id,
        message="Memory successfully deleted",
    )


@router.post(
    "/search",
    response_model=List[MemoryResponse],
    summary="Semantic vector search over personal memories",
    description="Finds relevant memories using pgvector cosine distance. Never exposes unrelated memories.",
)
async def search_memories(
    payload: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    results = await memory_service.search_similar_memories(
        db=db,
        user_id=current_user.id,
        query=payload.query,
        limit=payload.limit,
        min_similarity=payload.min_similarity,
        category=payload.category,
    )
    return [_format_memory_response(m, sim) for m, sim in results]


@router.post(
    "/extract",
    response_model=MemoryExtractResponse,
    summary="Extract durable personal memories from conversation or text",
    description="Distinguishes temporary conversation context (e.g. weather, time) from durable personal facts.",
)
async def extract_memories(
    payload: MemoryExtractRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    extracted, saved = await memory_service.extract_and_save_memories(
        db=db,
        user_id=current_user.id,
        text=payload.text,
        conversation_id=payload.conversation_id,
        auto_save=payload.auto_save,
    )
    return MemoryExtractResponse(
        extracted=extracted,
        saved_memories=[_format_memory_response(m) for m in saved],
    )
