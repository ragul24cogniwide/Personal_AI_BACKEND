"""
Conversation Endpoints.
Full PostgreSQL persistence and retrieval for User Conversations and Conversation Messages.
Endpoints:
- GET /conversations: List all conversations for the authenticated user.
- GET /conversations/{id}: Retrieve a conversation with chronological messages.
- POST /conversations: Create a new conversation for the authenticated user.
- DELETE /conversations/{id}: Delete a conversation and its messages.
- POST /conversations/{id}/messages: Append a message to a conversation.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user, get_db
from app.ai import ai_service
from app.ai.prompt import strip_hidden_reasoning
from app.models.conversation import Conversation, ConversationMessage
from app.models.user import User
from app.schemas.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationMessageCreate,
    ConversationMessageResponse,
    ConversationSummaryResponse,
    ConversationUpdate,
    DeleteConversationResponse,
)

router = APIRouter()


@router.get(
    "",
    response_model=List[ConversationSummaryResponse],
    summary="List all conversations for authenticated user",
)
@router.get(
    "/",
    response_model=List[ConversationSummaryResponse],
    include_in_schema=False,
)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100, description="Max conversations to return"),
    offset: int = Query(0, ge=0, description="Offset pagination"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve conversations belonging to the authenticated user.
    Ordered by most recently active (updated_at DESC).
    Includes message count and preview snippet of the last message.
    """
    # 1. Fetch conversations for user with pagination
    query = (
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(query)
    conversations = result.scalars().all()

    if not conversations:
        return []

    # 2. Compute message counts and last message previews for returned conversations
    conv_ids = [c.id for c in conversations]

    # Message counts per conversation
    counts_stmt = (
        select(ConversationMessage.conversation_id, func.count(ConversationMessage.id))
        .where(ConversationMessage.conversation_id.in_(conv_ids))
        .group_by(ConversationMessage.conversation_id)
    )
    counts_result = await db.execute(counts_stmt)
    count_map = dict(counts_result.fetchall())

    # Latest message for each conversation
    summaries: List[ConversationSummaryResponse] = []
    for conv in conversations:
        # Fetch the single most recent message for preview
        last_msg_stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conv.id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(1)
        )
        last_msg_res = await db.execute(last_msg_stmt)
        last_msg = last_msg_res.scalar_one_or_none()

        preview = None
        last_msg_at = None
        if last_msg:
            # Strip reasoning and format clean preview
            clean_content = strip_hidden_reasoning(last_msg.content)
            preview = (clean_content[:97] + "...") if len(clean_content) > 100 else clean_content
            last_msg_at = last_msg.created_at

        summaries.append(
            ConversationSummaryResponse(
                id=conv.id,
                user_id=conv.user_id,
                title=conv.title,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=count_map.get(conv.id, 0),
                last_message_preview=preview,
                last_message_at=last_msg_at,
            )
        )

    return summaries


@router.post(
    "",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation for authenticated user",
)
@router.post(
    "/",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_conversation(
    payload: Optional[ConversationCreate] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Creates a new conversation belonging to the authenticated user.
    """
    title = (payload.title if payload and payload.title else "New Conversation").strip()
    conversation = Conversation(
        user_id=current_user.id,
        title=title or "New Conversation",
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    return ConversationDetailResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=0,
        messages=[],
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Retrieve a conversation and its messages",
)
async def get_conversation(
    conversation_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
    offset: int = Query(0, ge=0, description="Offset pagination for messages"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve conversation metadata and chronological messages.
    Enforces user ownership (404 if conversation does not exist or belong to user).
    Do NOT return hidden reasoning.
    """
    # 1. Fetch conversation with user ownership check
    conv_stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    )
    conv_result = await db.execute(conv_stmt)
    conv = conv_result.scalar_one_or_none()

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # 2. Get total message count
    count_stmt = select(func.count(ConversationMessage.id)).where(
        ConversationMessage.conversation_id == conversation_id
    )
    count_result = await db.execute(count_stmt)
    total_messages = count_result.scalar() or 0

    # 3. Fetch chronological messages with pagination
    # If offset is 0 and total > limit, return the most recent `limit` messages in chronological order
    if offset == 0 and total_messages > limit:
        # Fetch latest N descending, then reverse for chronological order
        msgs_stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )
        msgs_res = await db.execute(msgs_stmt)
        messages = list(reversed(msgs_res.scalars().all()))
    else:
        msgs_stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        msgs_res = await db.execute(msgs_stmt)
        messages = msgs_res.scalars().all()

    # Build clean message response objects (ensuring hidden reasoning is stripped)
    msg_responses: List[ConversationMessageResponse] = []
    for msg in messages:
        clean_text = strip_hidden_reasoning(msg.content or "")
        msg_responses.append(
            ConversationMessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                user_id=msg.user_id,
                role=msg.role,
                content=clean_text,
                message_type=msg.message_type,
                metadata=msg.meta_data or {},
                created_at=msg.created_at,
            )
        )

    return ConversationDetailResponse(
        id=conv.id,
        user_id=conv.user_id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=total_messages,
        messages=msg_responses,
    )


@router.delete(
    "/{conversation_id}",
    response_model=DeleteConversationResponse,
    summary="Delete a conversation and all its messages",
)
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Deletes the conversation and cascades deletion to all associated messages in PostgreSQL.
    Enforces user ownership.
    """
    conv_stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    )
    conv_result = await db.execute(conv_stmt)
    conv = conv_result.scalar_one_or_none()

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    await db.delete(conv)
    await db.commit()

    return DeleteConversationResponse(
        success=True,
        message="Conversation deleted successfully",
        id=conversation_id,
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Append a message to an existing conversation",
)
async def add_message_to_conversation(
    conversation_id: UUID,
    payload: ConversationMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Appends a new message directly into a conversation belonging to the authenticated user.
    Supported roles: user, assistant, tool, system.
    Do NOT store hidden reasoning.
    """
    conv_stmt = select(Conversation).where(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id,
    )
    conv_result = await db.execute(conv_stmt)
    conv = conv_result.scalar_one_or_none()

    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    saved_msg = await ai_service.save_message(
        db=db,
        conversation_id=conv.id,
        user_id=current_user.id,
        role=payload.role,
        content=payload.content,
        message_type=payload.message_type,
        metadata=payload.metadata,
    )

    return ConversationMessageResponse(
        id=saved_msg.id,
        conversation_id=saved_msg.conversation_id,
        user_id=saved_msg.user_id,
        role=saved_msg.role,
        content=saved_msg.content,
        message_type=saved_msg.message_type,
        metadata=saved_msg.meta_data or {},
        created_at=saved_msg.created_at,
    )
