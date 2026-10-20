"""
High-Level AI Orchestration Service.
Coordinates context construction (system instructions, user profile, conversation history),
database persistence in PostgreSQL, and streaming LLM generation via Gemini.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional
from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.ai.context_builder import context_builder
from app.ai.llm.manager import llm_manager
from app.ai.prompt import build_system_instruction, strip_hidden_reasoning
from app.ai.schemas import ChatMessage
from app.database.session import AsyncSessionLocal
from app.mcp.models import ExecutionStatus
from app.models.conversation import Conversation, ConversationMessage
from app.models.user import User, UserProfile, UserPreference
from app.services.memory import memory_service

logger = logging.getLogger(__name__)

# Sensible context limits to ensure Gemini latency, accuracy, and token efficiency
DEFAULT_CONTEXT_MESSAGE_LIMIT = 15
MAX_CONTEXT_CHARACTERS = 8000


class AIService:
    """Orchestrates AI interactions, context injection, and conversation persistence."""

    async def get_or_create_conversation(
        self,
        db: AsyncSession,
        user_id: UUID,
        conversation_id: Optional[UUID] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """Finds or creates an active conversation for the given user."""
        if conversation_id:
            conv = await db.get(Conversation, conversation_id)
            if conv and conv.user_id == user_id:
                return conv

        # Find the most recently active conversation for this user
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        active_conv = result.scalar_one_or_none()

        if active_conv:
            return active_conv

        # Create new conversation
        new_conv = Conversation(
            user_id=user_id,
            title=title or "Voice Assistant Session",
        )
        db.add(new_conv)
        await db.commit()
        await db.refresh(new_conv)
        return new_conv

    async def load_user_context(
        self, db: AsyncSession, user_id: UUID
    ) -> tuple[
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Loads user profile, preferences, active goals, and habits for prompt injection."""
        stmt = (
            select(User)
            .options(
                selectinload(User.profile),
                selectinload(User.preferences),
                selectinload(User.goals),
                selectinload(User.habits),
            )
            .where(User.id == user_id)
        )
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            return None, None, [], []

        profile_dict = None
        if user.profile:
            p: UserProfile = user.profile
            profile_dict = {
                "name": p.name,
                "preferred_name": p.preferred_name,
                "occupation": p.occupation,
                "bio": p.bio,
                "communication_style": p.communication_style,
                "interests": p.interests,
                "hobbies": p.hobbies,
                "fitness_interests": p.fitness_interests,
                "learning_interests": p.learning_interests,
                "work_interests": p.work_interests,
                "daily_routine": p.daily_routine,
            }

        pref_dict = None
        if user.preferences:
            pref: UserPreference = user.preferences
            pref_dict = {
                "assistant_tone": pref.assistant_tone,
                "verbosity": pref.verbosity,
                "proactivity": pref.proactivity or "medium",
                "language": pref.language,
                "timezone": pref.timezone,
            }

        goals_list = []
        if user.goals:
            for g in user.goals:
                if g.status != "completed":
                    goals_list.append({
                        "category": g.category,
                        "title": g.title,
                        "description": g.description,
                        "priority": g.priority,
                        "status": g.status,
                    })

        habits_list = []
        if user.habits:
            for h in user.habits:
                habits_list.append({
                    "category": h.category,
                    "name": h.name,
                    "frequency": h.frequency,
                    "description": h.description,
                })

        return profile_dict, pref_dict, goals_list, habits_list

    async def load_recent_messages(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        limit: int = DEFAULT_CONTEXT_MESSAGE_LIMIT,
        max_chars: int = MAX_CONTEXT_CHARACTERS,
    ) -> List[ChatMessage]:
        """
        Loads the most recent conversation messages chronologically for LLM context.
        Applies sensible context limits:
        1. Caps to `limit` most recent messages (default 15).
        2. Bounds cumulative character count to `max_chars` (default 8000).
        3. Normalizes role sequence so context flows naturally into Gemini.
        """
        stmt = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == conversation_id)
            .order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        recent_messages = list(reversed(result.scalars().all()))

        # Trim oldest messages if character budget is exceeded
        total_chars = sum(len(m.content or "") for m in recent_messages)
        while recent_messages and total_chars > max_chars:
            popped = recent_messages.pop(0)
            total_chars -= len(popped.content or "")

        # If window begins on an assistant turn, trim it so history starts cleanly
        if recent_messages and recent_messages[0].role == "assistant":
            recent_messages.pop(0)

        chat_messages: List[ChatMessage] = []
        for msg in recent_messages:
            clean_text = strip_hidden_reasoning(msg.content or "")
            if clean_text:
                chat_messages.append(
                    ChatMessage(
                        role=msg.role,
                        content=clean_text,
                    )
                )
        return chat_messages

    async def save_message(
        self,
        db: AsyncSession,
        conversation_id: UUID,
        role: str,
        content: str,
        user_id: Optional[UUID] = None,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """
        Stores a conversation message in PostgreSQL.
        Guarantees:
        - conversation_id, user_id, role, content, message_type, metadata, created_at.
        - Supported roles: user, assistant, tool, system.
        - NEVER stores hidden reasoning (reasoning tags and thought metadata stripped).
        - Automatically updates conversation updated_at and first-turn title.
        """
        resolved_user_id = user_id
        conv = await db.get(Conversation, conversation_id)
        if conv and not resolved_user_id:
            resolved_user_id = conv.user_id

        if not resolved_user_id:
            raise ValueError(f"Cannot save message: user_id could not be resolved for conversation {conversation_id}")

        clean_content = strip_hidden_reasoning(content.strip())

        # Clean metadata: strip any hidden thought/reasoning keys
        clean_metadata = dict(metadata or {})
        for key in list(clean_metadata.keys()):
            if "thought" in key.lower() or "reasoning" in key.lower() or "hidden" in key.lower():
                del clean_metadata[key]

        msg = ConversationMessage(
            conversation_id=conversation_id,
            user_id=resolved_user_id,
            role=role,
            content=clean_content,
            message_type=message_type,
            meta_data=clean_metadata,
        )
        db.add(msg)

        # Update conversation timestamp & auto-title on first user turn
        if conv:
            conv.updated_at = func.now()
            if (
                role == "user"
                and conv.title in ("New Conversation", "Voice Assistant Session")
                and clean_content
            ):
                # Concise title from first user query (up to 50 chars)
                first_line = clean_content.split("\n")[0][:50].strip()
                if first_line:
                    conv.title = first_line

        await db.commit()
        await db.refresh(msg)
        return msg

    async def process_turn_stream(
        self,
        db: AsyncSession,
        user_id: Optional[UUID],
        user_text: str,
        conversation_id: Optional[UUID] = None,
        custom_system_instruction: Optional[str] = None,
        on_tool_executed: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ) -> AsyncIterator[str]:
        """
        Processes a full user turn with streaming:
        1. Resolves/creates conversation & context.
        2. Persists the user message to PostgreSQL with user_id.
        3. Streams partial response chunks from Gemini (stripping hidden reasoning).
        4. Persists the final assistant message to PostgreSQL upon stream completion.
        """
        # 1. User & Conversation resolution
        active_user_id = user_id
        if not active_user_id:
            # Query first available user in DB for fallback session
            user_stmt = select(User.id).limit(1)
            u_res = await db.execute(user_stmt)
            active_user_id = u_res.scalar_one_or_none()

        conv = None
        if active_user_id:
            conv = await self.get_or_create_conversation(
                db=db,
                user_id=active_user_id,
                conversation_id=conversation_id,
            )

        # 2. Build AI context & load history
        profile_dict, pref_dict, goals_list, habits_list = None, None, [], []
        if active_user_id:
            profile_dict, pref_dict, goals_list, habits_list = await self.load_user_context(db, active_user_id)

        # Load history with sensible context limits
        history: List[ChatMessage] = []
        if conv:
            history = await self.load_recent_messages(
                db=db,
                conversation_id=conv.id,
                limit=DEFAULT_CONTEXT_MESSAGE_LIMIT,
            )

        clean_user_text = strip_hidden_reasoning(user_text)

        # 3. ContextBuilder: Determine relevance, pgvector memory search, priority order
        system_instruction, turn_messages, relevant_memories = await context_builder.build_context(
            db=db,
            user_id=active_user_id,
            user_request=clean_user_text,
            user_profile=profile_dict,
            user_preferences=pref_dict,
            user_goals=goals_list,
            user_habits=habits_list,
            conversation_history=history,
            assistant_personality=custom_system_instruction,
        )

        # 4. Persist incoming user turn
        if conv and active_user_id:
            await self.save_message(
                db=db,
                conversation_id=conv.id,
                user_id=active_user_id,
                role="user",
                content=clean_user_text,
            )

        # 5. MCP Tool Router & Gemini Execution Flow
        llm = llm_manager.get_provider()
        full_response_chunks: List[str] = []

        try:
            from app.mcp.router import tool_router

            if not tool_router._servers:
                from app.mcp.servers import CalendarMCPServer, ContactsMCPServer
                tool_router.register_server(CalendarMCPServer())
                tool_router.register_server(ContactsMCPServer())
            if not tool_router._tools:
                await tool_router.discover_all_tools()

            mcp_tools = tool_router.get_gemini_tools()
            if mcp_tools:
                # 5a. Gemini tool decision step
                decision_res = await llm.generate_with_tools(
                    messages=turn_messages,
                    tools=mcp_tools,
                    system_instruction=system_instruction,
                )

                if decision_res.tool_calls:
                    # 5b. Backend executes tools concurrently via ToolRouter -> MCPClient -> MCPServer
                    async def _exec_single_tool(tc) -> Dict[str, Any]:
                        logger.info(
                            f"[AIService] Gemini requested MCP tool: '{tc.name}' "
                            f"with arguments: {tc.arguments}"
                        )
                        execution = await tool_router.route_tool_execution(
                            tool_name=tc.name,
                            arguments=tc.arguments,
                            user_id=active_user_id,
                            conversation_id=conv.id if conv else None,
                            db=db,
                        )
                        if execution.status == ExecutionStatus.COMPLETED:
                            res_payload = execution.result or {"status": "success"}
                            if isinstance(res_payload, dict) and "calendar_accessed" not in res_payload:
                                res_payload["calendar_accessed"] = True
                            if on_tool_executed:
                                try:
                                    await on_tool_executed(tc.name, res_payload)
                                except Exception as cb_err:
                                    logger.warning(f"[AIService] on_tool_executed error: {cb_err}")
                            return {
                                "name": tc.name,
                                "response": res_payload,
                            }
                        else:
                            return {
                                "name": tc.name,
                                "response": {
                                    "status": "error",
                                    "error_type": execution.error_type or "api_failure",
                                    "error": execution.error_message or "Tool execution denied or failed",
                                    "calendar_accessed": False,
                                },
                            }

                    tool_results: List[Dict[str, Any]] = list(
                        await asyncio.gather(*[_exec_single_tool(tc) for tc in decision_res.tool_calls])
                    )

                    # 5c. Stream final spoken synthesis from Gemini incorporating tool results
                    async for chunk in llm.stream_tool_response(
                        messages=turn_messages,
                        raw_model_content=decision_res.raw_candidate_content,
                        tool_results=tool_results,
                        tools=mcp_tools,
                        system_instruction=system_instruction,
                    ):
                        full_response_chunks.append(chunk)
                        yield chunk
                else:
                    # Direct response without tool calls
                    direct_content = decision_res.content
                    if direct_content:
                        clean_direct = strip_hidden_reasoning(direct_content)
                        full_response_chunks.append(clean_direct)
                        yield clean_direct
            else:
                # No MCP tools configured; standard stream
                async for chunk in llm.stream(
                    messages=turn_messages,
                    system_instruction=system_instruction,
                ):
                    full_response_chunks.append(chunk)
                    yield chunk
        finally:
            # 6. Persist assistant turn to PostgreSQL (never stores hidden reasoning)
            full_assistant_text = "".join(full_response_chunks).strip()
            clean_assistant_text = strip_hidden_reasoning(full_assistant_text)
            if conv and clean_assistant_text and active_user_id:
                await self.save_message(
                    db=db,
                    conversation_id=conv.id,
                    user_id=active_user_id,
                    role="assistant",
                    content=clean_assistant_text,
                )

                # 7. Post-turn background memory extraction
                asyncio.create_task(
                    self._safe_extract_memory(
                        user_id=active_user_id,
                        conversation_id=conv.id,
                        user_text=clean_user_text,
                    )
                )

    async def _safe_extract_memory(
        self,
        user_id: UUID,
        conversation_id: UUID,
        user_text: str,
    ) -> None:
        """
        Runs post-conversation memory extraction in background without blocking audio.
        Distinguishes temporary conversation from durable facts and saves to pgvector.
        """
        try:
            if context_builder.is_generic_query(user_text):
                return

            async with AsyncSessionLocal() as db:
                await memory_service.extract_and_save_memories(
                    db=db,
                    user_id=user_id,
                    text=user_text,
                    conversation_id=conversation_id,
                    auto_save=True,
                )
        except Exception as exc:
            logger.debug(f"[AIService] Background memory extraction error: {exc}")

    async def generate_response(
        self,
        db: AsyncSession,
        user_id: Optional[UUID],
        user_text: str,
        conversation_id: Optional[UUID] = None,
        custom_system_instruction: Optional[str] = None,
    ) -> str:
        """Non-streaming version of process_turn."""
        chunks = []
        async for chunk in self.process_turn_stream(
            db=db,
            user_id=user_id,
            user_text=user_text,
            conversation_id=conversation_id,
            custom_system_instruction=custom_system_instruction,
        ):
            chunks.append(chunk)
        return "".join(chunks)


# Global AIService singleton
ai_service = AIService()
