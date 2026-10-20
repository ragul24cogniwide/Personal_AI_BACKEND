"""
Context Builder for Personal AI Voice Assistant.
Combines:
- System instructions & Assistant personality
- User profile & preferences
- pgvector semantic memory retrieval
- Conversation history
- Current user request

Enforces Strict Priority Hierarchy:
1. Current user request (ALWAYS overrides old memories)
2. Safety
3. Explicit current preferences
4. Relevant memories
5. User profile
6. Conversation history
7. Assistant personality
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.embedding import embedding_service
from app.ai.prompt import BASE_PERSONALITY_INSTRUCTION
from app.ai.schemas import ChatMessage
from app.models.memory import Memory, MEMORY_CATEGORIES
from app.services.memory import memory_service

logger = logging.getLogger(__name__)

# Minimum cosine similarity threshold for pgvector semantic retrieval.
# Queries below this threshold are considered unrelated and filtered out.
DEFAULT_MIN_SIMILARITY = 0.52
MAX_RETRIEVED_MEMORIES = 5

# Keywords indicating generic / non-personal queries where personal memory retrieval is unnecessary
GENERIC_PATTERNS = [
    r"^\s*what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+",  # Math expressions
    r"^\s*calculate\b",
    r"^\s*what\s+time\s+is\s+it",
    r"^\s*what('s|\s+is)\s+the\s+weather",
    r"^\s*what\s+is\s+the\s+capital\s+of\b",
    r"^\s*tell\s+me\s+a\s+joke\b",
    r"^\s*(hi|hello|hey|good\s+morning|good\s+evening|good\s+night)\s*[\!\.\?]*$",
    r"^\s*who\s+won\s+the\b",
    r"^\s*translate\b",
    r"^\s*define\s+[a-zA-Z]+\b",
]

# Category keyword associations for targeted retrieval
CATEGORY_TRIGGERS: Dict[str, Set[str]] = {
    "fitness": {
        "workout", "exercise", "gym", "run", "running", "jog", "jogging",
        "cardio", "weights", "lift", "lifting", "training", "fitness",
        "stretching", "stretch", "pushups", "squats", "reps", "sets", "marathon", "diet", "calories",
        "tired", "lazy", "stamina"
    },
    "work": {
        "work", "job", "career", "office", "colleague", "boss", "manager",
        "company", "client", "meeting", "task", "deadline", "standup", "focus", "plan"
    },
    "projects": {
        "project", "coding", "code", "app", "develop", "software", "repo",
        "architecture", "stack", "build", "building", "saas", "frontend", "backend", "focus"
    },
    "preferences": {
        "prefer", "preference", "like", "dislike", "favorite", "favourite",
        "style", "format", "concise", "brief", "detailed", "verbose", "tone"
    },
    "goals": {
        "goal", "goals", "aim", "target", "milestone", "aspire", "ambition",
        "objective", "plan", "planning", "future", "focus", "priority", "priorities", "today"
    },
    "habits": {
        "habit", "habits", "morning", "evening", "daily", "routine",
        "coffee", "tea", "drink", "wake up", "sleep", "bedtime", "focus"
    },
    "learning": {
        "learn", "learning", "study", "studying", "book", "read", "reading",
        "course", "topic", "tutorial"
    },
    "identity": {
        "who am i", "my name", "about me", "my bio", "my age", "my birthday",
        "where do i live", "my hometown", "family"
    },
    "routine": {
        "routine", "schedule", "calendar", "agenda", "morning routine",
        "evening routine", "bedtime", "focus", "today", "plan"
    },
    "important_events": {
        "birthday", "anniversary", "event", "trip", "vacation", "flight",
        "holiday", "celebration"
    },
}


class ContextBuilder:
    """
    Constructs the end-to-end prompt and message context for Gemini.
    Combines personality, user profile, pgvector memories, and conversation history
    with strict precedence ordering.
    """

    @staticmethod
    def is_generic_query(user_request: str) -> bool:
        """
        Determines if user request is purely generic/transient (e.g. math, weather, generic trivia)
        where personal memories should NOT be searched or injected.
        """
        req_lower = user_request.strip().lower()
        for pattern in GENERIC_PATTERNS:
            if re.search(pattern, req_lower):
                return True
        return False

    @staticmethod
    def detect_candidate_categories(user_request: str) -> List[str]:
        """
        Detects specific memory categories referenced in the user request.
        """
        words = set(re.findall(r"\b[a-zA-Z]+\b", user_request.lower()))
        matched_categories: List[str] = []

        for category, triggers in CATEGORY_TRIGGERS.items():
            if words.intersection(triggers):
                matched_categories.append(category)

        return matched_categories

    async def retrieve_relevant_memories(
        self,
        db: AsyncSession,
        user_id: Optional[UUID],
        user_request: str,
        limit: int = MAX_RETRIEVED_MEMORIES,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> List[Memory]:
        """
        Determines whether personal memory is relevant to the request.
        If relevant, queries pgvector using cosine distance and returns
        only the most relevant memories. Unrelated memories are NEVER retrieved.
        """
        if not user_id or not user_request or not user_request.strip():
            return []

        # 1. Filter out purely generic queries (weather, time, math, generic jokes)
        if self.is_generic_query(user_request):
            logger.info("[ContextBuilder] Query classified as generic. Skipping personal memory retrieval.")
            return []

        # 2. Check candidate category hints
        candidate_categories = self.detect_candidate_categories(user_request)

        # 3. Perform pgvector similarity search
        try:
            # Query pgvector with cosine similarity
            results: List[Tuple[Memory, float]] = await memory_service.search_similar_memories(
                db=db,
                user_id=user_id,
                query=user_request,
                limit=limit * 2,  # Fetch slightly wider pool to filter and re-rank
                min_similarity=min_similarity,
            )

            if not results and candidate_categories:
                # If semantic vector didn't cross threshold but explicit category keywords were found,
                # retry with a slightly more permissive threshold for those specific categories
                for cat in candidate_categories:
                    cat_results = await memory_service.search_similar_memories(
                        db=db,
                        user_id=user_id,
                        query=user_request,
                        limit=limit,
                        min_similarity=max(0.44, min_similarity - 0.08),
                        category=cat,
                    )
                    results.extend(cat_results)

            if not results:
                logger.info("[ContextBuilder] No memories met relevance threshold for query.")
                return []

            # 4. Sort and select top matching memories
            # If candidate categories exist, prioritize matching categories
            def sort_key(item: Tuple[Memory, float]) -> float:
                mem, sim = item
                score = sim
                if candidate_categories and mem.category in candidate_categories:
                    score += 0.08  # Category match boost
                # Slight weight for high importance
                score += (mem.importance or 0.5) * 0.04
                return score

            results.sort(key=sort_key, reverse=True)

            selected_memories: List[Memory] = []
            seen_ids = set()
            for mem, sim in results:
                if mem.id not in seen_ids and len(selected_memories) < limit:
                    seen_ids.add(mem.id)
                    selected_memories.append(mem)

            logger.info(
                f"[ContextBuilder] Retrieved {len(selected_memories)} relevant memories for query: '{user_request[:40]}...'"
            )
            return selected_memories

        except Exception as exc:
            logger.warning(f"[ContextBuilder] Error retrieving memories from pgvector: {exc}")
            return []

    def build_system_instruction(
        self,
        user_profile: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        relevant_memories: Optional[List[Memory]] = None,
        user_goals: Optional[List[Dict[str, Any]]] = None,
        user_habits: Optional[List[Dict[str, Any]]] = None,
        assistant_personality: Optional[str] = None,
    ) -> str:
        """
        Compiles the system prompt integrating:
        - Assistant Personality
        - Strict Decision & Precedence Hierarchy
        - Unknown Information & Anti-Hallucination rules
        - External Tool & Calendar Integrity Rules
        - Active User Preferences (including proactivity level)
        - Active User Goals & Habits
        - Relevant Personal Memories
        - User Profile Context
        - Personalized Suggestions & Proactivity Guidelines
        """
        sections: List[str] = []

        # 1. Base Personality & Role
        personality = assistant_personality or BASE_PERSONALITY_INSTRUCTION
        sections.append(personality.strip())

        # 2. Strict Priority Order & Override Rules
        priority_rules = """### Strict Priority & Decision Hierarchy:
When responding, adhere strictly to this precedence hierarchy in descending order of authority:
1. CURRENT USER REQUEST: The user's immediate instruction, statement, or correction ALWAYS takes absolute precedence. If the user states a new instruction or contradicts an older memory (e.g., "Actually I don't run anymore, I swim now"), the CURRENT user request IMMEDIATELY OVERRIDES older memories and past profile entries.
2. SAFETY: Never violate safety, legal, or ethical guidelines.
3. EXPLICIT CURRENT PREFERENCES: Active communication style rules (conciseness, tone, language, proactivity).
4. RELEVANT MEMORIES: Factual, durable personal memories retrieved for this request.
5. USER PROFILE: General background details (occupation, bio, hobbies).
6. CONVERSATION HISTORY: Preceding conversation turns in this active session.
7. ASSISTANT PERSONALITY: Core tone and voice pacing.

Current user instructions MUST ALWAYS override old memories."""
        sections.append(priority_rules)

        # 3. Handling Unknown Information & Anti-Hallucination
        unknown_rules = """### Handling Unknown Information & Anti-Hallucination:
- If the user asks about personal information, past events, family, specific habits, tasks, or preferences that are NOT explicitly provided in the "Relevant Personal Memories", "Active User Goals", "Stated Habits", or "User Profile Context" below, you MUST clearly and honestly state that you do not have that information recorded.
- NEVER invent, assume, or hallucinate personal memories, habits, schedule events, or facts about the user.
- Acknowledge when you don't know, and invite the user to share if they'd like you to remember it."""
        sections.append(unknown_rules)

        # 4. External Tool & Calendar Integrity Rules
        tool_integrity_rules = """### External Tool & Calendar Integrity Rules:
- You must NEVER claim that you checked or accessed the user's calendar unless the tool was executed and returned 'status': 'success'.
- If the calendar tool succeeds with scheduled events: Naturally and conversationally speak the events and times for the user.
- If the calendar tool succeeds with an empty list of events (events_count: 0): Clearly confirm that you checked their calendar and they have no events scheduled for today (their calendar is clear).
- If a tool fails or returns 'status': 'error': The calendar was NOT accessed. You must NEVER say your calendar is clear or invent any events. Honestly and clearly explain the specific failure to the user:
  * permission_denied: State that permission to access the calendar was denied.
  * authentication_failure: State that the calendar account is not connected or authenticated.
  * mcp_unavailable: State that the calendar service is currently unavailable.
  * timeout: State that the request to the calendar timed out.
  * api_failure: State that there was a technical error reaching the calendar service.
- Never invent, guess, or hallucinate calendar meetings or schedule details."""
        sections.append(tool_integrity_rules)

        # 5. User Preferences & Proactivity Level
        proactivity = "medium"
        if user_preferences:
            proactivity = (user_preferences.get("proactivity") or "medium").lower().strip()
            pref_details: List[str] = []
            if user_preferences.get("assistant_tone"):
                pref_details.append(f"- Assistant Tone: {user_preferences['assistant_tone']}")
            if user_preferences.get("verbosity"):
                pref_details.append(f"- Desired Verbosity: {user_preferences['verbosity']}")
            pref_details.append(f"- Configured Proactivity Level: {proactivity}")
            if user_preferences.get("language"):
                pref_details.append(f"- Language: {user_preferences['language']}")
            if user_preferences.get("timezone"):
                pref_details.append(f"- Timezone: {user_preferences['timezone']}")

            if pref_details:
                sections.append("### Active User Preferences:\n" + "\n".join(pref_details))

        # 6. Active Goals
        if user_goals and len(user_goals) > 0:
            goal_lines: List[str] = []
            for g in user_goals:
                cat = g.get("category", "general").upper()
                prio = g.get("priority", "medium")
                desc = f" - {g['description']}" if g.get("description") else ""
                goal_lines.append(f"- [{cat}] ({prio} priority) {g.get('title')}{desc}")
            sections.append(
                "### Active User Goals:\n"
                "The user is actively pursuing the following personal and professional goals. "
                "Synthesize them when offering plans or suggestions:\n"
                + "\n".join(goal_lines)
            )

        # 7. Stated Daily & Weekly Habits
        if user_habits and len(user_habits) > 0:
            habit_lines: List[str] = []
            for h in user_habits:
                cat = h.get("category", "general").upper()
                freq = h.get("frequency", "daily")
                desc = f" - {h['description']}" if h.get("description") else ""
                habit_lines.append(f"- [{cat}] ({freq}) {h.get('name')}{desc}")
            sections.append(
                "### Stated Daily & Weekly Habits:\n"
                "The user maintains the following recurring habits. "
                "Consider these when discussing routines, schedules, or focus areas:\n"
                + "\n".join(habit_lines)
            )

        # 8. Relevant Personal Memories
        if relevant_memories and len(relevant_memories) > 0:
            memory_lines: List[str] = []
            for mem in relevant_memories:
                cat_tag = mem.category.upper() if mem.category else "GENERAL"
                imp_text = f"Importance: {round(mem.importance, 2)}" if mem.importance else ""
                meta_extra = f" ({imp_text})" if imp_text else ""
                memory_lines.append(f"- [{cat_tag}] {mem.content}{meta_extra}")

            sections.append(
                "### Relevant Personal Memories (Retrieved for this request):\n"
                "The following verified facts are relevant to the user's current request. "
                "Use them to personalize your response, unless the user's current utterance indicates an update:\n"
                + "\n".join(memory_lines)
            )

        # 9. User Profile Context
        if user_profile:
            profile_details: List[str] = []
            name = user_profile.get("preferred_name") or user_profile.get("name")
            if name:
                profile_details.append(f"- User's Name: {name}")
            if user_profile.get("occupation"):
                profile_details.append(f"- Occupation: {user_profile['occupation']}")
            if user_profile.get("bio"):
                profile_details.append(f"- Background: {user_profile['bio']}")
            if user_profile.get("communication_style"):
                profile_details.append(f"- Preferred Communication Style: {user_profile['communication_style']}")

            fitness_interests = user_profile.get("fitness_interests")
            if fitness_interests:
                if isinstance(fitness_interests, list):
                    profile_details.append(f"- Fitness Interests: {', '.join(str(f) for f in fitness_interests)}")
                elif isinstance(fitness_interests, str):
                    profile_details.append(f"- Fitness Interests: {fitness_interests}")

            interests = user_profile.get("interests")
            if interests:
                if isinstance(interests, list):
                    profile_details.append(f"- Interests: {', '.join(str(i) for i in interests)}")
                elif isinstance(interests, str):
                    profile_details.append(f"- Interests: {interests}")

            hobbies = user_profile.get("hobbies")
            if hobbies:
                if isinstance(hobbies, list):
                    profile_details.append(f"- Hobbies: {', '.join(str(h) for h in hobbies)}")
                elif isinstance(hobbies, str):
                    profile_details.append(f"- Hobbies: {hobbies}")

            daily_routine = user_profile.get("daily_routine")
            if daily_routine and isinstance(daily_routine, dict):
                profile_details.append(f"- Daily Routine Details: {daily_routine}")

            if profile_details:
                sections.append("### User Profile Context:\n" + "\n".join(profile_details))

        # 10. Personalized Suggestions & Proactivity Guidelines
        if proactivity == "low":
            proactivity_instructions = """* ACTIVE MODE: LOW PROACTIVITY (Reactive Mode)
- Answer the user's inquiry directly, accurately, and concisely.
- DO NOT provide unsolicited recommendations, tips, extra follow-ups, or unprompted next steps.
- Provide suggestions or plans ONLY if the user explicitly asks for advice, ideas, recommendations, or a daily plan (e.g., "What should I focus on today?", "What do you suggest?", "Can you recommend something?").
- Keep responses clean, focused, and free of pushy advice."""
        elif proactivity == "high":
            proactivity_instructions = """* ACTIVE MODE: HIGH PROACTIVITY (Proactive Partner Mode)
- Answer the user's question directly first.
- Actively synthesize connections between their immediate statement and their active goals, habits, upcoming schedule, and projects.
- Offer actionable, tailored recommendations, potential next steps, or complementary habits (e.g., reminding them of a habit that pairs well with their plan, or suggesting a focused work block).
- Always remain supportive, encouraging, and advisory—NEVER nagging, pushy, or overbearing."""
        else:  # default 'medium'
            proactivity_instructions = """* ACTIVE MODE: MEDIUM PROACTIVITY (Balanced Mode - Default)
- Answer the user's question directly, clearly, and completely first.
- When naturally relevant to the user's stated goals, habits, or situation, offer one gentle, brief suggestion or thoughtful question (maximum 1 sentence).
- Keep suggestions lightweight, natural, and unobtrusive—do NOT force suggestions into every turn."""

        suggestions_guidelines = f"""### Personalized Suggestions & Proactivity Guidelines:
You are an intelligent, empathetic companion designed to provide personalized, context-aware suggestions.
Adhere strictly to these core rules:

1. SUGGESTIONS ARE RECOMMENDATIONS, NOT COMMANDS:
- Frame all suggestions as gentle options, collaborative ideas, or polite recommendations (e.g., "If you'd like...", "One option you might consider...", "How about...", "Perhaps...", "You could try...").
- NEVER use demanding, bossy, or commanding language (avoid "You must", "You need to", "Go do this now").
- Always respect the user's autonomy, feelings, and energy levels with empathy.

2. DEEP CONTEXT GROUNDING (NO GENERIC ADVICE):
- When the user mentions an activity, dilemma, or task (e.g., fitness hesitation, coding, daily planning), leverage their specific context (profile, memories, goals, habits, schedule) rather than generic textbook advice.
- Example (Fitness context): If the user says "I don't feel like going to the gym", DO NOT give generic lectures on why exercise is healthy. Instead:
  * Acknowledge and validate how they feel with empathy.
  * Draw on their known fitness context (goals, preferred light workouts, habits, workout memories).
  * Offer a gentle alternative (e.g., a quick 15-minute mobility stretch, bodyweight movement, or light walk) that keeps their streak alive without causing burnout.
- Example (Day planning): If the user asks "What should I focus on today?", synthesize their active goals, ongoing projects, calendar events (from MCP calendar results if available), and daily habits into a cohesive, realistic focus suggestion.

3. STRICT GROUNDING & NO HALLUCINATION OF UNAVAILABLE INFORMATION:
- NEVER pretend to know unavailable information or invent fictional details.
- If calendar events, specific tasks, deadlines, or habit details are NOT present in your context or tool results, do NOT invent imaginary meetings, tasks, or appointments.
- State honestly what you know, and invite the user to clarify or share more details if needed.

4. PROACTIVITY BEHAVIOR:
{proactivity_instructions}

5. DO NOT BE ANNOYING OR OVERLY PROACTIVE:
- Under all proactivity settings, never be annoying, nagging, repetitive, or preachy.
- Keep spoken responses conversational, natural, and concise."""

        sections.append(suggestions_guidelines)

        return "\n\n".join(sections)

    async def build_context(
        self,
        db: AsyncSession,
        user_id: Optional[UUID],
        user_request: str,
        user_profile: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        user_goals: Optional[List[Dict[str, Any]]] = None,
        user_habits: Optional[List[Dict[str, Any]]] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        assistant_personality: Optional[str] = None,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> Tuple[str, List[ChatMessage], List[Memory]]:
        """
        Full Context Builder execution:
        1. Determines relevance and retrieves relevant memories from pgvector.
        2. Compiles system instruction adhering to priority order, goals, habits, and proactivity.
        3. Prepares turn messages (history + current user request).
        Returns:
            - system_instruction: str
            - turn_messages: List[ChatMessage]
            - relevant_memories: List[Memory]
        """
        # 1. Retrieve relevant memories via pgvector
        relevant_memories = await self.retrieve_relevant_memories(
            db=db,
            user_id=user_id,
            user_request=user_request,
            min_similarity=min_similarity,
        )

        # 2. Build prioritized system instruction with goals, habits, and proactivity
        system_instruction = self.build_system_instruction(
            user_profile=user_profile,
            user_preferences=user_preferences,
            relevant_memories=relevant_memories,
            user_goals=user_goals,
            user_habits=user_habits,
            assistant_personality=assistant_personality,
        )

        # 3. Format turn messages: conversation history + current user request
        history = list(conversation_history or [])
        turn_messages = history + [ChatMessage(role="user", content=user_request.strip())]

        return system_instruction, turn_messages, relevant_memories


# Singleton instance
context_builder = ContextBuilder()
