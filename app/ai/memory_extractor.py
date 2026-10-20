"""
Memory Extraction Service using Gemini.
Distinguishes temporary conversation context from long-term personal memory.
Filters out transient queries (weather, time, math, greetings) and extracts durable user facts.
"""

import asyncio
import json
import logging
from typing import List, Optional
from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.memory import ExtractedMemoryItem, MemoryCategory

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert Personal Memory Extractor for a voice AI assistant.
Your goal is to identify and extract durable, long-term personal facts about the USER that should be remembered across sessions.

CRITICAL RULES:
1. DISTINGUISH temporary conversation context from long-term personal memory.
   - "What is the weather?" / "What time is it?" / "Calculate 12 * 4" -> TRANSIENT CHATTER -> NEVER save as memory.
   - "I prefer concise answers." -> DURABLE FACT -> Extract as memory (category: preferences).
   - "I am working on a voice assistant." -> DURABLE FACT -> Extract as memory (category: projects / work).
   - "My name is Ragul and I work as an AI engineer." -> DURABLE FACT -> Extract as memory (category: identity / work).
   - "I run 5k every morning at 6 AM." -> DURABLE FACT -> Extract as memory (category: habits / fitness / routine).
2. DO NOT save everything. Only extract facts that have long-term utility for personalizing future interactions.
3. If the input contains NO durable personal facts about the user, you MUST return an empty JSON array: [].
4. Supported categories MUST be one of:
   identity, preferences, personality, goals, habits, fitness, learning, work, projects, routine, important_events, other.
5. Content format:
   Write each memory as a concise, objective declarative statement about the user.
   Example: "User prefers concise, direct answers without unnecessary fluff."
6. Importance score:
   Assign a float between 0.1 and 1.0 indicating significance (0.9 for core identity/goals, 0.5 for general preferences).
7. Confidence score:
   Assign a float between 0.1 and 1.0 indicating how clearly the user stated this fact.

Return valid JSON conforming to this schema:
[
  {
    "category": "preferences",
    "content": "User prefers concise and brief answers.",
    "importance": 0.8,
    "confidence": 0.95,
    "reasoning": "Explicit statement of communication style preference."
  }
]
"""


class MemoryExtractor:
    """
    Extracts high-value long-term personal memories from conversations using Gemini.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> Optional[genai.Client]:
        if not self.api_key:
            return None
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _extract_sync(self, conversation_text: str) -> List[ExtractedMemoryItem]:
        client = self._get_client()
        if not client or not conversation_text.strip():
            return []

        user_prompt = f"Analyze the following conversation text and extract durable personal memories according to your rules:\n\n---\n{conversation_text}\n---"

        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )

            text_resp = (response.text or "").strip()
            if not text_resp:
                return []

            data = json.loads(text_resp)
            if not isinstance(data, list):
                return []

            valid_categories = {
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
            }

            extracted_items: List[ExtractedMemoryItem] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                cat = item.get("category", "other").lower()
                if cat not in valid_categories:
                    cat = "other"
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                importance = float(item.get("importance", 0.5))
                confidence = float(item.get("confidence", 0.8))
                importance = max(0.0, min(1.0, importance))
                confidence = max(0.0, min(1.0, confidence))

                # Filter out low-confidence extractions
                if confidence < 0.4:
                    continue

                extracted_items.append(
                    ExtractedMemoryItem(
                        category=cat,  # type: ignore
                        content=content,
                        importance=importance,
                        confidence=confidence,
                        reasoning=item.get("reasoning"),
                    )
                )

            return extracted_items
        except Exception as exc:
            logger.warning(f"Memory extraction failed with error: {exc}")
            return []

    async def extract_memories(self, conversation_text: str) -> List[ExtractedMemoryItem]:
        """
        Asynchronously runs memory extraction against Gemini.
        """
        if not conversation_text or not conversation_text.strip():
            return []
        return await asyncio.to_thread(self._extract_sync, conversation_text)


memory_extractor = MemoryExtractor()
