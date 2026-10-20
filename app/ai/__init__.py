"""
AI and LLM Brain Layer.
Provides LLM provider abstractions, Gemini provider, context orchestration, and persistence.
"""

from app.ai.exceptions import (
    AIException,
    LLMAPIError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMTimeoutError,
)
from app.ai.llm import GeminiProvider, LLMManager, LLMProvider, llm_manager
from app.ai.prompt import BASE_PERSONALITY_INSTRUCTION, build_system_instruction
from app.ai.schemas import ChatMessage, LLMResponse, ToolCall
from app.ai.service import AIService, ai_service

__all__ = [
    "AIException",
    "LLMAPIError",
    "LLMConfigurationError",
    "LLMInvalidResponseError",
    "LLMTimeoutError",
    "GeminiProvider",
    "LLMManager",
    "LLMProvider",
    "llm_manager",
    "BASE_PERSONALITY_INSTRUCTION",
    "build_system_instruction",
    "ChatMessage",
    "LLMResponse",
    "ToolCall",
    "AIService",
    "ai_service",
]
