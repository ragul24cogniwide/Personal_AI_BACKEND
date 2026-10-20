"""
Schemas for LLM messages, responses, and streaming deltas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single message in a conversation turn."""
    role: str = Field(..., description="Role: 'user', 'assistant', 'system', or 'tool'")
    content: str = Field(..., description="The textual content of the message")
    name: Optional[str] = Field(None, description="Optional author identifier")


class ToolCall(BaseModel):
    """Represents a tool or function call requested by the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


class LLMResponse(BaseModel):
    """Complete response returned by an LLM provider."""
    content: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model identifier used for generation")
    provider: str = Field(..., description="Provider name (e.g. 'gemini')")
    finish_reason: Optional[str] = Field(None, description="Reason for generation stop")
    prompt_tokens: Optional[int] = Field(None, description="Prompt token count if available")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls if generated")
    raw_candidate_content: Optional[Any] = Field(None, description="Raw candidate content preserving thought signatures")

    model_config = {"arbitrary_types_allowed": True}
