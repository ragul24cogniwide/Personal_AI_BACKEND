"""
OpenAI Text-to-Speech Provider (Plug-and-play architecture stub).
"""

from typing import Optional
from app.voice.tts.base import TTSProvider
from app.voice.schemas import TTSResponse
from app.voice.exceptions import TTSException


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS Provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    async def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        pace: float = 1.0,
        pitch: float = 0.0,
        loudness: float = 1.0,
        **kwargs
    ) -> TTSResponse:
        raise TTSException("OpenAI TTS provider is available as a plug-and-play slot. Configure OPENAI_API_KEY to activate.", provider="openai")

    async def stream(
        self,
        text: str,
        speaker: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        pace: float = 1.0,
        pitch: float = 0.0,
        loudness: float = 1.0,
        chunk_size: Optional[int] = None,
        **kwargs
    ):
        raise TTSException("OpenAI TTS streaming is available as a plug-and-play slot. Configure OPENAI_API_KEY to activate.", provider="openai")
        yield  # type: ignore

