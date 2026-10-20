"""
OpenAI Whisper Speech-to-Text Provider (Plug-and-play architecture stub).
"""

from typing import Optional
from app.voice.stt.base import STTProvider
from app.voice.schemas import STTResponse
from app.voice.exceptions import STTException


class OpenAISTTProvider(STTProvider):
    """OpenAI Whisper STT Provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "openai"

    async def transcribe(
        self,
        audio_data: bytes,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> STTResponse:
        raise STTException("OpenAI STT provider is available as a plug-and-play slot. Configure OPENAI_API_KEY to activate.", provider="openai")
