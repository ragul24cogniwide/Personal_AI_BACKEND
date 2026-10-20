"""
ElevenLabs Text-to-Speech Provider (Plug-and-play architecture stub).
"""

from typing import Optional
from app.voice.tts.base import TTSProvider
from app.voice.schemas import TTSResponse
from app.voice.exceptions import TTSException


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs TTS Provider implementation."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

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
        raise TTSException("ElevenLabs TTS provider is available as a plug-and-play slot. Configure ELEVENLABS_API_KEY to activate.", provider="elevenlabs")

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
        raise TTSException("ElevenLabs TTS streaming is available as a plug-and-play slot. Configure ELEVENLABS_API_KEY to activate.", provider="elevenlabs")
        yield  # type: ignore


