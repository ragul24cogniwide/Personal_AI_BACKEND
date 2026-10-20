"""
Abstract Base Class for Text-to-Speech (TTS) Providers.
Enables plug-and-play synthesis engines (Sarvam, ElevenLabs, OpenAI).
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from app.voice.schemas import TTSResponse


class TTSProvider(ABC):
    """Abstract base class for all TTS providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the unique identifier for the provider (e.g. 'sarvam')."""
        pass

    @abstractmethod
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
        """
        Synthesizes text into high quality voice audio.

        :param text: Text string to convert to speech
        :param speaker: Voice/Speaker identity (e.g. 'shubh', 'aditya', 'ritu', 'priya')
        :param language_code: Target language code (e.g. 'en-IN', 'hi-IN')
        :param model: Provider model identifier (e.g. 'bulbul:v3')
        :param pace: Speech speed/pace multiplier (0.5 to 2.0)
        :param pitch: Voice pitch adjustment
        :param loudness: Output volume gain
        :return: Normalized TTSResponse containing base64 audio and metadata
        """
        pass

    @abstractmethod
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
    ) -> AsyncGenerator[TTSResponse, None]:
        """
        Streams audio responses in sequential chunks for real-time playback or long texts.

        :param text: Text string to convert to speech
        :param speaker: Voice/Speaker identity
        :param language_code: Target language code
        :param model: Provider model identifier
        :param pace: Speech speed/pace multiplier
        :param pitch: Voice pitch adjustment
        :param loudness: Output volume gain
        :param chunk_size: Max characters per synthesis segment
        :return: Async generator yielding TTSResponse instances for each segment
        """
        pass

