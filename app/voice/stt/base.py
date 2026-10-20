"""
Abstract Base Class for Speech-to-Text (STT) Providers.
Allows interchangeable providers (Sarvam, Deepgram, OpenAI) without changing voice pipeline.
"""

from abc import ABC, abstractmethod
from typing import Optional
from app.voice.schemas import STTResponse


class STTProvider(ABC):
    """Abstract base class that all STT providers must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the unique identifier for the provider (e.g. 'sarvam')."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> STTResponse:
        """
        Transcribes audio bytes into text.

        :param audio_data: Raw audio data (WAV container or PCM bytes)
        :param language_code: Optional ISO/BCP-47 language code (e.g. 'en-IN', 'hi-IN')
        :param model: Optional provider-specific model override
        :param kwargs: Additional provider-specific parameters
        :return: Normalized STTResponse
        """
        pass
