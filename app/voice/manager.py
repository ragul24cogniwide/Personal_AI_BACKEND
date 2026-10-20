"""
Voice Manager and Provider Factory.
Dynamically resolves and injects configured STT and TTS providers.
"""

from typing import Dict, Type
from app.core.config import settings
from app.voice.exceptions import ProviderConfigurationError
from app.voice.stt.base import STTProvider
from app.voice.stt.sarvam import SarvamSTTProvider
from app.voice.stt.deepgram import DeepgramSTTProvider
from app.voice.stt.openai import OpenAISTTProvider
from app.voice.tts.base import TTSProvider
from app.voice.tts.sarvam import SarvamTTSProvider
from app.voice.tts.elevenlabs import ElevenLabsTTSProvider
from app.voice.tts.openai import OpenAITTSProvider


class VoiceManager:
    """Orchestrates voice providers and manages active instances."""

    def __init__(self):
        self._stt_registry: Dict[str, Type[STTProvider]] = {
            "sarvam": SarvamSTTProvider,
            "deepgram": DeepgramSTTProvider,
            "openai": OpenAISTTProvider,
        }
        self._tts_registry: Dict[str, Type[TTSProvider]] = {
            "sarvam": SarvamTTSProvider,
            "elevenlabs": ElevenLabsTTSProvider,
            "openai": OpenAITTSProvider,
        }
        self._stt_instances: Dict[str, STTProvider] = {}
        self._tts_instances: Dict[str, TTSProvider] = {}

    def get_stt_provider(self, name: str = None) -> STTProvider:
        """Resolves the requested or globally configured STT provider."""
        provider_name = (name or settings.STT_PROVIDER).lower()
        if provider_name not in self._stt_registry:
            raise ProviderConfigurationError(
                f"Unsupported STT provider '{provider_name}'. Supported: {list(self._stt_registry.keys())}"
            )

        if provider_name not in self._stt_instances:
            provider_cls = self._stt_registry[provider_name]
            self._stt_instances[provider_name] = provider_cls()

        return self._stt_instances[provider_name]

    def get_tts_provider(self, name: str = None) -> TTSProvider:
        """Resolves the requested or globally configured TTS provider."""
        provider_name = (name or settings.TTS_PROVIDER).lower()
        if provider_name not in self._tts_registry:
            raise ProviderConfigurationError(
                f"Unsupported TTS provider '{provider_name}'. Supported: {list(self._tts_registry.keys())}"
            )

        if provider_name not in self._tts_instances:
            provider_cls = self._tts_registry[provider_name]
            self._tts_instances[provider_name] = provider_cls()

        return self._tts_instances[provider_name]


# Global VoiceManager singleton
voice_manager = VoiceManager()
