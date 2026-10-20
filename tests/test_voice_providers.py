"""
Unit tests for Voice Providers and VoiceManager.
Tests initialization, provider resolution, payload generation, and error conditions.
"""

import pytest
from app.voice import voice_manager, STTResponse, TTSResponse, ProviderConfigurationError
from app.voice.stt.sarvam import SarvamSTTProvider
from app.voice.tts.sarvam import SarvamTTSProvider


def test_voice_manager_resolves_sarvam():
    stt = voice_manager.get_stt_provider("sarvam")
    assert isinstance(stt, SarvamSTTProvider)
    assert stt.provider_name == "sarvam"

    tts = voice_manager.get_tts_provider("sarvam")
    assert isinstance(tts, SarvamTTSProvider)
    assert tts.provider_name == "sarvam"


def test_sarvam_stt_wav_packaging():
    provider = SarvamSTTProvider(api_key="test_dummy_key")
    # Raw PCM dummy bytes (silence)
    raw_pcm = b"\x00\x00" * 8000  # 16000 bytes = 0.5s of 16kHz 16-bit mono
    wav_bytes = provider._ensure_wav(raw_pcm)

    # Validate RIFF header
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes[:16]
    assert len(wav_bytes) == len(raw_pcm) + 44


@pytest.mark.asyncio
async def test_sarvam_stt_missing_key_raises_error():
    provider = SarvamSTTProvider(api_key="")
    with pytest.raises(ProviderConfigurationError):
        await provider.transcribe(b"dummy_audio")


@pytest.mark.asyncio
async def test_sarvam_tts_missing_key_raises_error():
    provider = SarvamTTSProvider(api_key="")
    with pytest.raises(ProviderConfigurationError):
        await provider.synthesize(text="Hello world")
