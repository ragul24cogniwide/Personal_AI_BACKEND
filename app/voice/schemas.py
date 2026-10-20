"""
Schemas and Data Transfer Objects for the Voice System.
Provides normalized types for Speech-to-Text and Text-to-Speech responses.
"""

from typing import Optional
from pydantic import BaseModel, Field


class STTResponse(BaseModel):
    """Normalized response from any Speech-to-Text provider."""
    transcript: str = Field(..., description="Transcribed text from audio")
    language_code: Optional[str] = Field(None, description="Detected or requested language code")
    confidence: Optional[float] = Field(None, description="Confidence score if available (0.0 - 1.0)")
    provider: str = Field(..., description="Provider name that produced this transcription (e.g. 'sarvam')")
    duration_seconds: Optional[float] = Field(None, description="Audio duration in seconds if known")


class TTSResponse(BaseModel):
    """Normalized response from any Text-to-Speech provider."""
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    format: str = Field("wav", description="Audio container format (wav, mp3, etc.)")
    provider: str = Field(..., description="Provider name that produced this synthesis (e.g. 'sarvam')")
    speaker: Optional[str] = Field(None, description="Speaker/Voice identifier used")
    language_code: Optional[str] = Field(None, description="Language code used for synthesis")

    @property
    def audio_bytes(self) -> bytes:
        """Helper to get raw audio bytes from base64 string."""
        import base64
        return base64.b64decode(self.audio_base64)


class VoiceStateMessage(BaseModel):
    """Message sent over WebSocket to synchronize mobile VoiceOrb state."""
    type: str = "state_change"
    state: str = Field(..., description="idle | listening | transcribing | thinking | speaking | error")
    message: Optional[str] = Field(None, description="Human-readable status description")


class VoiceSettings(BaseModel):
    """Configuration structure for Voice and Speech synthesis settings."""
    voice: Optional[str] = Field(None, description="Voice/Speaker identity (e.g. 'shubh', 'aditya', 'ritu', 'priya')")
    language: Optional[str] = Field(None, description="Target language code (e.g. 'en-IN', 'hi-IN')")
    speech_rate: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Pace/speech rate multiplier")
    model: Optional[str] = Field(None, description="TTS model identifier (e.g. 'bulbul:v3')")
    pitch: Optional[float] = Field(0.0, description="Voice pitch adjustment")
    loudness: Optional[float] = Field(1.0, ge=0.1, le=3.0, description="Output loudness gain")
    sample_rate: int = Field(16000, description="Audio sample rate (Hz)")

