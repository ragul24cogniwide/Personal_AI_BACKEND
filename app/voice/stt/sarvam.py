"""
Sarvam AI Speech-to-Text (STT) Provider Implementation.
Uses Sarvam AI REST API (model: saaras:v2 / saaras:v1 / saaras:v4).
"""

import io
import wave
import httpx
from typing import Optional
from app.core.config import settings
from app.voice.stt.base import STTProvider
from app.voice.schemas import STTResponse
from app.voice.exceptions import STTException, ProviderConfigurationError


class SarvamSTTProvider(STTProvider):
    """Concrete STT Provider for Sarvam AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_language: Optional[str] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.SARVAM_API_KEY
        self.base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.SARVAM_STT_MODEL or "saaras:v3"
        self.default_language = default_language or settings.SARVAM_LANGUAGE_CODE or "en-IN"

        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Returns or initializes the persistent HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=25.0, write=15.0, pool=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=40, keepalive_expiry=60.0),
                http2=False,
            )
        return self._client

    async def aclose(self):
        """Closes the persistent HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def provider_name(self) -> str:
        return "sarvam"

    def _ensure_wav(self, audio_data: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2) -> bytes:
        """
        Validates if audio_data is already a RIFF WAV container.
        If it is raw PCM bytes, encodes it into a standard 16-bit PCM WAV.
        """
        if audio_data.startswith(b"RIFF") and len(audio_data) >= 44:
            return audio_data

        # Wrap raw PCM in RIFF WAV
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)
        return buffer.getvalue()

    async def transcribe(
        self,
        audio_data: bytes,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> STTResponse:
        """
        Sends audio to Sarvam AI STT endpoint and returns normalized STTResponse.
        Uses connection pooling and keep-alive to minimize STT handshake latency.
        """
        if not self.api_key:
            raise ProviderConfigurationError(
                "SARVAM_API_KEY is not configured in backend environment or .env file.",
                provider="sarvam"
            )

        if not audio_data or len(audio_data) < 100:
            raise STTException("Audio payload is empty or too short to transcribe.", provider="sarvam")

        wav_bytes = self._ensure_wav(audio_data)
        target_model = model or self.default_model
        target_lang = language_code or self.default_language

        endpoint = f"{self.base_url}/speech-to-text"
        headers = {
            "api-subscription-key": self.api_key,
        }

        # Sarvam STT multipart form parameters
        form_data = {
            "model": target_model,
        }
        if target_lang and target_lang != "auto":
            form_data["language_code"] = target_lang

        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav"),
        }

        try:
            client = await self.get_client()
            response = await client.post(
                endpoint,
                headers=headers,
                data=form_data,
                files=files,
            )

            if response.status_code != 200:
                error_detail = response.text
                raise STTException(
                    f"Sarvam STT failed with HTTP {response.status_code}: {error_detail}",
                    provider="sarvam",
                    status_code=response.status_code
                )

            result = response.json()
            transcript = result.get("transcript", "").strip()
            detected_lang = result.get("language_code", target_lang)

            return STTResponse(
                transcript=transcript,
                language_code=detected_lang,
                provider="sarvam",
            )

        except httpx.RequestError as exc:
            raise STTException(
                f"Network error connecting to Sarvam STT: {str(exc)}",
                provider="sarvam"
            )
