"""
Sarvam AI Text-to-Speech (TTS) Provider Implementation.
Converts text into high-fidelity natural speech using Sarvam Bulbul models.
Supports single-shot synthesis, sentence-level streaming, long-response chunking,
and configurable voice parameters.
"""

import asyncio
import base64
import io
import re
import wave
import httpx
from typing import AsyncGenerator, List, Optional
from app.core.config import settings
from app.voice.tts.base import TTSProvider
from app.voice.schemas import TTSResponse, VoiceSettings
from app.voice.exceptions import TTSException, ProviderConfigurationError


def clean_text_for_speech(text: str) -> str:
    """
    Cleans markdown formatting, code blocks, and special characters
    from LLM text before passing to speech synthesis.
    """
    if not text:
        return ""
    # 1. Remove markdown code blocks ```...```
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    # 2. Remove inline code `...`
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # 3. Remove markdown headers ###
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    # 4. Remove bold/italics **text** or *text* or __text__
    cleaned = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", cleaned)
    # 5. Remove markdown links [title](url) -> title
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # 6. Remove bullet markers (e.g., "* ", "- ", "1. ")
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    # 7. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_text_into_chunks(text: str, max_chars: int = 500) -> List[str]:
    """
    Splits text into naturally paced sentence/clause chunks under max_chars.
    Prevents splitting words or abruptly cutting sentences.
    """
    cleaned = clean_text_for_speech(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    # Split by sentence terminators (. ! ? \n)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        # If single sentence exceeds max_chars, split on punctuation like commas or semicolons
        if len(s) > max_chars:
            clauses = re.split(r"(?<=[,;:—])\s+", s)
            for clause in clauses:
                c = clause.strip()
                if not c:
                    continue
                if current_len + len(c) + 1 > max_chars and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [c]
                    current_len = len(c)
                else:
                    current_chunk.append(c)
                    current_len += len(c) + 1
        else:
            if current_len + len(s) + 1 > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [s]
                current_len = len(s)
            else:
                current_chunk.append(s)
                current_len += len(s) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def combine_wav_bytes(wav_bytes_list: List[bytes]) -> bytes:
    """Concatenates multiple PCM WAV audio byte arrays into a single valid WAV byte array."""
    if not wav_bytes_list:
        return b""
    if len(wav_bytes_list) == 1:
        return wav_bytes_list[0]

    first = wav_bytes_list[0]
    with wave.open(io.BytesIO(first), "rb") as w:
        params = w.getparams()
        frames = [w.readframes(w.getnframes())]

    for other in wav_bytes_list[1:]:
        with wave.open(io.BytesIO(other), "rb") as w:
            frames.append(w.readframes(w.getnframes()))

    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as out_w:
        out_w.setparams(params)
        for f in frames:
            out_w.writeframes(f)

    return out_io.getvalue()


class SarvamTTSProvider(TTSProvider):
    """Concrete TTS Provider for Sarvam AI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_speaker: Optional[str] = None,
        default_language: Optional[str] = None,
        default_pace: Optional[float] = None,
        default_pitch: Optional[float] = None,
        default_loudness: Optional[float] = None,
        default_sample_rate: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.SARVAM_API_KEY
        self.base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self.default_model = default_model or settings.SARVAM_TTS_MODEL or "bulbul:v3"
        self.default_speaker = default_speaker or settings.SARVAM_TTS_SPEAKER or "shubh"
        self.default_language = default_language or settings.SARVAM_LANGUAGE_CODE or "en-IN"
        self.default_pace = default_pace if default_pace is not None else settings.SARVAM_TTS_PACE
        self.default_pitch = default_pitch if default_pitch is not None else settings.SARVAM_TTS_PITCH
        self.default_loudness = default_loudness if default_loudness is not None else settings.SARVAM_TTS_LOUDNESS
        self.default_sample_rate = default_sample_rate or settings.SARVAM_TTS_SAMPLE_RATE
        self._client: Optional[httpx.AsyncClient] = None

    async def get_client(self) -> httpx.AsyncClient:
        """Returns or initializes the persistent HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=25.0, write=15.0, pool=10.0),
                limits=httpx.Limits(max_keepalive_connections=30, max_connections=60, keepalive_expiry=60.0),
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

    def _resolve_voice_params(
        self,
        speaker: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        pace: Optional[float] = None,
        pitch: Optional[float] = None,
        loudness: Optional[float] = None,
        voice_settings: Optional[VoiceSettings] = None,
        **kwargs,
    ):
        """Resolves configuration hierarchy: explicit args -> voice_settings -> class defaults."""
        v_speaker = speaker or (voice_settings.voice if voice_settings else None) or self.default_speaker
        v_lang = language_code or (voice_settings.language if voice_settings else None) or self.default_language
        v_model = model or (voice_settings.model if voice_settings else None) or self.default_model
        v_pace = pace if pace is not None else ((voice_settings.speech_rate if voice_settings else None) or self.default_pace)
        v_pitch = pitch if pitch is not None else ((voice_settings.pitch if voice_settings else None) or self.default_pitch)
        v_loudness = loudness if loudness is not None else ((voice_settings.loudness if voice_settings else None) or self.default_loudness)
        v_sample_rate = (voice_settings.sample_rate if voice_settings else None) or self.default_sample_rate

        return v_speaker, v_lang, v_model, v_pace, v_pitch, v_loudness, v_sample_rate

    async def _call_sarvam_api(
        self,
        text_chunk: str,
        speaker: str,
        language_code: str,
        model: str,
        pace: float,
        pitch: float,
        loudness: float,
        sample_rate: int,
        timeout: float = 30.0,
    ) -> bytes:
        """Executes a single POST call to Sarvam AI text-to-speech endpoint with connection pooling."""
        if not self.api_key:
            raise ProviderConfigurationError(
                "SARVAM_API_KEY is not configured in backend environment or .env file.",
                provider="sarvam",
            )

        endpoint = f"{self.base_url}/text-to-speech"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text_chunk,
            "language_code": language_code,
            "speaker": speaker,
            "model": model,
            "pace": pace,
            "speech_sample_rate": sample_rate,
        }

        try:
            client = await self.get_client()
            response = await client.post(endpoint, headers=headers, json=payload)

            # If model requires alternate payload structure (legacy v1 fallback)
            if response.status_code in (400, 422) and "inputs" not in payload:
                alt_payload = {
                    "inputs": [text_chunk],
                    "target_language_code": language_code,
                    "speaker": speaker,
                    "model": model,
                    "pace": pace,
                    "pitch": pitch,
                    "loudness": loudness,
                    "speech_sample_rate": sample_rate,
                    "enable_preprocessing": True,
                }
                response = await client.post(endpoint, headers=headers, json=alt_payload)

            if response.status_code != 200:
                err_msg = f"Sarvam TTS failed with HTTP {response.status_code}: {response.text}"
                raise TTSException(err_msg, provider="sarvam", status_code=response.status_code)

            data = response.json()
            audios = data.get("audios", [])
            if not audios:
                raise TTSException("Sarvam TTS returned empty audio list.", provider="sarvam")

            return base64.b64decode(audios[0])

        except httpx.TimeoutException as exc:
            raise TTSException(
                f"Sarvam TTS request timed out after {timeout}s: {str(exc)}",
                provider="sarvam",
            )
        except httpx.RequestError as exc:
            raise TTSException(
                f"Network error connecting to Sarvam TTS: {str(exc)}",
                provider="sarvam",
            )

    async def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        pace: Optional[float] = None,
        pitch: Optional[float] = None,
        loudness: Optional[float] = None,
        voice_settings: Optional[VoiceSettings] = None,
        **kwargs,
    ) -> TTSResponse:
        """
        Synthesizes text into high quality voice audio.
        Automatically chunks long responses (>2000 chars) to prevent Sarvam 2500 char limit
        and combines audio frames into a single seamless WAV file.
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            raise TTSException("Text payload is empty for speech synthesis.", provider="sarvam")

        v_speaker, v_lang, v_model, v_pace, v_pitch, v_loudness, v_sample_rate = (
            self._resolve_voice_params(
                speaker=speaker,
                language_code=language_code,
                model=model,
                pace=pace,
                pitch=pitch,
                loudness=loudness,
                voice_settings=voice_settings,
                **kwargs,
            )
        )

        # For texts within 2000 chars, call Sarvam directly in one request
        if len(cleaned_text) <= 2000:
            wav_bytes = await self._call_sarvam_api(
                text_chunk=cleaned_text,
                speaker=v_speaker,
                language_code=v_lang,
                model=v_model,
                pace=v_pace,
                pitch=v_pitch,
                loudness=v_loudness,
                sample_rate=v_sample_rate,
            )
        else:
            # Long response: split into chunks and combine
            chunks = split_text_into_chunks(cleaned_text, max_chars=1500)
            chunk_wavs: List[bytes] = []
            for chunk in chunks:
                chunk_bytes = await self._call_sarvam_api(
                    text_chunk=chunk,
                    speaker=v_speaker,
                    language_code=v_lang,
                    model=v_model,
                    pace=v_pace,
                    pitch=v_pitch,
                    loudness=v_loudness,
                    sample_rate=v_sample_rate,
                )
                chunk_wavs.append(chunk_bytes)
            wav_bytes = combine_wav_bytes(chunk_wavs)

        return TTSResponse(
            audio_base64=base64.b64encode(wav_bytes).decode("utf-8"),
            sample_rate=v_sample_rate,
            format="wav",
            provider="sarvam",
            speaker=v_speaker,
            language_code=v_lang,
        )

    async def stream(
        self,
        text: str,
        speaker: Optional[str] = None,
        language_code: Optional[str] = None,
        model: Optional[str] = None,
        pace: Optional[float] = None,
        pitch: Optional[float] = None,
        loudness: Optional[float] = None,
        chunk_size: Optional[int] = None,
        voice_settings: Optional[VoiceSettings] = None,
        **kwargs,
    ) -> AsyncGenerator[TTSResponse, None]:
        """
        Streams audio responses in sequential chunks for real-time playback or long texts.
        Yields TTSResponse for each sentence/segment as soon as synthesized.
        """
        cleaned_text = clean_text_for_speech(text)
        if not cleaned_text:
            raise TTSException("Text payload is empty for speech synthesis.", provider="sarvam")

        v_speaker, v_lang, v_model, v_pace, v_pitch, v_loudness, v_sample_rate = (
            self._resolve_voice_params(
                speaker=speaker,
                language_code=language_code,
                model=model,
                pace=pace,
                pitch=pitch,
                loudness=loudness,
                voice_settings=voice_settings,
                **kwargs,
            )
        )

        max_chunk = chunk_size or settings.SARVAM_TTS_MAX_CHUNK_CHARS or 500
        chunks = split_text_into_chunks(cleaned_text, max_chars=max_chunk)

        for chunk in chunks:
            chunk_bytes = await self._call_sarvam_api(
                text_chunk=chunk,
                speaker=v_speaker,
                language_code=v_lang,
                model=v_model,
                pace=v_pace,
                pitch=v_pitch,
                loudness=v_loudness,
                sample_rate=v_sample_rate,
            )
            yield TTSResponse(
                audio_base64=base64.b64encode(chunk_bytes).decode("utf-8"),
                sample_rate=v_sample_rate,
                format="wav",
                provider="sarvam",
                speaker=v_speaker,
                language_code=v_lang,
            )
