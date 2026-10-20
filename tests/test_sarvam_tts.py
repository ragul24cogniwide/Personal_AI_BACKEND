"""
Comprehensive Test Suite for STEP 9: Sarvam Text-to-Speech (TTS).
Verifies:
1. Gemini text -> Sarvam TTS synthesis
2. Audio playback validity & WAV headers
3. Long response chunking & streaming (>2500 chars)
4. Audio interruption handling
5. TTS failure handling (invalid key, empty payload)
6. Network failure & timeout handling
7. Voice settings configuration
8. WebSocket full-turn with STT/Gemini/TTS audio delivery & playback_done
9. WebSocket mid-stream interruption
"""

import asyncio
import base64
import io
import wave
import pytest
import httpx
from unittest.mock import patch, AsyncMock
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.core.config import settings
from app.ai.llm.gemini import GeminiProvider
from app.voice import (
    voice_manager,
    SarvamTTSProvider,
    TTSResponse,
    VoiceSettings,
    TTSException,
    ProviderConfigurationError,
)
from app.voice.tts.sarvam import clean_text_for_speech, split_text_into_chunks, combine_wav_bytes


@pytest.mark.asyncio
async def test_01_gemini_text_to_sarvam_tts():
    """
    Test 1: Gemini text -> Sarvam TTS.
    Verifies that text produced by Gemini (or representative response)
    is synthesized by SarvamTTSProvider into valid WAV audio.
    """
    sample_text = "Hello! I am Aura, your personal AI assistant. How can I help you today?"
    
    tts_provider = voice_manager.get_tts_provider("sarvam")
    assert isinstance(tts_provider, SarvamTTSProvider)
    assert tts_provider.provider_name == "sarvam"

    response = await tts_provider.synthesize(text=sample_text)
    assert isinstance(response, TTSResponse)
    assert response.provider == "sarvam"
    assert response.format == "wav"
    assert response.sample_rate == 16000
    assert response.audio_base64 is not None
    assert len(response.audio_base64) > 1000

    audio_bytes = response.audio_bytes
    assert audio_bytes.startswith(b"RIFF"), "Audio does not have standard RIFF header"
    assert b"WAVE" in audio_bytes[:16], "Audio does not have WAVE format marker"


@pytest.mark.asyncio
async def test_02_audio_playback_structure():
    """
    Test 2: Audio playback validity.
    Opens the synthesized WAV audio with standard Python wave module to ensure
    it contains valid PCM frames, correct sample rate, channels, and playable duration.
    """
    tts_provider = voice_manager.get_tts_provider("sarvam")
    response = await tts_provider.synthesize(text="Testing audio playback structure for personal assistant.")

    audio_bytes = response.audio_bytes
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        samp_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration_sec = n_frames / frame_rate

        assert n_channels in (1, 2), f"Unexpected channel count: {n_channels}"
        assert samp_width == 2, f"Expected 16-bit audio (2 bytes), got {samp_width}"
        assert frame_rate in (16000, 22050, 24000), f"Unexpected frame rate: {frame_rate}"
        assert n_frames > 0, "Audio frames must be non-zero"
        assert duration_sec > 0.5, f"Expected audio duration > 0.5s, got {duration_sec:.2f}s"


@pytest.mark.asyncio
async def test_03_long_response_handling():
    """
    Test 3: Long response handling (>2500 characters).
    Sarvam's single request limit is 2500 characters.
    Verifies that SarvamTTSProvider splits text exceeding 2500 characters into chunks,
    synthesizes each chunk without hitting HTTP 400, combines the audio,
    and also streams chunks sequentially via stream().
    """
    paragraphs = [
        "Welcome to your personal productivity briefing. Today we will focus on three key priorities: your software engineering goals, your daily fitness routine, and your upcoming learning sessions. Making consistent progress each day compound into incredible outcomes over the course of a year.",
        "First, let us review your software architecture design. We have integrated FastAPI with modern async SQLAlchemy and PostgreSQL, connected to Google Gemini 2.5 Flash for high speed cognitive intelligence, and Sarvam AI for natural regional voice speech to text and text to speech.",
        "Second, maintaining a healthy physical routine is essential for sharp mental clarity. Spending thirty minutes exercising in the morning boosts focus and helps sustain high energy throughout demanding technical problem-solving sessions.",
        "Finally, dedicating thirty minutes in the evening to reading technical books or articles helps expand your horizon and uncover creative solutions to complex programming challenges. Feel free to ask me to summarize articles or suggest new topics.",
    ]
    long_text = " ".join(paragraphs) * 3  # ~3200 characters!
    assert len(long_text) > 2500, f"Expected text > 2500 chars, got {len(long_text)}"

    chunks = split_text_into_chunks(long_text, max_chars=1200)
    assert len(chunks) >= 3, "Expected multiple chunks"
    for c in chunks:
        assert len(c) <= 2000, f"Chunk too long: {len(c)}"

    tts_provider = voice_manager.get_tts_provider("sarvam")

    # 1. Test synthesize() automatic combination on long text
    combined_response = await tts_provider.synthesize(text=long_text)
    assert combined_response.audio_base64 is not None
    assert len(combined_response.audio_base64) > 10000

    with wave.open(io.BytesIO(combined_response.audio_bytes), "rb") as w:
        duration = w.getnframes() / w.getframerate()
        assert duration > 5.0, f"Expected long audio > 5.0s, got {duration:.2f}s"

    # 2. Test stream() yielding sequential chunks
    stream_chunks = []
    async for chunk in tts_provider.stream(text=long_text, chunk_size=800):
        stream_chunks.append(chunk)
        assert chunk.format == "wav"
        assert len(chunk.audio_base64) > 0
        if len(stream_chunks) >= 2:
            break  # Test first 2 chunks to keep test fast

    assert len(stream_chunks) >= 2, "Expected at least 2 stream chunks"


@pytest.mark.asyncio
async def test_04_audio_interruption():
    """
    Test 4: Audio interruption.
    Simulates user interrupting speech synthesis / playback by cancelling the task mid-turn.
    Verifies that the task halts immediately and raises asyncio.CancelledError.
    """
    tts_provider = voice_manager.get_tts_provider("sarvam")
    long_text = "This is a detailed and lengthy explanation that will be interrupted by the user mid-stream." * 5

    task = asyncio.create_task(tts_provider.synthesize(long_text))
    # Give it a tiny fraction of time to start running
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled() is True


@pytest.mark.asyncio
async def test_05_tts_failure_handling():
    """
    Test 5: TTS failure handling.
    Verifies that invalid API credentials or empty text payloads
    raise appropriate TTSException without crashing the system.
    """
    # 1. Invalid API key
    bad_provider = SarvamTTSProvider(api_key="sk_invalid_test_key_that_will_fail")
    with pytest.raises(TTSException) as exc_info:
        await bad_provider.synthesize(text="Hello world")
    assert "Sarvam TTS failed" in str(exc_info.value) or "401" in str(exc_info.value) or "403" in str(exc_info.value)

    # 2. Empty text payload
    good_provider = voice_manager.get_tts_provider("sarvam")
    with pytest.raises(TTSException) as exc_info_empty:
        await good_provider.synthesize(text="    ")
    assert "empty" in str(exc_info_empty.value).lower()


@pytest.mark.asyncio
async def test_06_network_failure_handling():
    """
    Test 6: Network failure & timeout handling.
    Simulates HTTP network error during TTS request.
    Verifies that TTSException with informative message is raised.
    """
    provider = SarvamTTSProvider()
    with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused to api.sarvam.ai")):
        with pytest.raises(TTSException) as exc_info:
            await provider.synthesize(text="Hello network test")
        assert "Network error connecting to Sarvam TTS" in str(exc_info.value)

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Request timed out")):
        with pytest.raises(TTSException) as exc_info:
            await provider.synthesize(text="Hello timeout test")
        assert "timed out" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_07_voice_settings_configuration():
    """
    Test 7: Voice settings configuration.
    Verifies that voice settings (speaker, language, speech_rate, model, pitch, loudness)
    are respected during synthesis.
    """
    provider = voice_manager.get_tts_provider("sarvam")
    custom_settings = VoiceSettings(
        voice="aditya",
        language="en-IN",
        speech_rate=1.1,
        model="bulbul:v3",
        pitch=0.0,
        loudness=1.0,
    )

    response = await provider.synthesize(
        text="Testing custom voice settings with speaker Aditya.",
        voice_settings=custom_settings,
    )
    assert response.speaker == "aditya"
    assert response.language_code == "en-IN"
    assert response.format == "wav"
    assert len(response.audio_base64) > 1000


import socket

def _is_server_listening() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _is_server_listening(), reason="Requires live uvicorn server on localhost:8000")
@pytest.mark.asyncio
async def test_08_voice_websocket_full_turn_with_tts():
    """
    Test 8: Full WebSocket Turn with Sarvam TTS.
    Connects to ws://localhost:8000/ws/voice, sends text input,
    observes: thinking -> assistant_text_delta -> assistant_text_done -> speaking -> audio_chunk -> audio_end,
    sends playback_done, and confirms return to idle state.
    """
    import json
    import websockets

    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        # 1. Initial connection ACK
        init_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert init_msg["type"] == "connected"
        assert init_msg["tts_provider"] == "sarvam"

        # 2. Send text prompt
        await ws.send(json.dumps({"type": "text_input", "text": "Hello! Say hi in one brief sentence."}))

        received_thinking = False
        received_deltas = False
        received_done = False
        received_speaking = False
        received_audio = False
        received_audio_end = False

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")
            print(f"[Test 8 Event] {mtype} - {msg.get('state', '')}")

            if mtype in ("state", "state_change") and msg.get("state") == "thinking":
                received_thinking = True
            elif mtype == "assistant_text_delta":
                received_deltas = True
            elif mtype == "assistant_text_done":
                received_done = True
            elif mtype in ("state", "state_change") and msg.get("state") == "speaking":
                received_speaking = True
            elif mtype in ("assistant_audio", "audio_chunk"):
                received_audio = True
                assert msg.get("format") == "wav"
                assert len(msg.get("data", "")) > 100
            elif mtype == "audio_end":
                received_audio_end = True
                break
            elif mtype == "error":
                print(f"[Test 8 Error] {msg.get('message')}")
                break


        assert received_thinking, "Did not receive thinking state"
        assert received_deltas, "Did not receive assistant text deltas"
        assert received_done, "Did not receive assistant_text_done"
        assert received_speaking, "Did not receive speaking state"
        assert received_audio, "Did not receive synthesized audio chunk"
        assert received_audio_end, "Did not receive audio_end"

        # 3. Simulate React Native completing audio playback
        await ws.send(json.dumps({"type": "playback_done"}))
        raw_idle = await asyncio.wait_for(ws.recv(), timeout=10.0)
        idle_msg = json.loads(raw_idle)
        assert idle_msg["type"] in ("state", "state_change")
        assert idle_msg["state"] == "idle"


@pytest.mark.skipif(not _is_server_listening(), reason="Requires live uvicorn server on localhost:8000")
@pytest.mark.asyncio
async def test_09_voice_websocket_interruption():
    """
    Test 9: WebSocket Interruption mid-turn.
    Initiates a turn, sends an interrupt frame,
    and verifies that the assistant halts and returns to idle state immediately.
    """
    import json
    import websockets

    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        init_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert init_msg["type"] == "connected"

        # Send prompt
        await ws.send(json.dumps({"type": "text_input", "text": "Explain quantum mechanics in detail."}))

        # Wait for thinking state (following transcript frame)
        while True:
            raw_msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw_msg)
            if msg.get("type") in ("state", "state_change") and msg.get("state") == "thinking":
                break

        # Send interrupt immediately
        await ws.send(json.dumps({"type": "interrupt"}))


        interrupted_received = False
        idle_received = False

        for _ in range(10):
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            m = json.loads(raw)
            if m.get("type") == "interrupted":
                interrupted_received = True
            elif m.get("type") in ("state", "state_change") and m.get("state") == "idle":
                idle_received = True
                break

        assert interrupted_received, "Did not receive interrupted event"
        assert idle_received, "Did not return to idle state after interruption"

