"""
End-to-End Real-Time Voice Pipeline Test Suite.
Validates the complete milestone:
VOICE -> STT -> GEMINI -> TTS -> VOICE

Verifies:
1. Complete Voice Pipeline & State Machine:
   idle -> listening -> transcribing -> thinking -> speaking -> interrupted -> error
2. Natural Conversation & Multi-Turn Context:
   Turn 1: "Hello, my name is Ragul." -> Gemini -> Sarvam TTS audio
   Turn 2: "What did I just tell you?" -> Gemini uses DB conversation context, remembers "Ragul"
3. Real Speech Audio Streaming (WAV chunks -> Sarvam STT -> Gemini -> Sarvam TTS audio -> audio_end)
4. Barge-In Interruption:
   User interrupts during playback -> TTS halted -> cancelled -> switches to listening -> answers new query
5. Error Resilience:
   Invalid audio, provider errors handled cleanly without crashing or closing WebSocket
6. Greeting flow on connect
"""

import asyncio
import base64
import json
import socket
import pytest
import websockets
from unittest.mock import patch, AsyncMock

from app.core.config import settings
from app.voice import voice_manager, STTException, TTSException

def _is_server_listening() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
            return True
    except OSError:
        return False

pytestmark = pytest.mark.skipif(not _is_server_listening(), reason="Requires live uvicorn server on localhost:8000")


@pytest.mark.asyncio
async def test_01_greeting_flow():
    """
    Test 1: Greeting Flow.
    When a client connects and requests a greeting (or connects with greeting=true),
    the assistant produces personalized greeting text and streams Sarvam TTS audio.
    """
    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        # 1. Connection acknowledgement
        conn_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert conn_msg["type"] == "connected"
        assert conn_msg["stt_provider"] == "sarvam"
        assert conn_msg["llm_provider"] == "gemini"
        assert conn_msg["tts_provider"] == "sarvam"
        assert conn_msg["conversation_id"] is not None

        # 2. Request greeting
        await ws.send(json.dumps({"type": "greeting_request"}))

        received_text = ""
        received_speaking = False
        received_audio_chunks = 0
        received_audio_end = False

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype in ("assistant_text_done", "ai_response_text"):
                received_text = msg.get("text", "")
            elif mtype in ("state", "state_change") and msg.get("state") == "speaking":
                received_speaking = True
            elif mtype in ("assistant_audio", "audio_chunk"):
                received_audio_chunks += 1
                assert len(msg.get("data", "")) > 100
                assert msg.get("format") == "wav"
            elif mtype == "audio_end":
                received_audio_end = True
                break

        assert "Aura" in received_text or "help" in received_text.lower()
        assert received_speaking, "VoiceOrb should transition to speaking"
        assert received_audio_chunks > 0, "Should receive synthesized audio chunks"
        assert received_audio_end, "Should receive audio_end frame"

        # 3. Simulate playback completed on client
        await ws.send(json.dumps({"type": "playback_done"}))
        idle_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        idle_msg = json.loads(idle_raw)
        assert idle_msg["type"] in ("state", "state_change")
        assert idle_msg["state"] == "idle"


@pytest.mark.asyncio
async def test_02_ragul_multi_turn_context_continuity():
    """
    Test 2: Natural Multi-Turn Conversation with Context Retention.
    User Turn 1: "Hello, my name is Ragul."
    -> STT/Text -> Gemini -> Sarvam TTS audio.
    User Turn 2: "What did I just tell you?"
    -> Gemini must utilize the active PostgreSQL conversation history and state that the user's name is Ragul.
    """
    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        init_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        init_msg = json.loads(init_raw)
        assert init_msg["type"] == "connected"
        conv_id = init_msg["conversation_id"]
        assert conv_id is not None

        # --- TURN 1: User introduces name ---
        print("\n[Turn 1] Sending: 'Hello, my name is Ragul.'")
        await ws.send(json.dumps({
            "type": "text_input",
            "text": "Hello, my name is Ragul."
        }))

        t1_transcript = False
        t1_thinking = False
        t1_speaking = False
        t1_text = ""
        t1_audio_chunks = 0

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "transcript":
                t1_transcript = True
                assert "Ragul" in msg.get("text", "")
            elif mtype in ("state", "state_change") and msg.get("state") == "thinking":
                t1_thinking = True
            elif mtype in ("assistant_text_done", "ai_response_text"):
                t1_text = msg.get("text", "")
            elif mtype in ("state", "state_change") and msg.get("state") == "speaking":
                t1_speaking = True
            elif mtype in ("assistant_audio", "audio_chunk"):
                t1_audio_chunks += 1
            elif mtype == "audio_end":
                break

        assert t1_transcript, "Should receive transcript frame"
        assert t1_thinking, "Should transition to thinking"
        assert t1_speaking, "Should transition to speaking"
        assert len(t1_text) > 0, "Gemini should generate response text"
        assert t1_audio_chunks > 0, "Sarvam TTS should synthesize audio chunks"

        # Signal audio finished on speaker
        await ws.send(json.dumps({"type": "playback_done"}))
        idle_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        assert json.loads(idle_raw)["state"] == "idle"

        # --- TURN 2: Follow-up question relying on memory ---
        print("\n[Turn 2] Sending: 'What did I just tell you?'")
        await ws.send(json.dumps({
            "type": "text_input",
            "text": "What did I just tell you?"
        }))

        t2_text = ""
        t2_audio_chunks = 0

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype in ("assistant_text_done", "ai_response_text"):
                t2_text = msg.get("text", "")
            elif mtype in ("assistant_audio", "audio_chunk"):
                t2_audio_chunks += 1
            elif mtype == "audio_end":
                break

        print(f"[Turn 2 Response] {t2_text}")
        assert "ragul" in t2_text.lower(), f"Gemini failed to retain context! Response: '{t2_text}'"
        assert t2_audio_chunks > 0, "Sarvam TTS should synthesize response audio"

        await ws.send(json.dumps({"type": "playback_done"}))
        idle_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        assert json.loads(idle_raw)["state"] == "idle"


@pytest.mark.asyncio
async def test_03_real_audio_streaming_stt_gemini_tts():
    """
    Test 3: Raw Audio Streaming Pipeline.
    Streams real audio WAV bytes through WebSocket:
    audio_start -> audio_chunk... -> audio_end
    -> Sarvam STT -> Gemini -> Sarvam TTS -> assistant_audio -> audio_end
    """
    tts = voice_manager.get_tts_provider("sarvam")
    prompt_speech = await tts.synthesize(text="What is the weather today?")
    audio_bytes = prompt_speech.audio_bytes

    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        await ws.recv() # connected

        # 1. Start audio streaming
        await ws.send(json.dumps({"type": "audio_start"}))
        st = json.loads(await ws.recv())
        assert st["state"] == "listening"

        # 2. Send chunks
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            await ws.send(json.dumps({
                "type": "audio_chunk",
                "data": base64.b64encode(chunk).decode("utf-8")
            }))
            await asyncio.sleep(0.01)

        # 3. End audio stream
        await ws.send(json.dumps({"type": "audio_end"}))

        received_transcribing = False
        received_thinking = False
        received_speaking = False
        received_audio = False
        transcript_text = ""
        response_text = ""

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=40.0)
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype in ("state", "state_change"):
                s = msg.get("state")
                if s == "transcribing":
                    received_transcribing = True
                elif s == "thinking":
                    received_thinking = True
                elif s == "speaking":
                    received_speaking = True
            elif mtype == "transcript":
                transcript_text = msg.get("text", "")
            elif mtype in ("assistant_text_done", "ai_response_text"):
                response_text = msg.get("text", "")
            elif mtype in ("assistant_audio", "audio_chunk"):
                received_audio = True
            elif mtype == "audio_end":
                break

        print(f"\n[Real Audio STT] Detected transcript: '{transcript_text}'")
        print(f"[Real Audio Gemini] Assistant response: '{response_text}'")

        assert received_transcribing, "State machine must transition to transcribing"
        assert len(transcript_text) > 0, "Sarvam STT must transcribe spoken audio"
        assert received_thinking, "State machine must transition to thinking"
        assert len(response_text) > 0, "Gemini must generate response"
        assert received_speaking, "State machine must transition to speaking"
        assert received_audio, "Sarvam TTS must stream assistant audio"

        await ws.send(json.dumps({"type": "playback_done"}))
        idle_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
        assert json.loads(idle_raw)["state"] == "idle"


@pytest.mark.asyncio
async def test_04_barge_in_interruption():
    """
    Test 4: Interruption / Barge-in mid-turn.
    User asks a detailed query. While assistant is generating/speaking,
    the user interrupts by sending 'interrupt'.
    The assistant must immediately halt TTS and cancel the response,
    acknowledge interruption, and be ready for new input.
    """
    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        await ws.recv() # connected

        # Request long speech
        await ws.send(json.dumps({
            "type": "text_input",
            "text": "Count slowly from one to twenty and describe each number in detail."
        }))

        # Wait until thinking or speaking begins
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
            msg = json.loads(raw)
            if msg.get("type") in ("state", "state_change") and msg.get("state") in ("thinking", "speaking"):
                break

        # User interrupts!
        print("\n[Barge-in] Sending interrupt frame mid-speech...")
        await ws.send(json.dumps({"type": "interrupt"}))

        interrupted_seen = False
        settled_to_idle = False

        for _ in range(8):
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "interrupted":
                interrupted_seen = True
            elif msg.get("type") in ("state", "state_change") and msg.get("state") in ("idle", "interrupted"):
                settled_to_idle = True
                if interrupted_seen:
                    break

        assert interrupted_seen, "Server must emit interrupted event on barge-in"

        # Now send a clean new query to verify pipeline recovered and accepts new input
        await ws.send(json.dumps({
            "type": "text_input",
            "text": "What is two plus two? Answer in one word."
        }))

        new_text = ""
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            msg = json.loads(raw)
            if msg.get("type") in ("assistant_text_done", "ai_response_text"):
                new_text = msg.get("text", "")
            elif msg.get("type") == "audio_end":
                break

        assert "4" in new_text or "four" in new_text.lower(), f"Expected '4' or 'four', got '{new_text}'"


@pytest.mark.asyncio
async def test_05_error_handling_resilience():
    """
    Test 5: Error Handling and Resilience.
    1. Empty audio buffer handled gracefully without crashing.
    2. Invalid audio data handled gracefully without crashing.
    3. Simulated STT error returns error frame and settles to idle.
    4. Simulated TTS error returns error frame and settles to idle.
    5. WebSocket remains alive and fully functional after errors.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    # 1. Live WebSocket tests for empty audio and resilience
    uri = "ws://localhost:8000/ws/voice"
    async with websockets.connect(uri) as ws:
        await ws.recv() # connected

        # 1a. Empty audio send
        await ws.send(json.dumps({"type": "audio_start"}))
        await ws.recv() # listening
        await ws.send(json.dumps({"type": "audio_end"})) # no chunks sent

        # Should transition: transcribing -> idle
        msg1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert msg1["state"] == "transcribing"
        msg2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=10.0))
        assert msg2["state"] == "idle"

        # 1b. Follow-up turn succeeds after empty audio
        await ws.send(json.dumps({"type": "text_input", "text": "Are you online?"}))
        healthy = False
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=20.0)
            msg = json.loads(raw)
            if msg.get("type") in ("assistant_text_done", "ai_response_text"):
                healthy = True
            elif msg.get("type") == "audio_end":
                break
        assert healthy, "WebSocket should remain healthy after empty audio"

    # 2. In-process provider failure tests using TestClient
    client = TestClient(app)

    # 2a. Simulated STT failure
    with patch.object(
        voice_manager.get_stt_provider("sarvam"),
        "transcribe",
        new_callable=AsyncMock,
        side_effect=STTException("Sarvam STT service unavailable")
    ):
        with client.websocket_connect("/ws/voice") as ws:
            ws.receive_json() # connected
            ws.send_json({"type": "audio_start"})
            ws.receive_json() # listening
            ws.send_json({
                "type": "audio_chunk",
                "data": base64.b64encode(b"\x00" * 1000).decode("utf-8")
            })
            ws.send_json({"type": "audio_end"})

            received_stt_error = False
            for _ in range(5):
                msg = ws.receive_json()
                if msg.get("type") == "error" and msg.get("code") == "STT_ERROR":
                    received_stt_error = True
                    break

            assert received_stt_error, "Server should emit STT_ERROR on provider failure"

    # 2b. Simulated TTS failure
    async def mock_tts_stream_error(*args, **kwargs):
        raise TTSException("Sarvam TTS service unavailable")
        yield  # Make it an async generator

    with patch.object(
        voice_manager.get_tts_provider("sarvam"),
        "stream",
        side_effect=mock_tts_stream_error
    ):
        with client.websocket_connect("/ws/voice") as ws:
            ws.receive_json() # connected
            ws.send_json({"type": "text_input", "text": "Test TTS error handling"})

            received_tts_error = False
            for _ in range(15):
                msg = ws.receive_json()
                if msg.get("type") == "error" and msg.get("code") == "TTS_SYNTHESIS_ERROR":
                    received_tts_error = True
                    break

            assert received_tts_error, "Server should emit TTS_SYNTHESIS_ERROR on TTS failure"
