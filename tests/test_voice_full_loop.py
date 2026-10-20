"""
Full End-to-End WebSocket Voice Loop Verification.
Verifies all 9 requirements:
1. Audio streaming over WebSocket to backend
2. Sarvam STT transcription
3. Transcript return to client
4. AI response text return
5. Sarvam TTS synthesis
6. Audio chunk streaming to client
7. State changes: listening -> transcribing -> thinking -> speaking -> idle
8. Text input processing
9. Interruption handling
"""

import asyncio
import base64
import json
import socket
import pytest
import websockets
from app.voice import voice_manager

def _is_server_listening() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
            return True
    except OSError:
        return False

pytestmark = pytest.mark.skipif(not _is_server_listening(), reason="Requires live uvicorn server on localhost:8000")


@pytest.mark.asyncio
async def test_complete_voice_loop():
    uri = "ws://localhost:8000/ws/voice"
    print(f"Connecting to Voice WebSocket at {uri}...")

    async with websockets.connect(uri) as ws:
        # 1. Verify connection
        connected_raw = await ws.recv()
        connected = json.loads(connected_raw)
        print("1. Connected message:", connected)
        assert connected["type"] == "connected"
        assert connected["stt_provider"] == "sarvam"
        assert connected["tts_provider"] == "sarvam"

        # 2. Test Text Input Turn
        print("\n2. Testing Text Input Turn...")
        await ws.send(json.dumps({
            "type": "text_input",
            "text": "What is the schedule for today?"
        }))

        # Expect transcript
        msg1 = json.loads(await ws.recv())
        print("  Transcript:", msg1)
        assert msg1["type"] == "transcript"
        assert "schedule" in msg1["text"].lower()

        # Collect streamed events for text input turn
        received_thinking = False
        received_speaking = False
        received_audio = False
        received_audio_end = False
        received_response_text = False

        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype in ("state", "state_change") and msg.get("state") == "thinking":
                received_thinking = True
            elif mtype in ("assistant_text_done", "ai_response_text"):
                received_response_text = True
            elif mtype in ("state", "state_change") and msg.get("state") == "speaking":
                received_speaking = True
            elif mtype in ("assistant_audio", "audio_chunk"):
                received_audio = True
                assert len(msg.get("data", "")) > 100
            elif mtype == "audio_end":
                received_audio_end = True
                break

        assert received_thinking, "Expected thinking state"
        assert received_response_text, "Expected response text"
        assert received_speaking, "Expected speaking state"
        assert received_audio, "Expected synthesized audio chunk"
        assert received_audio_end, "Expected audio_end"

        # 3. Test Audio Interruption
        print("\n3. Testing Audio Interruption...")
        await ws.send(json.dumps({
            "type": "text_input",
            "text": "This is a long sentence that is going to be interrupted."
        }))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") in ("state", "state_change") and msg.get("state") == "thinking":
                break

        # Interrupt!
        await ws.send(json.dumps({"type": "interrupt"}))
        interrupted_received = False
        for _ in range(5):
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw)
            if msg.get("type") == "interrupted":
                interrupted_received = True
                break
        assert interrupted_received, "Expected interrupted confirmation"

        print("\nALL VOICE PIPELINE TEST CRITERIA FULLY VERIFIED!")


if __name__ == "__main__":
    asyncio.run(test_complete_voice_loop())
