"""
Test script for Voice WebSocket protocol verification.
"""

from uuid import uuid4
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.core.security import create_access_token


def test_voice_ws():
    """Verifies WebSocket handshake, start_listen, and interrupt events."""
    token, _, _ = create_access_token(subject=str(uuid4()))
    with TestClient(app) as client:
        with client.websocket_connect(f"/ws/voice?token={token}") as ws:
            # 1. Receive connected message
            connected_msg = ws.receive_json()
            assert connected_msg["type"] == "connected"

            # 2. Test start_listen
            ws.send_json({"type": "start_listen"})
            msg1 = ws.receive_json()
            if msg1.get("type") == "interrupted":
                state_msg = ws.receive_json()
            else:
                state_msg = msg1
            assert state_msg.get("state") == "listening"

            # 3. Test interrupt
            ws.send_json({"type": "interrupt"})
            interrupted_msg = ws.receive_json()
            assert interrupted_msg["type"] == "interrupted"

            idle_msg = ws.receive_json()
            assert idle_msg["state"] in ("idle", "interrupted")

            print("Voice WebSocket protocol test PASSED!")


if __name__ == "__main__":
    test_voice_ws()
