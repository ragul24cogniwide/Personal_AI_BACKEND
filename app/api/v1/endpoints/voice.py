"""
Voice Endpoints & Real-time Full-Duplex WebSocket Protocol.
Complete Pipeline: React Native Mic -> WebSocket -> Sarvam STT -> Gemini LLM -> Sarvam TTS -> WebSocket -> React Native Speaker.
Supports Voice State Machine: idle, listening, transcribing, thinking, speaking, interrupted, error.
Handles continuous turn-taking, interruption (barge-in), automatic greetings, and error resilience.
"""

import asyncio
import base64
import json
import logging
import re
import time
from typing import Optional, Set
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("app.voice.pipeline")

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.security import decode_token
from app.database.session import AsyncSessionLocal
from app.ai import ai_service
from app.models.user import User
from app.services.greeting import greeting_service
from app.voice import (
    voice_manager,
    STTException,
    TTSException,
    ProviderConfigurationError,
    VoiceSettings,
)

router = APIRouter()


class TestTTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    language_code: Optional[str] = None
    model: Optional[str] = None
    pace: Optional[float] = None
    pitch: Optional[float] = None
    loudness: Optional[float] = None


@router.post("/test-tts", summary="Test Text-to-Speech synthesis")
async def test_tts(request: TestTTSRequest):
    """Direct HTTP test endpoint for Text-to-Speech synthesis."""
    try:
        tts_provider = voice_manager.get_tts_provider()
        response = await tts_provider.synthesize(
            text=request.text,
            speaker=request.speaker,
            language_code=request.language_code,
            model=request.model,
            pace=request.pace,
            pitch=request.pitch,
            loudness=request.loudness,
        )
        return {
            "status": "success",
            "provider": response.provider,
            "speaker": response.speaker,
            "language_code": response.language_code,
            "audio_base64": response.audio_base64,
            "format": response.format,
            "sample_rate": response.sample_rate,
        }
    except ProviderConfigurationError as pce:
        raise HTTPException(status_code=400, detail=str(pce))
    except TTSException as te:
        raise HTTPException(status_code=502, detail=str(te))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {str(exc)}")


@router.post("/test-stt", summary="Test Speech-to-Text transcription")
async def test_stt(file: UploadFile = File(...), language_code: Optional[str] = None):
    """Direct HTTP test endpoint for Speech-to-Text transcription."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        stt_provider = voice_manager.get_stt_provider()
        response = await stt_provider.transcribe(
            audio_data=content,
            language_code=language_code,
        )
        return {
            "status": "success",
            "provider": response.provider,
            "transcript": response.transcript,
            "language_code": response.language_code,
        }
    except ProviderConfigurationError as pce:
        raise HTTPException(status_code=400, detail=str(pce))
    except STTException as se:
        raise HTTPException(status_code=502, detail=str(se))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"STT transcription error: {str(exc)}")


@router.get("/greeting", summary="Fetch context-aware greeting for user")
async def get_greeting_endpoint(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Evaluates user onboarding status:
    - New user: returns onboarding flow action
    - Returning user: returns Gemini-synthesized context-aware greeting
    """
    return await greeting_service.generate_greeting(db=db, user_id=current_user.id)


@router.websocket("")
@router.websocket("/")
@router.websocket("/ws")
async def voice_websocket_endpoint(websocket: WebSocket):
    """
    Real-time Full-Duplex Voice WebSocket Handler.
    Protocol:
      Client -> Server: audio_start, audio_chunk, audio_end, interrupt, playback_done, greeting_request, text_input, update_settings
      Server -> Client: connected, state, transcript, assistant_text_delta, assistant_text_done, assistant_audio, audio_end, interrupted, error
    """
    await websocket.accept()

    # Session State
    audio_buffer = bytearray()
    current_task: Optional[asyncio.Task] = None
    active_user_id: Optional[UUID] = None
    active_conversation_id: Optional[UUID] = None
    active_turn_id: str = f"turn_{int(time.time()*1000)}_{uuid4().hex[:6]}"
    processed_message_ids: Set[str] = set()

    # Configurable Session Voice Settings
    session_voice_settings = VoiceSettings(
        voice=settings.SARVAM_TTS_SPEAKER,
        language=settings.SARVAM_LANGUAGE_CODE,
        speech_rate=settings.SARVAM_TTS_PACE,
        model=settings.SARVAM_TTS_MODEL,
        pitch=settings.SARVAM_TTS_PITCH,
        loudness=settings.SARVAM_TTS_LOUDNESS,
        sample_rate=settings.SARVAM_TTS_SAMPLE_RATE,
    )

    # Check query params for token and conversation
    token_param = websocket.query_params.get("token")
    if token_param:
        try:
            token_payload = decode_token(token_param)
            active_user_id = UUID(token_payload.get("sub"))
        except Exception:
            pass

    conv_param = websocket.query_params.get("conversation_id")
    if conv_param:
        try:
            active_conversation_id = UUID(conv_param)
        except Exception:
            pass

    async def send_json(data: dict):
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    async def send_state(state: str, message: Optional[str] = None):
        """Sends standardized state frame for VoiceOrb reactivity."""
        await send_json({
            "type": "state",
            "state": state,
            "message": message,
        })

    async def send_audio_chunk(tts_chunk, chunk_index: int = 0, is_final: bool = False, turn_id: Optional[str] = None):
        """Sends standardized assistant_audio frames with monotonic turn_id and chunk index."""
        payload = {
            "type": "assistant_audio",
            "data": tts_chunk.audio_base64,
            "format": tts_chunk.format,
            "sample_rate": tts_chunk.sample_rate,
            "speaker": tts_chunk.speaker,
            "language_code": tts_chunk.language_code,
            "chunk_index": chunk_index,
            "turn_id": turn_id or active_turn_id,
            "is_final": is_final,
            "timestamp": time.time(),
        }
        await send_json(payload)

    async def ensure_active_conversation():
        """Ensures an active conversation exists in PostgreSQL and is linked to the session."""
        nonlocal active_conversation_id, active_user_id
        if not active_conversation_id:
            try:
                async with AsyncSessionLocal() as db:
                    if not active_user_id:
                        user_stmt = select(User.id).limit(1)
                        u_res = await db.execute(user_stmt)
                        active_user_id = u_res.scalar_one_or_none()
                    if active_user_id:
                        conv = await ai_service.get_or_create_conversation(db=db, user_id=active_user_id)
                        active_conversation_id = conv.id
            except Exception as e:
                print(f"[VoiceWS] Conversation resolution warning: {e}")

    # Establish initial conversation binding
    await ensure_active_conversation()

    # Acknowledge connection
    await send_json({
        "type": "connected",
        "stt_provider": settings.STT_PROVIDER,
        "llm_provider": settings.LLM_PROVIDER,
        "tts_provider": settings.TTS_PROVIDER,
        "user_id": str(active_user_id) if active_user_id else None,
        "conversation_id": str(active_conversation_id) if active_conversation_id else None,
        "voice_settings": session_voice_settings.model_dump(),
        "message": "Voice pipeline connected and ready",
    })

    async def send_greeting():
        """Generates and speaks a context-aware greeting using greeting_service & Sarvam TTS."""
        try:
            await ensure_active_conversation()

            async with AsyncSessionLocal() as db:
                greeting_res = await greeting_service.generate_greeting(
                    db=db,
                    user_id=active_user_id,
                )

            # If new user: trigger onboarding flow and do not play returning greeting
            if greeting_res.get("is_new_user"):
                await send_json({
                    "type": "onboarding_required",
                    "action": "onboarding",
                    "message": "User needs onboarding",
                })
                await send_state("idle", "Onboarding required")
                return

            greeting_text = greeting_res.get("greeting_text")
            if not greeting_text:
                greeting_text = "Good morning! Ready to get started?"

            await send_json({
                "type": "greeting",
                "text": greeting_text,
                "user_name": greeting_res.get("user_name"),
                "context_used": greeting_res.get("context_used", {}),
            })
            await send_json({
                "type": "assistant_text_done",
                "text": greeting_text,
            })
            await send_json({
                "type": "ai_response_text",
                "text": greeting_text,
            })

            tts_provider = voice_manager.get_tts_provider()
            chunk_idx = 0
            has_sent_audio = False

            async for tts_chunk in tts_provider.stream(
                text=greeting_text,
                voice_settings=session_voice_settings,
            ):
                if not has_sent_audio:
                    has_sent_audio = True
                    # VoiceOrb transitions to speaking state
                    await send_state("speaking", "Speaking greeting...")

                await send_audio_chunk(tts_chunk, chunk_idx)
                chunk_idx += 1

            await send_json({
                "type": "audio_end",
                "total_chunks": chunk_idx,
            })

        except asyncio.CancelledError:
            # Clean interruption (barge-in): greeting stopped when user started speaking
            await send_json({"type": "interrupted", "message": "Greeting interrupted"})
        except Exception as exc:
            print(f"[VoiceWS] Greeting error: {exc}")
            await send_state("idle", "Ready")

    async def process_turn(text_override: Optional[str] = None):
        """
        High-Performance Full-Duplex Turn Pipeline:
        1. STT with persistent connection pool & latency measurement.
        2. Concurrent Sentence/Clause Pipelining:
           - As Gemini produces tokens, we segment into natural speech units.
           - Each segment is pushed to an asyncio.Queue for immediate asynchronous TTS synthesis.
           - Audio chunks are sent to client as soon as synthesized.
        3. Monotonic turn_id tagging on every frame to prevent overlapping or stale audio.
        4. Safe latency metrics emission and logging (zero private content leaked).
        5. Instant cancellation on user interruption (barge-in).
        """
        nonlocal active_turn_id
        turn_id = f"turn_{int(time.time()*1000)}_{uuid4().hex[:6]}"
        active_turn_id = turn_id
        t_turn_start = time.perf_counter()
        t_stt_start: Optional[float] = None
        t_stt_end: Optional[float] = None
        t_llm_start: Optional[float] = None
        t_llm_first_token: Optional[float] = None
        t_first_audio_sent: Optional[float] = None

        try:
            await ensure_active_conversation()
            transcript = ""

            if text_override:
                transcript = text_override
                stt_duration_ms = 0.0
            else:
                # 1. State: Transcribing
                await send_state("transcribing", "Transcribing your voice with Sarvam STT...")

                if not audio_buffer or len(audio_buffer) < 100:
                    await send_state("idle", "No audio received. Tap or speak to begin.")
                    return

                current_audio = bytes(audio_buffer)
                audio_buffer.clear()

                # Clamp to 28s to strictly respect Sarvam STT limit (16kHz 16-bit mono = 32000 bytes/sec)
                max_bytes = 16000 * 2 * 28
                if len(current_audio) > max_bytes:
                    current_audio = current_audio[-max_bytes:]

                t_stt_start = time.perf_counter()
                stt_provider = voice_manager.get_stt_provider()
                stt_result = await stt_provider.transcribe(current_audio)
                t_stt_end = time.perf_counter()
                stt_duration_ms = (t_stt_end - t_stt_start) * 1000
                transcript = stt_result.transcript

            # If transcription produced no clear words
            if not transcript or not transcript.strip():
                await send_json({
                    "type": "transcript",
                    "text": "",
                    "is_final": True,
                    "turn_id": turn_id,
                })
                await send_state("idle", "Could not detect clear speech. Please speak again.")
                return

            # 2. Emit detected transcript
            await send_json({
                "type": "transcript",
                "text": transcript,
                "is_final": True,
                "turn_id": turn_id,
                "conversation_id": str(active_conversation_id) if active_conversation_id else None,
            })

            # 3. State: Thinking (Gemini LLM is processing)
            await send_state("thinking", "Thinking...")

            # 4. Concurrent Sentence/Clause Pipelining:
            # We spin up an asynchronous TTS worker that synthesizes sentences as soon as Gemini yields them.
            sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            tts_chunk_idx = 0
            has_sent_audio = False

            async def tts_worker():
                nonlocal tts_chunk_idx, has_sent_audio, t_first_audio_sent
                tts_provider = voice_manager.get_tts_provider()
                while True:
                    segment = await sentence_queue.get()
                    if segment is None:
                        # Sentinel indicating end of stream
                        break
                    seg_clean = segment.strip()
                    if not seg_clean:
                        continue
                    try:
                        async for tts_chunk in tts_provider.stream(
                            text=seg_clean,
                            voice_settings=session_voice_settings,
                        ):
                            if not has_sent_audio:
                                has_sent_audio = True
                                t_first_audio_sent = time.perf_counter()
                                await send_state("speaking", "Speaking...")

                            await send_audio_chunk(tts_chunk, tts_chunk_idx, turn_id=turn_id)
                            tts_chunk_idx += 1
                    except asyncio.CancelledError:
                        break
                    except Exception as err:
                        logger.warning(f"[VoiceWS] TTS worker segment error: {err}")

            tts_worker_task = asyncio.create_task(tts_worker())

            full_assistant_text = ""
            accumulated_segment = ""
            t_llm_start = time.perf_counter()

            try:
                async def handle_tool_execution(tool_name: str, result_payload: dict):
                    logger.info(f"[VoiceWS] Tool executed: {tool_name}")
                    if tool_name == "schedule_calendar_event":
                        event_data = result_payload.get("event") or {
                            "title": result_payload.get("title", "Meeting"),
                            "date": result_payload.get("date", "today"),
                            "time": result_payload.get("time", "10:00 AM"),
                        }
                        logger.info(f"[VoiceWS] Dispatching calendar_event_scheduled: {event_data}")
                        await send_json({
                            "type": "calendar_event_scheduled",
                            "turn_id": turn_id,
                            "event": event_data,
                            "message": result_payload.get("message", "Event scheduled"),
                        })
                    await send_json({
                        "type": "tool_executed",
                        "turn_id": turn_id,
                        "tool": tool_name,
                        "result": result_payload,
                    })

                async with AsyncSessionLocal() as db:
                    async for chunk in ai_service.process_turn_stream(
                        db=db,
                        user_id=active_user_id,
                        user_text=transcript,
                        conversation_id=active_conversation_id,
                        on_tool_executed=handle_tool_execution,
                    ):
                        if t_llm_first_token is None:
                            t_llm_first_token = time.perf_counter()

                        full_assistant_text += chunk
                        accumulated_segment += chunk

                        await send_json({
                            "type": "assistant_text_delta",
                            "text": chunk,
                            "turn_id": turn_id,
                        })

                        # Early dispatch heuristic:
                        # Split on sentence boundaries (. ! ? \n) or clause break if segment length > 70
                        if re.search(r"[.!?]\s+$|\n+", accumulated_segment) or (
                            len(accumulated_segment) > 70 and re.search(r"[,;:—]\s+$", accumulated_segment)
                        ):
                            await sentence_queue.put(accumulated_segment.strip())
                            accumulated_segment = ""

                # Push final segment if any remaining
                if accumulated_segment.strip():
                    await sentence_queue.put(accumulated_segment.strip())
                # Sentinel to tell tts_worker to finish
                await sentence_queue.put(None)

                # Wait for TTS audio synthesis to complete
                await tts_worker_task

            except Exception:
                tts_worker_task.cancel()
                raise

            # Complete text response emitted
            await send_json({
                "type": "assistant_text_done",
                "text": full_assistant_text,
                "turn_id": turn_id,
            })
            await send_json({
                "type": "ai_response_text",
                "text": full_assistant_text,
                "turn_id": turn_id,
            })

            if has_sent_audio:
                await send_json({
                    "type": "audio_end",
                    "total_chunks": tts_chunk_idx,
                    "turn_id": turn_id,
                })
            else:
                await send_state("idle", "Ready")

            # 5. Measure & Safely Log Latency Telemetry (Zero private content logged)
            t_turn_end = time.perf_counter()
            llm_ttft_ms = (t_llm_first_token - t_llm_start) * 1000 if (t_llm_first_token and t_llm_start) else 0.0
            tts_ttfa_ms = (t_first_audio_sent - t_llm_first_token) * 1000 if (t_first_audio_sent and t_llm_first_token) else 0.0
            total_turnaround_ms = (t_first_audio_sent - t_turn_start) * 1000 if t_first_audio_sent else ((t_turn_end - t_turn_start) * 1000)

            # Safe latency logging: strictly logs timing and counts, NEVER user messages or transcripts
            logger.info(
                f"[VoiceLatencySafe] turn_id={turn_id} "
                f"stt_ms={stt_duration_ms:.1f} "
                f"llm_ttft_ms={llm_ttft_ms:.1f} "
                f"tts_ttfa_ms={tts_ttfa_ms:.1f} "
                f"turnaround_ms={total_turnaround_ms:.1f} "
                f"chunks={tts_chunk_idx}"
            )

            await send_json({
                "type": "latency_metrics",
                "turn_id": turn_id,
                "metrics": {
                    "stt_ms": round(stt_duration_ms, 1),
                    "llm_ttft_ms": round(llm_ttft_ms, 1),
                    "tts_ttfa_ms": round(tts_ttfa_ms, 1),
                    "total_turnaround_ms": round(total_turnaround_ms, 1),
                    "audio_chunks": tts_chunk_idx,
                },
            })

        except asyncio.CancelledError:
            # Clean interruption handling (barge-in)
            await send_json({
                "type": "interrupted",
                "turn_id": turn_id,
                "message": "Speech canceled by user interruption",
            })

        except ProviderConfigurationError as pce:
            await send_json({
                "type": "error",
                "message": str(pce),
                "code": "CONFIG_ERROR",
            })
            await send_state("error", "Configuration error")
            await send_state("idle", "Ready")

        except STTException as se:
            await send_json({
                "type": "error",
                "message": f"Speech transcription failed: {str(se)}",
                "code": "STT_ERROR",
            })
            await send_state("error", "Transcription failed")
            await send_state("idle", "Ready")

        except Exception as exc:
            await send_json({
                "type": "error",
                "message": f"AI pipeline error: {str(exc)}",
                "code": "PIPELINE_ERROR",
            })
            await send_state("error", "Processing error")
            await send_state("idle", "Ready")

    # If greeting requested via query param (e.g. ws://host/ws/voice?greeting=true)
    if websocket.query_params.get("greeting") == "true":
        current_task = asyncio.create_task(send_greeting())

    try:
        while True:
            raw_message = await websocket.receive()

            if "text" in raw_message:
                try:
                    payload = json.loads(raw_message["text"])
                except Exception:
                    continue

                msg_type = payload.get("type")

                # Deduplicate client messages if msg_id provided
                msg_id = payload.get("msg_id")
                if msg_id:
                    if msg_id in processed_message_ids:
                        continue
                    processed_message_ids.add(msg_id)
                    if len(processed_message_ids) > 1000:
                        processed_message_ids.clear()

                if msg_type in ("audio_start", "start_listen"):
                    # User started speaking -> cancel any ongoing assistant speech immediately (barge-in)
                    had_active_task = False
                    if current_task and not current_task.done():
                        had_active_task = True
                        current_task.cancel()
                        current_task = None
                    active_turn_id = f"turn_{int(time.time()*1000)}_{uuid4().hex[:6]}"
                    audio_buffer.clear()
                    if had_active_task:
                        await send_json({
                            "type": "interrupted",
                            "turn_id": active_turn_id,
                            "message": "User began speaking, previous generation cancelled",
                        })
                    await send_state("listening", "Listening...")

                elif msg_type == "audio_chunk":
                    chunk_base64 = payload.get("data")
                    if chunk_base64:
                        if current_task and not current_task.done():
                            current_task.cancel()
                            current_task = None
                            active_turn_id = f"turn_{int(time.time()*1000)}_{uuid4().hex[:6]}"
                            await send_json({
                                "type": "interrupted",
                                "turn_id": active_turn_id,
                                "message": "User spoke during playback",
                            })
                            await send_state("listening", "Listening...")
                        try:
                            audio_buffer.extend(base64.b64decode(chunk_base64))
                        except Exception:
                            pass

                elif msg_type == "audio_end":
                    # User stopped speaking, initiate complete turn
                    if current_task and not current_task.done():
                        current_task.cancel()
                    current_task = asyncio.create_task(process_turn())

                elif msg_type == "text_input":
                    text = payload.get("text", "")
                    if text:
                        if current_task and not current_task.done():
                            current_task.cancel()
                        current_task = asyncio.create_task(process_turn(text_override=text))

                elif msg_type in ("greeting_request", "get_greeting"):
                    if current_task and not current_task.done():
                        current_task.cancel()
                    current_task = asyncio.create_task(send_greeting())

                elif msg_type == "interrupt":
                    # User interrupted while assistant was speaking or synthesizing
                    if current_task and not current_task.done():
                        current_task.cancel()
                        current_task = None
                    active_turn_id = f"turn_{int(time.time()*1000)}_{uuid4().hex[:6]}"
                    await send_json({
                        "type": "interrupted",
                        "turn_id": active_turn_id,
                        "message": "Audio playback interrupted",
                    })
                    await send_state("interrupted", "Interrupted")
                    await send_state("idle", "Ready")

                elif msg_type == "ping":
                    # Heartbeat ping from client
                    await send_json({
                        "type": "pong",
                        "timestamp": payload.get("timestamp"),
                        "client_timestamp": payload.get("timestamp"),
                        "server_timestamp": time.time(),
                    })

                elif msg_type == "playback_done":
                    # React Native completed playback of all audio chunks
                    await send_state("idle", "Tap the orb or mic to begin")

                elif msg_type in ("update_settings", "set_voice_settings"):
                    settings_data = payload.get("settings", {})
                    if isinstance(settings_data, dict):
                        if "voice" in settings_data:
                            session_voice_settings.voice = settings_data["voice"]
                        if "language" in settings_data:
                            session_voice_settings.language = settings_data["language"]
                        if "speech_rate" in settings_data:
                            session_voice_settings.speech_rate = float(settings_data["speech_rate"])
                        if "pitch" in settings_data:
                            session_voice_settings.pitch = float(settings_data["pitch"])
                        if "loudness" in settings_data:
                            session_voice_settings.loudness = float(settings_data["loudness"])
                        if "model" in settings_data:
                            session_voice_settings.model = settings_data["model"]
                    await send_json({
                        "type": "settings_updated",
                        "settings": session_voice_settings.model_dump(),
                    })

                elif msg_type == "auth":
                    token_str = payload.get("token")
                    if token_str:
                        try:
                            token_payload = decode_token(token_str)
                            active_user_id = UUID(token_payload.get("sub"))
                            await ensure_active_conversation()
                            await send_json({
                                "type": "auth_success",
                                "user_id": str(active_user_id),
                                "conversation_id": str(active_conversation_id) if active_conversation_id else None,
                            })
                        except Exception as e:
                            await send_json({
                                "type": "auth_error",
                                "message": f"Token verification failed: {str(e)}",
                            })

                elif msg_type in ("set_conversation", "switch_conversation"):
                    conv_str = payload.get("conversation_id")
                    if conv_str and active_user_id:
                        try:
                            req_conv_id = UUID(conv_str)
                            async with AsyncSessionLocal() as db:
                                conv = await ai_service.get_or_create_conversation(
                                    db=db,
                                    user_id=active_user_id,
                                    conversation_id=req_conv_id,
                                )
                                active_conversation_id = conv.id
                            await send_json({
                                "type": "conversation_set",
                                "conversation_id": str(active_conversation_id),
                                "title": conv.title,
                            })
                        except Exception as e:
                            print(f"[VoiceWS] Conversation switch warning: {e}")


            elif "bytes" in raw_message:
                audio_buffer.extend(raw_message["bytes"])

    except WebSocketDisconnect:
        if current_task and not current_task.done():
            current_task.cancel()
    except Exception:
        if current_task and not current_task.done():
            current_task.cancel()
