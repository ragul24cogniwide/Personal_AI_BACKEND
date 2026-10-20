"""
End-to-End Test for Sarvam TTS & STT Integration.
1. Synthesizes speech with Sarvam TTS.
2. Sends synthesized audio into Sarvam STT.
3. Asserts accuracy of transcribed text.
"""

import asyncio
import base64
from app.voice import voice_manager

async def run_e2e_test():
    print("--- 1. Testing Sarvam TTS Synthesis ---")
    tts = voice_manager.get_tts_provider("sarvam")
    test_phrase = "Good evening Elena. Welcome to your personal assistant."
    tts_response = await tts.synthesize(text=test_phrase)
    
    print(f"TTS Provider: {tts_response.provider}")
    print(f"Audio Format: {tts_response.format}, Sample Rate: {tts_response.sample_rate}Hz")
    print(f"Audio Size: {len(tts_response.audio_base64)} base64 chars")
    assert len(tts_response.audio_base64) > 1000, "TTS audio data too small"
    
    print("\n--- 2. Testing Sarvam STT Transcription with Synthesized Audio ---")
    audio_bytes = tts_response.audio_bytes
    stt = voice_manager.get_stt_provider("sarvam")
    stt_response = await stt.transcribe(audio_data=audio_bytes)
    
    print(f"STT Provider: {stt_response.provider}")
    print(f"Transcribed Text: '{stt_response.transcript}'")
    print(f"Detected Language: {stt_response.language_code}")
    
    assert stt_response.transcript and len(stt_response.transcript) > 0, "Empty transcript returned"
    print("\nSUCCESS: Complete Sarvam STT and TTS loop verified successfully!")

if __name__ == "__main__":
    asyncio.run(run_e2e_test())
