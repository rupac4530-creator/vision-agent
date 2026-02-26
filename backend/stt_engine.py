"""
stt_engine.py — Speech-to-Text Engine
Supports: Whisper (local), Gemini STT (cloud fallback), Browser Web Speech API stub

Usage:
    from stt_engine import STTEngine
    stt = STTEngine()
    text = stt.transcribe_bytes(audio_bytes, fmt="wav")
"""
import os
import io
import logging
import tempfile
import base64
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("stt_engine")


class STTEngine:
    """Multi-provider speech-to-text engine with automatic fallback."""

    def __init__(self):
        self._whisper_model = None
        self._whisper_name = os.getenv("WHISPER_MODEL", "base")
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")

    def _load_whisper(self):
        if self._whisper_model is not None:
            return True
        try:
            import whisper  # type: ignore
            logger.info("Loading Whisper model: %s", self._whisper_name)
            self._whisper_model = whisper.load_model(self._whisper_name)
            return True
        except ImportError:
            logger.warning("openai-whisper not installed — STT via browser only")
            return False
        except Exception as e:
            logger.warning("Whisper load failed: %s", e)
            return False

    def transcribe_bytes(self, audio_bytes: bytes, fmt: str = "wav") -> dict:
        """Transcribe raw audio bytes. Returns {text, provider, latency_ms}."""
        t0 = time.time()

        # Try Whisper local
        if self._load_whisper():
            try:
                with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                result = self._whisper_model.transcribe(tmp_path, fp16=False)
                os.unlink(tmp_path)
                return {
                    "text": result.get("text", "").strip(),
                    "provider": "whisper-local",
                    "model": self._whisper_name,
                    "latency_ms": round((time.time() - t0) * 1000),
                }
            except Exception as e:
                logger.warning("Whisper transcribe error: %s", e)

        # Try Gemini STT (REST API)
        if self._gemini_key:
            try:
                return self._gemini_stt(audio_bytes, fmt, t0)
            except Exception as e:
                logger.warning("Gemini STT error: %s", e)

        # Fallback: empty result
        return {
            "text": "",
            "provider": "unavailable",
            "model": "none",
            "latency_ms": round((time.time() - t0) * 1000),
            "error": "No STT provider available. Use browser Web Speech API.",
        }

    def _gemini_stt(self, audio_bytes: bytes, fmt: str, t0: float) -> dict:
        """Use Gemini multimodal API for audio transcription."""
        import requests
        b64 = base64.b64encode(audio_bytes).decode()
        mime = "audio/wav" if fmt == "wav" else "audio/mpeg"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={self._gemini_key}"
        )
        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": mime, "data": b64}},
                    {"text": "Transcribe this audio accurately. Return ONLY the transcribed text, nothing else."},
                ]
            }],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1000},
        }
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {
            "text": text,
            "provider": "gemini-stt",
            "model": "gemini-2.0-flash",
            "latency_ms": round((time.time() - t0) * 1000),
        }

    def transcribe_base64(self, audio_b64: str, fmt: str = "wav") -> dict:
        """Convenience wrapper: base64 audio → transcription."""
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return {"text": "", "provider": "error", "error": f"Invalid base64: {e}"}
        return self.transcribe_bytes(audio_bytes, fmt)


# Singleton
stt_engine = STTEngine()
