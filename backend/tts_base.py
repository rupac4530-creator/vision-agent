# backend/tts_base.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned TTS (Text-to-Speech) Base Class.

Abstract base for all TTS provider implementations. Provides:
- Streaming audio synthesis
- Output format configuration (sample rate, channels)
- Audio event emission with chunked delivery
- Synthesis lifecycle events (start/complete/error)

Subclasses: CartesiaTTS, ElevenLabsTTS, EdgeTTS, etc.

Usage:
    class MyTTS(TTSBase):
        async def synthesize(self, text: str) -> AsyncIterator[bytes]:
            async for chunk in my_provider.stream(text):
                yield chunk
"""

import abc
import uuid
import time
import logging
from typing import Any, AsyncIterator, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("tts_base")


@dataclass
class TTSAudioChunk:
    """A chunk of synthesized audio."""
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "pcm_s16"  # pcm_s16, pcm_f32, mp3, opus
    chunk_index: int = 0
    is_final: bool = False
    duration_ms: float = 0

    def to_dict(self) -> Dict:
        return {
            "chunk_index": self.chunk_index,
            "data_len": len(self.data),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "is_final": self.is_final,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class TTSEvent:
    """Event emitted during TTS synthesis."""
    type: str  # "start", "audio", "complete", "error"
    synthesis_id: str = ""
    provider: str = ""
    text: str = ""
    chunk_index: int = 0
    total_duration_ms: float = 0
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


class TTSBase(abc.ABC):
    """Abstract base class for Text-to-Speech implementations.

    SDK-aligned pattern from Vision-Agents agents-core/vision_agents/core/tts/tts.py.
    Handles streaming synthesis, audio format conversion, and event emission.
    """

    def __init__(self, provider_name: Optional[str] = None):
        self.provider_name = provider_name or self.__class__.__name__
        self.session_id = str(uuid.uuid4())
        self._event_listeners: List = []

        # Output format configuration
        self._sample_rate: int = 16000
        self._channels: int = 1
        self._format: str = "pcm_s16"

        # Stats
        self._total_syntheses: int = 0
        self._total_chars: int = 0
        self._total_audio_ms: float = 0

    # ── Abstract method (subclasses must implement) ──────────────

    @abc.abstractmethod
    async def synthesize(self, text: str, **kwargs) -> AsyncIterator[bytes]:
        """Synthesize text to audio, yielding raw audio chunks.

        Args:
            text: Text to synthesize

        Yields:
            Raw audio bytes in the configured format
        """
        ...

    # ── Output Format ────────────────────────────────────────────

    def set_output_format(self, sample_rate: int = 16000,
                          channels: int = 1, format: str = "pcm_s16"):
        """Set desired output audio format."""
        self._sample_rate = sample_rate
        self._channels = channels
        self._format = format

    # ── High-level Synthesis ─────────────────────────────────────

    async def speak(self, text: str, **kwargs) -> List[TTSAudioChunk]:
        """Synthesize text and return all audio chunks.

        This is the high-level API that handles lifecycle events.
        """
        synthesis_id = str(uuid.uuid4())[:8]
        self._total_syntheses += 1
        self._total_chars += len(text)
        t0 = time.time()

        # Emit start event
        self._emit(TTSEvent(
            type="start", synthesis_id=synthesis_id,
            provider=self.provider_name, text=text[:100],
        ))

        chunks = []
        try:
            chunk_idx = 0
            async for audio_data in self.synthesize(text, **kwargs):
                # Calculate duration from audio data length
                bytes_per_sample = 2 if "s16" in self._format else 4
                samples = len(audio_data) / (bytes_per_sample * self._channels)
                duration_ms = (samples / self._sample_rate) * 1000

                chunk = TTSAudioChunk(
                    data=audio_data,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    format=self._format,
                    chunk_index=chunk_idx,
                    duration_ms=duration_ms,
                )
                chunks.append(chunk)
                chunk_idx += 1

                self._emit(TTSEvent(
                    type="audio", synthesis_id=synthesis_id,
                    provider=self.provider_name, chunk_index=chunk_idx,
                ))

            # Mark last chunk as final
            if chunks:
                chunks[-1].is_final = True

            total_ms = sum(c.duration_ms for c in chunks)
            self._total_audio_ms += total_ms

            self._emit(TTSEvent(
                type="complete", synthesis_id=synthesis_id,
                provider=self.provider_name, text=text[:100],
                total_duration_ms=total_ms,
            ))

            return chunks

        except Exception as e:
            self._emit(TTSEvent(
                type="error", synthesis_id=synthesis_id,
                provider=self.provider_name, error=str(e),
            ))
            logger.error("TTS synthesis error: %s", e)
            return []

    # ── Event Emission ───────────────────────────────────────────

    def on_event(self, listener):
        self._event_listeners.append(listener)

    def _emit(self, event: TTSEvent):
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("TTS event listener error: %s", e)

    # ── Lifecycle ────────────────────────────────────────────────

    async def close(self):
        logger.info("TTS '%s' closed", self.provider_name)

    def get_stats(self) -> Dict:
        return {
            "provider": self.provider_name,
            "total_syntheses": self._total_syntheses,
            "total_chars": self._total_chars,
            "total_audio_ms": round(self._total_audio_ms, 1),
            "output_format": {
                "sample_rate": self._sample_rate,
                "channels": self._channels,
                "format": self._format,
            },
        }
