# backend/stt_base.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned STT (Speech-to-Text) Base Class.

Abstract base for all STT provider implementations. Provides:
- Session management with unique IDs
- Partial and final transcript event emission
- Turn detection integration (start/end speaking)
- Error handling with recovery support

Subclasses: DeepgramSTT, WhisperSTT, GoogleSTT, etc.

Usage:
    class MySTT(STTBase):
        async def process_audio(self, audio_data, participant=None):
            # transcribe audio chunk
            text = await my_provider.transcribe(audio_data)
            self._emit_transcript("Hello world", participant)
"""

import abc
import uuid
import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("stt_base")


@dataclass
class TranscriptResponse:
    """Structured transcription response metadata."""
    text: str = ""
    language: str = "en"
    confidence: float = 1.0
    is_final: bool = True
    words: List[Dict] = field(default_factory=list)
    duration_ms: float = 0
    provider: str = ""

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "is_final": self.is_final,
            "duration_ms": self.duration_ms,
            "provider": self.provider,
            "word_count": len(self.words),
        }


@dataclass
class STTEvent:
    """Event emitted by STT processing."""
    type: str  # "transcript", "partial", "error", "turn_start", "turn_end"
    session_id: str = ""
    provider: str = ""
    text: str = ""
    participant_id: Optional[str] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


class STTBase(abc.ABC):
    """Abstract base class for Speech-to-Text implementations.

    SDK-aligned pattern from Vision-Agents agents-core/vision_agents/core/stt/stt.py.
    Handles session management, transcript events, and turn detection.
    """

    closed: bool = False
    started: bool = False
    turn_detection: bool = False

    def __init__(self, provider_name: Optional[str] = None):
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self._event_listeners: List = []
        self._transcripts: List[TranscriptResponse] = []
        self._total_audio_ms: float = 0

    # ── Abstract method (subclasses must implement) ──────────────

    @abc.abstractmethod
    async def process_audio(self, audio_data: bytes,
                            participant_id: Optional[str] = None) -> Optional[TranscriptResponse]:
        """Process an audio chunk and return transcript if available.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, 16kHz mono)
            participant_id: Optional speaker ID

        Returns:
            TranscriptResponse if transcription available, else None
        """
        ...

    # ── Event Emission ───────────────────────────────────────────

    def on_event(self, listener):
        """Register an event listener."""
        self._event_listeners.append(listener)

    def _emit(self, event: STTEvent):
        for listener in self._event_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("STT event listener error: %s", e)

    def _emit_transcript(self, text: str, participant_id: Optional[str] = None,
                         confidence: float = 1.0, response: Optional[TranscriptResponse] = None):
        """Emit a final transcript event."""
        resp = response or TranscriptResponse(
            text=text, confidence=confidence, provider=self.provider_name
        )
        self._transcripts.append(resp)
        self._emit(STTEvent(
            type="transcript", session_id=self.session_id,
            provider=self.provider_name, text=text,
            participant_id=participant_id, confidence=confidence,
        ))

    def _emit_partial_transcript(self, text: str, participant_id: Optional[str] = None,
                                 confidence: float = 0.5):
        """Emit a partial (interim) transcript event."""
        self._emit(STTEvent(
            type="partial", session_id=self.session_id,
            provider=self.provider_name, text=text,
            participant_id=participant_id, confidence=confidence,
        ))

    def _emit_turn_started(self, participant_id: Optional[str] = None,
                           confidence: float = 0.5):
        self._emit(STTEvent(
            type="turn_start", session_id=self.session_id,
            provider=self.provider_name, participant_id=participant_id,
            confidence=confidence,
        ))

    def _emit_turn_ended(self, participant_id: Optional[str] = None,
                         confidence: float = 0.5):
        self._emit(STTEvent(
            type="turn_end", session_id=self.session_id,
            provider=self.provider_name, participant_id=participant_id,
            confidence=confidence,
        ))

    def _emit_error(self, error: Exception, context: str = "",
                    is_recoverable: bool = True):
        """Emit an error event (for temporary/recoverable errors)."""
        self._emit(STTEvent(
            type="error", session_id=self.session_id,
            provider=self.provider_name, text=str(error),
            metadata={"context": context, "recoverable": is_recoverable},
        ))

    # ── Lifecycle ────────────────────────────────────────────────

    async def start(self):
        if self.started:
            raise ValueError("STT already started")
        self.started = True
        logger.info("STT '%s' started (session: %s)", self.provider_name, self.session_id[:8])

    async def clear(self):
        """Clear pending audio/state."""
        pass

    async def close(self):
        self.closed = True
        logger.info("STT '%s' closed", self.provider_name)

    def get_stats(self) -> Dict:
        return {
            "provider": self.provider_name,
            "session_id": self.session_id[:8],
            "started": self.started,
            "closed": self.closed,
            "total_transcripts": len(self._transcripts),
            "total_audio_ms": round(self._total_audio_ms, 1),
            "turn_detection": self.turn_detection,
        }
