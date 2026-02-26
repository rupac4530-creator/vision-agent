# backend/turn_detection.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Turn Detection.

Detects when a speaker has finished talking (end-of-turn) using
silence duration analysis. Used for voice-based agent interactions.

Usage:
    detector = SilenceTurnDetector(silence_threshold_ms=800)
    detector.on_audio_level(0.02)  # low level → possible silence
    if detector.should_respond():
        # User has stopped speaking, agent can respond
"""

import time
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import abc

logger = logging.getLogger("turn_detection")


@dataclass
class TurnEvent:
    """Represents a detected turn event."""
    type: str  # "start_speaking", "end_speaking", "silence_detected"
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0
    confidence: float = 1.0


class TurnDetector(abc.ABC):
    """Abstract base class for turn detection implementations."""

    @abc.abstractmethod
    def on_audio_level(self, level: float) -> Optional[TurnEvent]:
        """Process an audio level sample. Returns TurnEvent if turn detected."""
        ...

    @abc.abstractmethod
    def should_respond(self) -> bool:
        """Check if the agent should respond (user finished speaking)."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the detector state."""
        ...


class SilenceTurnDetector(TurnDetector):
    """Silence-based turn detection.

    Monitors audio levels and detects when a speaker has been silent
    for longer than a threshold, indicating they've finished speaking.
    """

    def __init__(self, silence_threshold_ms: float = 800,
                 speech_threshold: float = 0.05,
                 min_speech_duration_ms: float = 200):
        self._silence_threshold_ms = silence_threshold_ms
        self._speech_threshold = speech_threshold
        self._min_speech_duration_ms = min_speech_duration_ms

        self._is_speaking = False
        self._speech_start: Optional[float] = None
        self._silence_start: Optional[float] = None
        self._last_level = 0.0
        self._events: List[TurnEvent] = []
        self._turn_complete = False

    def on_audio_level(self, level: float) -> Optional[TurnEvent]:
        """Process an audio level sample (0.0 to 1.0)."""
        self._last_level = level
        now = time.time()
        event = None

        if level >= self._speech_threshold:
            # Speech detected
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                self._turn_complete = False
                event = TurnEvent(type="start_speaking", timestamp=now)
                self._events.append(event)

            self._silence_start = None

        else:
            # Silence detected
            if self._is_speaking:
                if self._silence_start is None:
                    self._silence_start = now
                else:
                    silence_ms = (now - self._silence_start) * 1000
                    if silence_ms >= self._silence_threshold_ms:
                        # End of turn detected
                        speech_duration = (now - (self._speech_start or now)) * 1000
                        if speech_duration >= self._min_speech_duration_ms:
                            self._is_speaking = False
                            self._turn_complete = True
                            event = TurnEvent(
                                type="end_speaking",
                                timestamp=now,
                                duration_ms=speech_duration,
                            )
                            self._events.append(event)

        return event

    def should_respond(self) -> bool:
        """Check if user has finished their turn."""
        if self._turn_complete:
            self._turn_complete = False  # Reset after checking
            return True
        return False

    def reset(self) -> None:
        """Reset all state."""
        self._is_speaking = False
        self._speech_start = None
        self._silence_start = None
        self._turn_complete = False
        self._events.clear()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def get_stats(self) -> Dict:
        return {
            "is_speaking": self._is_speaking,
            "last_level": round(self._last_level, 4),
            "silence_threshold_ms": self._silence_threshold_ms,
            "speech_threshold": self._speech_threshold,
            "total_events": len(self._events),
            "recent_events": [
                {"type": e.type, "duration_ms": round(e.duration_ms, 1)}
                for e in self._events[-5:]
            ],
        }
