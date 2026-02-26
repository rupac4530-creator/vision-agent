# backend/transcript_buffer.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Transcript Buffer.

Accumulates partial STT (Speech-to-Text) results into complete sentences.
Handles interim results, final results, and sentence boundary detection.

Usage:
    buf = TranscriptBuffer()
    buf.push_interim("Hello how")
    buf.push_final("Hello, how are you?")
    sentences = buf.get_complete_sentences()
"""

import re
import time
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("transcript_buffer")

# Sentence boundary patterns
_SENTENCE_END = re.compile(r'[.!?]+\s*$')
_WORD_BOUNDARY = re.compile(r'\s+')


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    text: str
    is_final: bool
    timestamp: float = field(default_factory=time.time)
    speaker_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "is_final": self.is_final,
            "timestamp": self.timestamp,
            "speaker_id": self.speaker_id,
            "confidence": self.confidence,
        }


class TranscriptBuffer:
    """Accumulates partial STT results into complete sentences.

    Handles:
    - Interim (partial) results that get replaced
    - Final results that are committed
    - Sentence boundary detection
    - Speaker diarization (optional)
    - Silence-based sentence completion
    """

    def __init__(self, silence_timeout: float = 2.0, max_buffer_size: int = 50):
        self._segments: List[TranscriptSegment] = []
        self._interim: Optional[TranscriptSegment] = None
        self._complete_sentences: List[str] = []
        self._silence_timeout = silence_timeout
        self._max_buffer_size = max_buffer_size
        self._last_activity = time.time()

    def push_interim(self, text: str, speaker_id: Optional[str] = None,
                     confidence: float = 0.5) -> None:
        """Push an interim (partial) transcription result.
        Replaces any previous interim result.
        """
        self._interim = TranscriptSegment(
            text=text.strip(),
            is_final=False,
            speaker_id=speaker_id,
            confidence=confidence,
        )
        self._last_activity = time.time()

    def push_final(self, text: str, speaker_id: Optional[str] = None,
                   confidence: float = 1.0) -> None:
        """Push a final (committed) transcription result.
        Clears interim and adds to committed segments.
        """
        text = text.strip()
        if not text:
            return

        self._interim = None
        segment = TranscriptSegment(
            text=text,
            is_final=True,
            speaker_id=speaker_id,
            confidence=confidence,
        )
        self._segments.append(segment)
        self._last_activity = time.time()

        # Trim old segments
        if len(self._segments) > self._max_buffer_size:
            self._segments = self._segments[-self._max_buffer_size:]

        # Check for sentence boundaries
        self._extract_sentences()

    def _extract_sentences(self) -> None:
        """Extract complete sentences from committed segments."""
        full_text = " ".join(s.text for s in self._segments if s.is_final)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        for s in sentences:
            s = s.strip()
            if s and _SENTENCE_END.search(s):
                if s not in self._complete_sentences:
                    self._complete_sentences.append(s)

    def get_current_text(self) -> str:
        """Get the current full text including any interim result."""
        parts = [s.text for s in self._segments if s.is_final]
        if self._interim:
            parts.append(f"({self._interim.text})")
        return " ".join(parts)

    def get_complete_sentences(self) -> List[str]:
        """Get all complete sentences detected so far."""
        return list(self._complete_sentences)

    def get_latest_sentence(self) -> Optional[str]:
        """Get the most recent complete sentence."""
        return self._complete_sentences[-1] if self._complete_sentences else None

    def is_silence(self) -> bool:
        """Check if there's been silence longer than the timeout."""
        return (time.time() - self._last_activity) > self._silence_timeout

    def clear(self) -> None:
        """Clear all buffered content."""
        self._segments.clear()
        self._interim = None
        self._complete_sentences.clear()
        self._last_activity = time.time()

    def get_stats(self) -> Dict:
        """Get buffer statistics."""
        return {
            "total_segments": len(self._segments),
            "final_segments": sum(1 for s in self._segments if s.is_final),
            "has_interim": self._interim is not None,
            "complete_sentences": len(self._complete_sentences),
            "is_silence": self.is_silence(),
            "seconds_since_activity": round(time.time() - self._last_activity, 1),
        }
