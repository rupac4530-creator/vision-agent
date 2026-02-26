# backend/vad.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned VAD (Voice Activity Detection).

Provides energy-based voice activity detection for real-time audio streams.
Used to determine when a user is speaking vs silent, enabling:
- Efficient STT (only process speech segments)
- Turn detection (detect end of utterance)
- Noise filtering

Usage:
    vad = EnergyVAD(threshold=0.02)
    is_speech = vad.process(audio_chunk)
"""

import time
import logging
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("vad")


@dataclass
class VADResult:
    """Result of VAD processing."""
    is_speech: bool = False
    energy: float = 0.0
    threshold: float = 0.02
    duration_ms: float = 0
    timestamp: float = field(default_factory=time.time)


class EnergyVAD:
    """Energy-based Voice Activity Detection.

    Simple but effective VAD that compares RMS energy of audio frames
    against a threshold. Includes:
    - Adaptive threshold (optional)
    - Hangover smoothing (prevents rapid on/off)
    - Speech/silence duration tracking
    """

    def __init__(self, threshold: float = 0.02,
                 hangover_frames: int = 5,
                 adaptive: bool = False,
                 sample_rate: int = 16000):
        self._threshold = threshold
        self._hangover_frames = hangover_frames
        self._adaptive = adaptive
        self._sample_rate = sample_rate

        # State
        self._is_speech = False
        self._hangover_count = 0
        self._frame_count = 0
        self._speech_frames = 0
        self._noise_floor = threshold * 0.5

        # History for adaptive threshold
        self._energy_history: List[float] = []
        self._max_history = 100

    def process(self, audio_data: bytes) -> VADResult:
        """Process an audio frame (16-bit PCM) and detect speech.

        Args:
            audio_data: Raw 16-bit PCM bytes

        Returns:
            VADResult with is_speech flag and energy level
        """
        self._frame_count += 1
        energy = self._rms_energy(audio_data)

        # Adaptive threshold
        if self._adaptive:
            self._energy_history.append(energy)
            if len(self._energy_history) > self._max_history:
                self._energy_history = self._energy_history[-self._max_history:]
            # Update noise floor (use bottom 20% of energy values)
            sorted_e = sorted(self._energy_history)
            noise_idx = max(1, len(sorted_e) // 5)
            self._noise_floor = sum(sorted_e[:noise_idx]) / noise_idx
            effective_threshold = max(self._threshold, self._noise_floor * 3)
        else:
            effective_threshold = self._threshold

        # Speech detection with hangover
        if energy >= effective_threshold:
            self._is_speech = True
            self._hangover_count = self._hangover_frames
            self._speech_frames += 1
        else:
            if self._hangover_count > 0:
                self._hangover_count -= 1
            else:
                self._is_speech = False

        return VADResult(
            is_speech=self._is_speech,
            energy=energy,
            threshold=effective_threshold,
        )

    def _rms_energy(self, audio_data: bytes) -> float:
        """Calculate RMS energy of 16-bit PCM audio."""
        if len(audio_data) < 2:
            return 0.0

        # Convert bytes to 16-bit samples
        samples = []
        for i in range(0, len(audio_data) - 1, 2):
            sample = int.from_bytes(audio_data[i:i+2], byteorder='little', signed=True)
            samples.append(sample / 32768.0)  # Normalize to [-1, 1]

        if not samples:
            return 0.0

        # RMS
        rms = math.sqrt(sum(s * s for s in samples) / len(samples))
        return rms

    def reset(self):
        """Reset VAD state."""
        self._is_speech = False
        self._hangover_count = 0
        self._frame_count = 0
        self._speech_frames = 0
        self._energy_history.clear()

    @property
    def is_speech(self) -> bool:
        return self._is_speech

    @property
    def speech_ratio(self) -> float:
        """Ratio of speech frames to total frames."""
        return self._speech_frames / max(1, self._frame_count)

    def get_stats(self) -> Dict:
        return {
            "is_speech": self._is_speech,
            "threshold": self._threshold,
            "adaptive": self._adaptive,
            "noise_floor": round(self._noise_floor, 4),
            "frame_count": self._frame_count,
            "speech_frames": self._speech_frames,
            "speech_ratio": round(self.speech_ratio, 3),
        }
