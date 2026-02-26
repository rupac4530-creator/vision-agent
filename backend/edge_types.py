# backend/edge_types.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Edge Types.

Types and data classes used across the Vision-Agents SDK for
participants, audio data, video frames, and stream metadata.
Adapted from agents-core/vision_agents/core/edge/types.py.
"""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class AudioFormat(str, Enum):
    """Audio encoding formats."""
    PCM_S16 = "pcm_s16"     # 16-bit signed integer PCM
    PCM_F32 = "pcm_f32"     # 32-bit float PCM
    MP3 = "mp3"
    OPUS = "opus"
    WAV = "wav"
    WEBM = "webm"


@dataclass
class Participant:
    """Represents a participant in a call/session."""
    id: str = ""
    name: str = ""
    role: str = "user"  # "user", "agent", "moderator"
    joined_at: float = field(default_factory=time.time)
    is_speaking: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "role": self.role,
            "is_speaking": self.is_speaking,
        }


@dataclass
class PcmData:
    """PCM audio data container.

    Adapted from getstream.video.rtc.PcmData — provides a portable
    audio container without WebRTC dependencies.
    """
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_S16
    participant: Optional[Participant] = None

    @classmethod
    def from_bytes(cls, data: bytes, sample_rate: int = 16000,
                   channels: int = 1, format: str = "pcm_s16") -> "PcmData":
        return cls(data=data, sample_rate=sample_rate,
                   channels=channels, format=AudioFormat(format))

    @property
    def duration_ms(self) -> float:
        """Duration of audio in milliseconds."""
        bytes_per_sample = 2 if self.format == AudioFormat.PCM_S16 else 4
        samples = len(self.data) / (bytes_per_sample * self.channels)
        return (samples / self.sample_rate) * 1000

    @property
    def frame_count(self) -> int:
        """Number of audio frames (samples per channel)."""
        bytes_per_sample = 2 if self.format == AudioFormat.PCM_S16 else 4
        return len(self.data) // (bytes_per_sample * self.channels)

    def to_mono(self) -> "PcmData":
        """Convert stereo to mono by averaging channels."""
        if self.channels == 1:
            return self
        # Average stereo channels
        bytes_per_sample = 2 if self.format == AudioFormat.PCM_S16 else 4
        mono_data = bytearray()
        for i in range(0, len(self.data), bytes_per_sample * self.channels):
            left = int.from_bytes(self.data[i:i+bytes_per_sample], 'little', signed=True)
            right = int.from_bytes(self.data[i+bytes_per_sample:i+bytes_per_sample*2], 'little', signed=True)
            avg = (left + right) // 2
            mono_data.extend(avg.to_bytes(bytes_per_sample, 'little', signed=True))
        return PcmData(data=bytes(mono_data), sample_rate=self.sample_rate,
                       channels=1, format=self.format, participant=self.participant)

    def to_dict(self) -> Dict:
        return {
            "data_len": len(self.data),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format.value,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class VideoFrame:
    """A single video frame."""
    data: bytes
    width: int = 0
    height: int = 0
    format: str = "rgb24"  # rgb24, bgr24, yuv420p, jpeg
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    participant: Optional[Participant] = None

    def to_dict(self) -> Dict:
        return {
            "width": self.width, "height": self.height,
            "format": self.format, "frame_number": self.frame_number,
            "data_len": len(self.data),
        }


@dataclass
class StreamMetadata:
    """Metadata about a media stream."""
    stream_id: str = ""
    type: str = "video"  # "video", "audio", "screen_share"
    codec: str = ""
    bitrate: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0
    sample_rate: int = 0
    channels: int = 0
    participant: Optional[Participant] = None
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "stream_id": self.stream_id, "type": self.type,
            "codec": self.codec, "bitrate": self.bitrate,
        }
