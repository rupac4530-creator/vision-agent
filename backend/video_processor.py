# backend/video_processor.py
"""
Video Processor Pipeline — SDK-Aligned Frame Processing

Ported from Vision-Agents SDK (agents-core/vision_agents/core/processors/)
Provides a composable pipeline for video frame processing with
chaining support (e.g., YOLO → Pose → LLM annotation).
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("video_processor")


@dataclass
class FrameContext:
    """Context object passed through the processor pipeline."""
    frame_data: bytes
    frame_id: int = 0
    width: int = 0
    height: int = 0
    timestamp: float = field(default_factory=time.time)
    source: str = "webcam"  # webcam, upload, screen_share

    # Accumulated results from processors
    detections: List[dict] = field(default_factory=list)
    poses: List[dict] = field(default_factory=list)
    annotations: List[dict] = field(default_factory=list)
    text_overlay: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_times_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def total_processing_ms(self) -> float:
        return sum(self.processing_times_ms.values())

    def add_detection(self, label: str, confidence: float, bbox: Tuple[int, int, int, int], **extra):
        self.detections.append({
            "label": label, "confidence": round(confidence, 3),
            "bbox": list(bbox), **extra,
        })

    def add_pose(self, person_id: int, keypoints: list, confidence: float = 0.0, **extra):
        self.poses.append({
            "person_id": person_id, "keypoints": keypoints,
            "confidence": round(confidence, 3), **extra,
        })

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "detections": self.detections,
            "poses": self.poses,
            "annotations": self.annotations,
            "text_overlay": self.text_overlay,
            "processing_ms": self.processing_times_ms,
            "total_ms": round(self.total_processing_ms, 1),
        }


class BaseProcessor(ABC):
    """
    Abstract base class for video frame processors.
    Mirrors the Vision-Agents SDK BaseProcessor pattern.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.is_initialized = False
        self.frames_processed = 0
        self.total_time_ms = 0.0

    async def initialize(self):
        """Called once before first frame. Override for model loading."""
        self.is_initialized = True

    @abstractmethod
    async def process_frame(self, ctx: FrameContext) -> FrameContext:
        """Process a single frame and return the enriched context.
        Must be implemented by subclasses."""
        ...

    async def cleanup(self):
        """Called on shutdown. Override for resource cleanup."""
        pass

    @property
    def avg_time_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_time_ms / self.frames_processed


class ProcessorPipeline:
    """
    Composable pipeline of frame processors.
    Processors run sequentially, each enriching the FrameContext.

    Usage:
        pipeline = ProcessorPipeline()
        pipeline.add(YOLOProcessor())
        pipeline.add(PoseProcessor())
        pipeline.add(LLMAnnotator())

        result = await pipeline.process(frame_data)
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.processors: List[BaseProcessor] = []
        self._initialized = False
        self._frame_counter = 0

    def add(self, processor: BaseProcessor) -> "ProcessorPipeline":
        """Add a processor to the pipeline. Returns self for chaining."""
        self.processors.append(processor)
        logger.info("Pipeline '%s': added %s", self.name, processor.name)
        return self

    async def initialize(self):
        """Initialize all processors in the pipeline."""
        for proc in self.processors:
            if not proc.is_initialized:
                await proc.initialize()
                logger.info("Initialized processor: %s", proc.name)
        self._initialized = True

    async def process(self, frame_data: bytes, source: str = "webcam",
                      width: int = 0, height: int = 0) -> FrameContext:
        """Process a frame through all processors in sequence."""
        if not self._initialized:
            await self.initialize()

        self._frame_counter += 1
        ctx = FrameContext(
            frame_data=frame_data,
            frame_id=self._frame_counter,
            width=width,
            height=height,
            source=source,
        )

        for proc in self.processors:
            start = time.monotonic()
            try:
                ctx = await proc.process_frame(ctx)
                elapsed = (time.monotonic() - start) * 1000
                ctx.processing_times_ms[proc.name] = round(elapsed, 1)
                proc.frames_processed += 1
                proc.total_time_ms += elapsed
            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                ctx.processing_times_ms[proc.name] = round(elapsed, 1)
                logger.error("Processor %s failed on frame %d: %s", proc.name, ctx.frame_id, exc)
                ctx.annotations.append({"type": "error", "processor": proc.name, "error": str(exc)})

        return ctx

    async def cleanup(self):
        """Cleanup all processors."""
        for proc in self.processors:
            await proc.cleanup()

    def get_stats(self) -> dict:
        return {
            "pipeline": self.name,
            "processors": [
                {"name": p.name, "frames": p.frames_processed, "avg_ms": round(p.avg_time_ms, 1)}
                for p in self.processors
            ],
            "total_frames": self._frame_counter,
        }


# ── Built-in Processors ──────────────────────────────────────────────

class MotionDetector(BaseProcessor):
    """
    Simple motion detection processor.
    Compares consecutive frames to detect significant changes.
    """

    def __init__(self, threshold: float = 0.15):
        super().__init__(name="MotionDetector")
        self.threshold = threshold
        self._prev_frame_hash: Optional[int] = None

    async def process_frame(self, ctx: FrameContext) -> FrameContext:
        frame_hash = hash(ctx.frame_data[:1024])  # Quick hash of first 1KB
        if self._prev_frame_hash is not None:
            # Very rough motion detection via hash comparison
            motion = frame_hash != self._prev_frame_hash
            ctx.metadata["motion_detected"] = motion
        else:
            ctx.metadata["motion_detected"] = True  # First frame always "has motion"

        self._prev_frame_hash = frame_hash
        return ctx


class FrameRateController(BaseProcessor):
    """
    Controls processing frame rate by skipping frames.
    Useful for reducing GPU/API load on high-frequency streams.
    """

    def __init__(self, target_fps: float = 2.0):
        super().__init__(name="FrameRateController")
        self.target_fps = target_fps
        self.min_interval_sec = 1.0 / target_fps
        self._last_process_time = 0.0

    async def process_frame(self, ctx: FrameContext) -> FrameContext:
        now = time.time()
        elapsed = now - self._last_process_time

        if elapsed < self.min_interval_sec:
            ctx.metadata["frame_skipped"] = True
        else:
            ctx.metadata["frame_skipped"] = False
            self._last_process_time = now

        return ctx


# ══════════════════════════════════════════════════════════════════════
# Default pipeline factory
# ══════════════════════════════════════════════════════════════════════

def create_default_pipeline() -> ProcessorPipeline:
    """Create the default processing pipeline."""
    pipeline = ProcessorPipeline(name="default")
    pipeline.add(FrameRateController(target_fps=2.0))
    pipeline.add(MotionDetector())
    return pipeline


logger.info("VideoProcessor module ready")
