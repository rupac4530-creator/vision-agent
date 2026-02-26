# backend/http_transport.py
# adapted from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
HTTP Transport Adapter.

Maps the upstream SDK's WebRTC-based processor API to our HTTP chunking
and stream_chunk API. This adapter allows Vision-Agents SDK processors
to work without WebRTC dependencies.

Pattern: adapter bridges between SDK's VideoProcessor.process_video()
interface and our HTTP-based frame streaming.
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("http_transport")


@dataclass
class ChunkResult:
    """Result from processing a streamed chunk."""
    chunk_id: int
    processed: bool = True
    detections: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0


class HTTPTransport:
    """Adapter that bridges SDK processor patterns to HTTP streaming.

    Instead of WebRTC tracks, this accepts HTTP-uploaded video/audio chunks
    and routes them through the same processor pipeline.
    """

    def __init__(self):
        self._processors: List[Callable] = []
        self._chunk_counter = 0
        self._total_bytes = 0
        self._results: List[ChunkResult] = []
        self._max_results = 100

    def add_processor(self, processor: Callable):
        """Add a processing function to the pipeline."""
        self._processors.append(processor)
        logger.info("Added processor: %s", getattr(processor, '__name__', str(processor)))

    async def process_chunk(self, chunk_data: bytes,
                            content_type: str = "video/webm",
                            metadata: Optional[Dict] = None) -> ChunkResult:
        """Process an HTTP-uploaded chunk through all processors.

        This is the HTTP equivalent of SDK's process_video() — it takes
        raw bytes from an HTTP upload and routes through the pipeline.
        """
        self._chunk_counter += 1
        self._total_bytes += len(chunk_data)
        t0 = time.time()

        result = ChunkResult(
            chunk_id=self._chunk_counter,
            metadata=metadata or {},
        )

        for processor in self._processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    proc_result = await processor(chunk_data, content_type, metadata)
                else:
                    proc_result = processor(chunk_data, content_type, metadata)

                if isinstance(proc_result, dict):
                    if "detections" in proc_result:
                        result.detections.extend(proc_result["detections"])
                    result.metadata.update(proc_result)

            except Exception as e:
                logger.error("Processor error: %s", e)
                result.metadata["error"] = str(e)

        result.latency_ms = (time.time() - t0) * 1000
        result.processed = True

        # Store result
        self._results.append(result)
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results:]

        return result

    async def process_frame_base64(self, frame_b64: str,
                                   metadata: Optional[Dict] = None) -> ChunkResult:
        """Process a base64-encoded frame (used by REST endpoints)."""
        import base64
        chunk_data = base64.b64decode(frame_b64)
        return await self.process_chunk(chunk_data, "image/jpeg", metadata)

    def get_latest_results(self, n: int = 10) -> List[Dict]:
        """Get the N most recent processing results."""
        return [
            {
                "chunk_id": r.chunk_id,
                "detections": r.detections[:5],
                "latency_ms": round(r.latency_ms, 1),
                "processed": r.processed,
            }
            for r in self._results[-n:]
        ]

    def get_stats(self) -> Dict:
        return {
            "total_chunks": self._chunk_counter,
            "total_bytes": self._total_bytes,
            "total_bytes_mb": round(self._total_bytes / (1024 * 1024), 2),
            "processors": len(self._processors),
            "avg_latency_ms": round(
                sum(r.latency_ms for r in self._results) / max(1, len(self._results)), 1
            ) if self._results else 0,
        }


# ── Singleton transport instance ──────────────────────────────────────
http_transport = HTTPTransport()
