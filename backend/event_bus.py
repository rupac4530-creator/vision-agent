# backend/event_bus.py
"""
Event Bus — SDK-Aligned Typed Event System

Ported from Vision-Agents SDK (agents-core/vision_agents/core/events/)
Provides a typed async event system with subscribe/emit pattern,
event history, and SSE (Server-Sent Events) streaming support.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger("event_bus")


# ── Event Types ───────────────────────────────────────────────────────

class EventType(str, Enum):
    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_STOPPED = "agent.stopped"
    AGENT_ERROR = "agent.error"

    # LLM events
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_FALLBACK = "llm.fallback"
    LLM_ERROR = "llm.error"
    LLM_STREAM_TOKEN = "llm.stream_token"

    # Vision events
    DETECTION_RESULT = "vision.detection"
    POSE_UPDATE = "vision.pose"
    FRAME_PROCESSED = "vision.frame"

    # Streaming events
    STREAM_STARTED = "stream.started"
    STREAM_CHUNK = "stream.chunk"
    STREAM_STOPPED = "stream.stopped"

    # Tool/Function events
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    # Analysis events
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETE = "analysis.complete"
    NOTES_GENERATED = "notes.generated"

    # Coach events
    REP_COUNTED = "coach.rep"
    MILESTONE = "coach.milestone"
    CORRECTION = "coach.correction"

    # Security events
    SECURITY_ALERT = "security.alert"
    PERSON_DETECTED = "security.person"

    # Generic
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Event:
    """A typed event with payload and metadata."""
    type: EventType
    data: dict = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.type.value}-{int(self.timestamp * 1000)}"

    def to_dict(self) -> dict:
        return {
            "id": self.event_id,
            "type": self.type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        return f"id: {self.event_id}\nevent: {self.type.value}\ndata: {json.dumps(self.data)}\n\n"


# ── Event Bus ─────────────────────────────────────────────────────────

class EventBus:
    """
    Async event bus with typed events, history, and SSE support.
    Mirrors the Vision-Agents SDK EventManager pattern.
    """

    def __init__(self, max_history: int = 500):
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._global_handlers: List[Callable] = []
        self._history: deque = deque(maxlen=max_history)
        self._sse_queues: List[asyncio.Queue] = []
        self._stats: Dict[str, int] = {}
        logger.info("EventBus initialized (max_history=%d)", max_history)

    # ── Subscribe ─────────────────────────────────────────────────────

    def on(self, event_type: EventType) -> Callable:
        """Decorator to subscribe a handler to an event type."""
        def decorator(fn: Callable) -> Callable:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(fn)
            logger.debug("Handler registered for %s: %s", event_type.value, fn.__name__)
            return fn
        return decorator

    def subscribe(self, event_type: EventType, handler: Callable):
        """Imperative subscription."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def on_all(self, handler: Callable):
        """Subscribe to ALL events."""
        self._global_handlers.append(handler)

    # ── Emit ──────────────────────────────────────────────────────────

    async def emit(self, event: Event):
        """Emit an event to all subscribed handlers."""
        self._history.append(event)
        self._stats[event.type.value] = self._stats.get(event.type.value, 0) + 1

        # Push to SSE queues
        for q in self._sse_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

        # Execute type-specific handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as exc:
                logger.error("Event handler error for %s: %s", event.type.value, exc)

        # Execute global handlers
        for handler in self._global_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as exc:
                logger.error("Global handler error: %s", exc)

    def emit_sync(self, event: Event):
        """Synchronous emit (for use outside async context)."""
        self._history.append(event)
        self._stats[event.type.value] = self._stats.get(event.type.value, 0) + 1

        for q in self._sse_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # ── Convenience emit methods ──────────────────────────────────────

    async def emit_info(self, message: str, source: str = "system", **extra):
        await self.emit(Event(type=EventType.INFO, data={"message": message, **extra}, source=source))

    async def emit_llm_response(self, provider: str, model: str, latency_ms: float, tokens: int = 0, **extra):
        await self.emit(Event(
            type=EventType.LLM_RESPONSE,
            data={"provider": provider, "model": model, "latency_ms": latency_ms, "tokens": tokens, **extra},
            source=provider,
        ))

    async def emit_detection(self, detections: list, frame_id: int = 0, **extra):
        await self.emit(Event(
            type=EventType.DETECTION_RESULT,
            data={"detections": detections, "frame_id": frame_id, "count": len(detections), **extra},
            source="yolo",
        ))

    async def emit_tool_call(self, name: str, args: dict, result: Any, latency_ms: float):
        await self.emit(Event(
            type=EventType.TOOL_RESULT,
            data={"function": name, "arguments": args, "result": result, "latency_ms": latency_ms},
            source="function_registry",
        ))

    # ── SSE Streaming ─────────────────────────────────────────────────

    async def sse_stream(self, event_types: Optional[List[EventType]] = None):
        """
        Async generator for Server-Sent Events streaming.
        Yields SSE-formatted strings for real-time event consumption.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._sse_queues.append(queue)
        try:
            while True:
                event = await queue.get()
                if event_types and event.type not in event_types:
                    continue
                yield event.to_sse()
        finally:
            self._sse_queues.remove(queue)

    # ── History & Stats ───────────────────────────────────────────────

    def get_history(self, limit: int = 50, event_type: Optional[EventType] = None) -> List[dict]:
        """Get recent event history."""
        events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return [e.to_dict() for e in events[-limit:]]

    def get_stats(self) -> dict:
        """Get event emission statistics."""
        return {
            "total_events": sum(self._stats.values()),
            "by_type": dict(self._stats),
            "history_size": len(self._history),
            "sse_clients": len(self._sse_queues),
        }

    def clear_history(self):
        self._history.clear()


# ══════════════════════════════════════════════════════════════════════
# Global event bus singleton
# ══════════════════════════════════════════════════════════════════════

event_bus = EventBus()
logger.info("Global EventBus ready")
