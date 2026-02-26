# backend/profiling.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Performance Profiler.

Provides timing decorators and per-request latency tracking for
agents, LLM calls, video processing, and tool execution.

Usage:
    @profiler.track("llm_call")
    async def call_llm(...): ...

    profiler.get_stats()  # → timing histograms
"""

import time
import asyncio
import functools
import logging
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger("profiling")


class TimingRecord:
    """A single timing measurement."""
    __slots__ = ("name", "start_time", "duration", "metadata")

    def __init__(self, name: str, duration: float, metadata: Optional[Dict] = None):
        self.name = name
        self.start_time = time.time()
        self.duration = duration
        self.metadata = metadata or {}


class Profiler:
    """Request-level performance profiler with histogram tracking.

    Tracks timing for categorized operations (llm_call, video_process,
    tool_exec, etc.) and provides aggregate statistics.
    """

    def __init__(self, max_history: int = 1000):
        self._records: Dict[str, List[TimingRecord]] = defaultdict(list)
        self._max_history = max_history
        self._active_timers: Dict[str, float] = {}

    def start_timer(self, name: str) -> str:
        """Start a named timer. Returns timer ID."""
        timer_id = f"{name}_{time.time()}"
        self._active_timers[timer_id] = time.time()
        return timer_id

    def stop_timer(self, timer_id: str, metadata: Optional[Dict] = None) -> float:
        """Stop a timer and record the duration. Returns duration in seconds."""
        start = self._active_timers.pop(timer_id, None)
        if start is None:
            return 0.0
        duration = time.time() - start
        name = timer_id.rsplit("_", 1)[0]
        self._record(name, duration, metadata)
        return duration

    def record(self, name: str, duration: float, metadata: Optional[Dict] = None):
        """Manually record a timing measurement."""
        self._record(name, duration, metadata)

    def _record(self, name: str, duration: float, metadata: Optional[Dict] = None):
        """Internal: add a timing record."""
        records = self._records[name]
        records.append(TimingRecord(name, duration, metadata))
        # Trim to max history
        if len(records) > self._max_history:
            self._records[name] = records[-self._max_history:]

    def track(self, name: str):
        """Decorator to automatically time an async or sync function."""
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    t0 = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        self._record(name, time.time() - t0, {"status": "success"})
                        return result
                    except Exception as e:
                        self._record(name, time.time() - t0, {"status": "error", "error": str(e)})
                        raise
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    t0 = time.time()
                    try:
                        result = func(*args, **kwargs)
                        self._record(name, time.time() - t0, {"status": "success"})
                        return result
                    except Exception as e:
                        self._record(name, time.time() - t0, {"status": "error", "error": str(e)})
                        raise
                return sync_wrapper
        return decorator

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregate statistics for all or a specific operation."""
        if name:
            return self._compute_stats(name, self._records.get(name, []))

        return {
            category: self._compute_stats(category, records)
            for category, records in self._records.items()
        }

    def _compute_stats(self, name: str, records: List[TimingRecord]) -> Dict:
        """Compute min/max/avg/p50/p95/p99 for a set of timing records."""
        if not records:
            return {"name": name, "count": 0}

        durations = sorted(r.duration for r in records)
        n = len(durations)

        return {
            "name": name,
            "count": n,
            "min_ms": round(durations[0] * 1000, 2),
            "max_ms": round(durations[-1] * 1000, 2),
            "avg_ms": round(sum(durations) / n * 1000, 2),
            "p50_ms": round(durations[n // 2] * 1000, 2),
            "p95_ms": round(durations[int(n * 0.95)] * 1000, 2) if n >= 20 else None,
            "p99_ms": round(durations[int(n * 0.99)] * 1000, 2) if n >= 100 else None,
            "error_rate": round(
                sum(1 for r in records if r.metadata.get("status") == "error") / n, 3
            ),
        }

    def clear(self):
        """Clear all profiling data."""
        self._records.clear()
        self._active_timers.clear()


# ── Singleton instance ─────────────────────────────────────────────────
profiler = Profiler()
