# backend/observability.py
"""
Observability Module — SDK-Aligned Metrics & Provider Health

Ported from Vision-Agents SDK (agents-core/vision_agents/core/observability/)
Provides provider health tracking, rolling latency metrics, and
Prometheus-compatible text output for monitoring.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("observability")


# ── Provider Health Tracker ───────────────────────────────────────────

@dataclass
class ProviderStats:
    """Rolling statistics for a single LLM provider."""
    name: str
    total_calls: int = 0
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    is_healthy: bool = True
    last_success: float = 0.0
    last_error: float = 0.0
    last_error_msg: str = ""
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    disabled_until: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return self.success_count / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(len(sorted_lats) * 0.95)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lats = sorted(self.latencies_ms)
        idx = int(len(sorted_lats) * 0.99)
        return sorted_lats[min(idx, len(sorted_lats) - 1)]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": round(self.success_rate, 3),
            "is_healthy": self.is_healthy,
            "consecutive_errors": self.consecutive_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "last_success": self.last_success,
            "last_error": self.last_error,
            "last_error_msg": self.last_error_msg[:100] if self.last_error_msg else "",
        }


class ProviderHealthTracker:
    """
    Tracks health and performance of all LLM providers.
    Auto-disables providers after consecutive failures with exponential backoff.
    """

    MAX_CONSECUTIVE_ERRORS = 5
    BASE_BACKOFF_SEC = 10

    def __init__(self):
        self._providers: Dict[str, ProviderStats] = {}
        self._global_stats = {
            "total_requests": 0,
            "total_fallbacks": 0,
            "total_errors": 0,
        }

    def _ensure(self, name: str) -> ProviderStats:
        if name not in self._providers:
            self._providers[name] = ProviderStats(name=name)
        return self._providers[name]

    def record_success(self, provider_name: str, latency_ms: float):
        """Record a successful LLM call."""
        stats = self._ensure(provider_name)
        stats.total_calls += 1
        stats.success_count += 1
        stats.consecutive_errors = 0
        stats.is_healthy = True
        stats.last_success = time.time()
        stats.latencies_ms.append(latency_ms)
        stats.disabled_until = 0.0
        self._global_stats["total_requests"] += 1

    def record_error(self, provider_name: str, error_msg: str, latency_ms: float = 0):
        """Record a failed LLM call."""
        stats = self._ensure(provider_name)
        stats.total_calls += 1
        stats.error_count += 1
        stats.consecutive_errors += 1
        stats.last_error = time.time()
        stats.last_error_msg = error_msg
        if latency_ms > 0:
            stats.latencies_ms.append(latency_ms)

        self._global_stats["total_requests"] += 1
        self._global_stats["total_errors"] += 1

        # Auto-disable after too many consecutive errors
        if stats.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            backoff = self.BASE_BACKOFF_SEC * (2 ** (stats.consecutive_errors - self.MAX_CONSECUTIVE_ERRORS))
            backoff = min(backoff, 300)  # Max 5 minutes
            stats.disabled_until = time.time() + backoff
            stats.is_healthy = False
            logger.warning("Provider %s disabled for %.0fs (consecutive errors: %d)",
                           provider_name, backoff, stats.consecutive_errors)

    def record_fallback(self, from_provider: str, to_provider: str, reason: str = ""):
        """Record a provider fallback event."""
        self._global_stats["total_fallbacks"] += 1
        logger.info("Fallback: %s → %s (reason: %s)", from_provider, to_provider, reason)

    def is_available(self, provider_name: str) -> bool:
        """Check if a provider is available (not disabled)."""
        stats = self._providers.get(provider_name)
        if not stats:
            return True
        if stats.disabled_until > 0 and time.time() < stats.disabled_until:
            return False
        # Re-enable if backoff has expired
        if stats.disabled_until > 0 and time.time() >= stats.disabled_until:
            stats.disabled_until = 0.0
            stats.is_healthy = True
            logger.info("Provider %s re-enabled after backoff", provider_name)
        return True

    def get_all_stats(self) -> dict:
        """Get complete health stats for all providers."""
        return {
            "providers": {name: s.to_dict() for name, s in self._providers.items()},
            "global": self._global_stats,
        }

    def get_provider_stats(self, name: str) -> Optional[dict]:
        stats = self._providers.get(name)
        return stats.to_dict() if stats else None

    def to_prometheus(self) -> str:
        """Generate Prometheus-compatible metrics text."""
        lines = [
            "# HELP vision_agent_llm_requests_total Total LLM requests",
            "# TYPE vision_agent_llm_requests_total counter",
        ]
        for name, s in self._providers.items():
            lines.append(f'vision_agent_llm_requests_total{{provider="{name}",status="success"}} {s.success_count}')
            lines.append(f'vision_agent_llm_requests_total{{provider="{name}",status="error"}} {s.error_count}')

        lines.extend([
            "",
            "# HELP vision_agent_llm_latency_ms LLM response latency",
            "# TYPE vision_agent_llm_latency_ms gauge",
        ])
        for name, s in self._providers.items():
            lines.append(f'vision_agent_llm_latency_avg_ms{{provider="{name}"}} {s.avg_latency_ms:.1f}')
            lines.append(f'vision_agent_llm_latency_p95_ms{{provider="{name}"}} {s.p95_latency_ms:.1f}')
            lines.append(f'vision_agent_llm_latency_p99_ms{{provider="{name}"}} {s.p99_latency_ms:.1f}')

        lines.extend([
            "",
            "# HELP vision_agent_provider_healthy Provider health status",
            "# TYPE vision_agent_provider_healthy gauge",
        ])
        for name, s in self._providers.items():
            lines.append(f'vision_agent_provider_healthy{{provider="{name}"}} {1 if s.is_healthy else 0}')

        lines.extend([
            "",
            f"# HELP vision_agent_fallbacks_total Total provider fallbacks",
            f"# TYPE vision_agent_fallbacks_total counter",
            f"vision_agent_fallbacks_total {self._global_stats['total_fallbacks']}",
        ])

        return "\n".join(lines) + "\n"


# ── Global Metrics Counters ───────────────────────────────────────────

@dataclass
class PlatformMetrics:
    """Global platform metrics tracker."""
    start_time: float = field(default_factory=time.time)
    frames_processed: int = 0
    chunks_received: int = 0
    videos_uploaded: int = 0
    analyses_completed: int = 0
    notes_generated: int = 0
    questions_answered: int = 0
    pose_frames: int = 0
    security_frames: int = 0
    stt_calls: int = 0
    tts_calls: int = 0
    tool_calls: int = 0
    active_streams: int = 0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "frames_processed": self.frames_processed,
            "chunks_received": self.chunks_received,
            "videos_uploaded": self.videos_uploaded,
            "analyses_completed": self.analyses_completed,
            "notes_generated": self.notes_generated,
            "questions_answered": self.questions_answered,
            "pose_frames": self.pose_frames,
            "security_frames": self.security_frames,
            "stt_calls": self.stt_calls,
            "tts_calls": self.tts_calls,
            "tool_calls": self.tool_calls,
            "active_streams": self.active_streams,
        }

    def to_prometheus(self) -> str:
        lines = []
        for key, val in self.to_dict().items():
            prom_name = f"vision_agent_{key}"
            lines.append(f"# TYPE {prom_name} gauge")
            lines.append(f"{prom_name} {val}")
        return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════
# Global singletons
# ══════════════════════════════════════════════════════════════════════

health_tracker = ProviderHealthTracker()
platform_metrics = PlatformMetrics()

logger.info("Observability module ready")
