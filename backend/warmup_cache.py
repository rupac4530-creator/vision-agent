# backend/warmup_cache.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Warmup Cache.

Provides lazy model loading with caching so expensive resources (YOLO, pose models)
are loaded once and shared across requests.

Usage:
    cache = WarmupCache()
    await cache.warmup(my_yolo_processor)  # loads model once, caches it
"""

import abc
import asyncio
import time
import logging
from typing import Any, Dict, Generic, Optional, Type, TypeVar

logger = logging.getLogger("warmup_cache")

T = TypeVar("T")


class WarmupCache:
    """Cache for expensive resources loaded by Warmable objects.

    Key = the class of the Warmable, so all instances of the same class share
    the same loaded resource (e.g., YOLO model loaded once).
    """

    def __init__(self):
        self._cache: Dict[Type, Any] = {}
        self._locks: Dict[Type, asyncio.Lock] = {}
        self._load_times: Dict[str, float] = {}  # Track load durations

    async def warmup(self, warmable: "Warmable") -> None:
        """Load resource if not cached, then set it on the warmable."""
        warmable_cls = type(warmable)

        # Already cached — just set and return
        resource = self._cache.get(warmable_cls)
        if resource is not None:
            warmable.on_warmed_up(resource)
            return

        # Protect against concurrent loading
        lock = self._locks.setdefault(warmable_cls, asyncio.Lock())
        async with lock:
            resource = self._cache.get(warmable_cls)
            if resource is None:
                t0 = time.time()
                logger.info("Warming up %s...", warmable_cls.__name__)
                resource = await warmable.on_warmup()
                duration = time.time() - t0
                self._cache[warmable_cls] = resource
                self._load_times[warmable_cls.__name__] = duration
                logger.info("Warmed up %s in %.2fs", warmable_cls.__name__, duration)

            warmable.on_warmed_up(resource)

    def is_warmed(self, cls: Type) -> bool:
        """Check if a class has been warmed up."""
        return cls in self._cache

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_types": list(self._load_times.keys()),
            "load_times": self._load_times,
            "total_cached": len(self._cache),
        }

    def clear(self):
        """Clear all cached resources."""
        self._cache.clear()
        self._locks.clear()
        self._load_times.clear()


class Warmable(abc.ABC, Generic[T]):
    """Base class for components needing expensive resource pre-loading.

    Subclasses implement:
    - on_warmup() -> T:  Load the resource and return it
    - on_warmed_up(resource: T):  Store the resource on the instance

    Example:
        class YOLOWarmable(Warmable[Any]):
            def __init__(self):
                self._model = None

            async def on_warmup(self) -> Any:
                from ultralytics import YOLO
                return YOLO("yolov8n.pt")

            def on_warmed_up(self, resource: Any) -> None:
                self._model = resource
    """

    @abc.abstractmethod
    async def on_warmup(self) -> T:
        """Load the resource. Called once during startup."""
        ...

    @abc.abstractmethod
    def on_warmed_up(self, resource: T) -> None:
        """Store the loaded resource. Called each time an agent starts."""
        ...

    async def warmup(self, cache: Optional[WarmupCache] = None) -> None:
        """Perform warmup using provided cache (or a temporary one)."""
        cache = cache or WarmupCache()
        await cache.warmup(warmable=self)


# ── Global singleton cache ─────────────────────────────────────────────
warmup_cache = WarmupCache()
