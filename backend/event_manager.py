# backend/event_manager.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Event Manager.

Low-level event registration and dispatching system used by SDK plugins.
Complements our EventBus with typed event class registration,
module-level auto-discovery, and synchronous in-process dispatch.

Usage:
    em = EventManager()
    em.register(MyEvent)
    em.on(MyEvent, handler)
    em.send(MyEvent(data="hello"))
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type
from dataclasses import dataclass, field

logger = logging.getLogger("event_manager")


@dataclass
class BaseEvent:
    """Base class for all typed events."""
    plugin_name: str = ""
    session_id: str = ""

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ToolStartEvent(BaseEvent):
    """Emitted when a tool call starts."""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None


@dataclass
class ToolEndEvent(BaseEvent):
    """Emitted when a tool call completes."""
    tool_name: str = ""
    success: bool = True
    result: Optional[Any] = None
    error: Optional[str] = None
    tool_call_id: Optional[str] = None
    execution_time_ms: float = 0


@dataclass
class AgentStateEvent(BaseEvent):
    """Emitted when agent state changes."""
    state: str = ""  # "starting", "running", "idle", "processing", "stopping"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRequestEvent(BaseEvent):
    """Emitted when an LLM request is made."""
    provider: str = ""
    model: str = ""
    message_count: int = 0
    has_tools: bool = False


@dataclass
class LLMResponseEvent(BaseEvent):
    """Emitted when an LLM response is received."""
    provider: str = ""
    model: str = ""
    text_length: int = 0
    tool_call_count: int = 0
    latency_ms: float = 0
    tokens_used: int = 0


class EventManager:
    """Typed event registration and dispatch system.

    SDK-aligned pattern from agents-core/vision_agents/core/events/manager.py.
    Supports:
    - Event class registration
    - Module-level auto-discovery of event classes
    - Typed handler subscription
    - Synchronous dispatch
    """

    def __init__(self):
        self._registered_events: Dict[str, Type] = {}
        self._handlers: Dict[str, List[Callable]] = {}
        self._event_count: Dict[str, int] = {}

    def register(self, event_class: Type):
        """Register an event class."""
        name = event_class.__name__
        self._registered_events[name] = event_class
        if name not in self._handlers:
            self._handlers[name] = []

    def register_events_from_module(self, module, ignore_not_compatible: bool = False):
        """Auto-discover and register all event classes from a module."""
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and
                    issubclass(obj, BaseEvent) and
                    obj is not BaseEvent):
                try:
                    self.register(obj)
                except Exception as e:
                    if not ignore_not_compatible:
                        raise
                    logger.debug("Skipping event %s: %s", name, e)

    def on(self, event_class: Type, handler: Callable):
        """Subscribe a handler to an event type."""
        name = event_class.__name__
        if name not in self._handlers:
            self._handlers[name] = []
        self._handlers[name].append(handler)

    def send(self, event: Any):
        """Dispatch an event to all registered handlers."""
        name = type(event).__name__
        self._event_count[name] = self._event_count.get(name, 0) + 1

        handlers = self._handlers.get(name, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Event handler error for %s: %s", name, e)

    def list_events(self) -> List[str]:
        """List all registered event types."""
        return list(self._registered_events.keys())

    def get_stats(self) -> Dict:
        return {
            "registered_events": len(self._registered_events),
            "event_types": self.list_events(),
            "total_dispatched": sum(self._event_count.values()),
            "by_type": self._event_count,
        }
