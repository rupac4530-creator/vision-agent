# backend/llm_base.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned LLM Base Class.

Abstract base for all LLM provider implementations. Provides:
- Tool-calling loop with deduplication and timeout
- Instruction management
- Conversation integration
- Event emission for tool start/end
- Before/after response listeners

Subclasses: GeminiLLM, OpenAILLM, AnthropicLLM, etc.

Usage:
    class MyLLM(LLMBase):
        async def generate(self, messages, tools=None) -> LLMResponse:
            # call your provider here
            ...
"""

import abc
import asyncio
import json
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("llm_base")


@dataclass
class LLMResponseEvent:
    """Event emitted after an LLM response."""
    text: str = ""
    exception: Optional[Exception] = None
    raw: Optional[Any] = None
    latency_ms: float = 0


@dataclass
class ToolCallEvent:
    """Event emitted when a tool is called or completes."""
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None
    success: bool = True
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0


class LLMBase(abc.ABC):
    """Abstract base class for LLM providers.

    SDK-aligned pattern from Vision-Agents agents-core/vision_agents/core/llm/llm.py.
    Provides tool-calling loop, instruction management, and event hooks.
    """

    def __init__(self, provider_name: str = "base"):
        self.provider_name = provider_name
        self._instructions: str = ""
        self._conversation_history: List[Dict] = []
        self._tool_registry: Dict[str, Callable] = {}
        self._tool_timeout_s: float = 30.0
        self._max_tool_rounds: int = 5

        # Event listeners
        self._before_listeners: List[Callable] = []
        self._after_listeners: List[Callable] = []
        self._tool_listeners: List[Callable] = []

    # ── Abstract methods (subclasses must implement) ──────────────

    @abc.abstractmethod
    async def generate(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict:
        """Generate a response from the LLM.

        Returns:
            Dict with keys: text, tool_calls (list), model, tokens, raw
        """
        ...

    @abc.abstractmethod
    def get_provider_tool_format(self, tool_schemas: List[Dict]) -> List[Dict]:
        """Convert generic tool schemas to provider-specific format."""
        ...

    # ── Instructions ─────────────────────────────────────────────

    def set_instructions(self, instructions) -> None:
        """Set the system instructions for the LLM."""
        if isinstance(instructions, str):
            self._instructions = instructions
        elif hasattr(instructions, 'full_reference'):
            self._instructions = instructions.full_reference
        else:
            self._instructions = str(instructions)

    @property
    def instructions(self) -> str:
        return self._instructions

    # ── Tool Registration ────────────────────────────────────────

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the LLM can call."""
        self._tool_registry[name] = func

    def register_function(self, name: Optional[str] = None,
                          description: Optional[str] = None) -> Callable:
        """Decorator to register a function as a tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self._tool_registry[tool_name] = func
            return func
        return decorator

    def get_available_tools(self) -> List[str]:
        return list(self._tool_registry.keys())

    # ── Event Listeners ──────────────────────────────────────────

    def on_before_response(self, listener: Callable):
        self._before_listeners.append(listener)

    def on_after_response(self, listener: Callable):
        self._after_listeners.append(listener)

    def on_tool_event(self, listener: Callable):
        self._tool_listeners.append(listener)

    def _emit_tool_event(self, event: ToolCallEvent):
        for listener in self._tool_listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error("Tool event listener error: %s", e)

    # ── Tool Calling Loop (core SDK pattern) ─────────────────────

    async def run_with_tools(self, user_input: str,
                             context: Optional[Dict] = None) -> LLMResponseEvent:
        """Run the full LLM loop with automatic tool calling.

        This is the core SDK pattern:
        1. Build messages from instructions + history + user input
        2. Call LLM
        3. If tool calls returned, execute them with timeout + dedup
        4. Feed results back to LLM
        5. Repeat until no more tool calls or max rounds
        6. Return final text response
        """
        t0 = time.time()
        messages = self._build_messages(user_input, context)
        tool_schemas = self._get_tool_schemas()

        seen_tool_calls = set()  # Deduplication

        for round_num in range(self._max_tool_rounds):
            # Call LLM
            for listener in self._before_listeners:
                listener(messages)

            result = await self.generate(messages, tools=tool_schemas if tool_schemas else None)

            tool_calls = result.get("tool_calls", [])
            text = result.get("text", "")

            if not tool_calls:
                response = LLMResponseEvent(
                    text=text,
                    raw=result.get("raw"),
                    latency_ms=(time.time() - t0) * 1000,
                )
                for listener in self._after_listeners:
                    listener(response)
                return response

            # Execute tool calls with deduplication
            tool_results = []
            for tc in tool_calls:
                tc_key = self._tc_key(tc)
                if tc_key in seen_tool_calls:
                    continue
                seen_tool_calls.add(tc_key)

                tc_result = await self._run_one_tool(tc)
                tool_results.append(tc_result)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "name": tc.get("name", ""),
                    "content": json.dumps(tc_result.get("result", {}), default=str),
                })

        # Max rounds reached
        return LLMResponseEvent(
            text=text or "I've reached my tool-calling limit.",
            latency_ms=(time.time() - t0) * 1000,
        )

    def _tc_key(self, tc: Dict) -> Tuple:
        """Generate unique key for tool call deduplication."""
        return (
            tc.get("id"),
            tc.get("name", ""),
            json.dumps(tc.get("arguments", {}), sort_keys=True),
        )

    async def _run_one_tool(self, tc: Dict) -> Dict:
        """Run a single tool call with timeout and event emission."""
        name = tc.get("name", "")
        args = tc.get("arguments", {})
        t0 = time.time()

        # Emit start event
        self._emit_tool_event(ToolCallEvent(
            tool_name=name, arguments=args,
            tool_call_id=tc.get("id"),
        ))

        func = self._tool_registry.get(name)
        if not func:
            event = ToolCallEvent(
                tool_name=name, success=False,
                error=f"Tool '{name}' not registered",
                execution_time_ms=(time.time() - t0) * 1000,
            )
            self._emit_tool_event(event)
            return {"name": name, "result": {"error": f"Unknown tool: {name}"}}

        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**args), timeout=self._tool_timeout_s
                )
            else:
                result = func(**args)

            elapsed = (time.time() - t0) * 1000
            self._emit_tool_event(ToolCallEvent(
                tool_name=name, success=True, result=result,
                execution_time_ms=elapsed,
            ))
            return {"name": name, "result": result}

        except asyncio.TimeoutError:
            elapsed = (time.time() - t0) * 1000
            self._emit_tool_event(ToolCallEvent(
                tool_name=name, success=False,
                error=f"Timeout after {self._tool_timeout_s}s",
                execution_time_ms=elapsed,
            ))
            return {"name": name, "result": {"error": "Tool timed out"}}

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            self._emit_tool_event(ToolCallEvent(
                tool_name=name, success=False, error=str(e),
                execution_time_ms=elapsed,
            ))
            return {"name": name, "result": {"error": str(e)}}

    def _build_messages(self, user_input: str, context: Optional[Dict] = None) -> List[Dict]:
        messages = []
        if self._instructions:
            sys_text = self._instructions
            if context:
                sys_text += "\n\nContext:\n" + "\n".join(f"- {k}: {v}" for k, v in context.items())
            messages.append({"role": "system", "content": sys_text})
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": user_input})
        return messages

    def _get_tool_schemas(self) -> List[Dict]:
        """Build tool schemas from registered tools."""
        try:
            from function_registry import tool_registry
            return tool_registry.get_tools_schema()
        except ImportError:
            return []

    def get_stats(self) -> Dict:
        return {
            "provider": self.provider_name,
            "tools_registered": len(self._tool_registry),
            "instructions_length": len(self._instructions),
            "conversation_length": len(self._conversation_history),
        }
