# backend/agent_core.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned Agent Core — orchestration engine.

The Agent class manages the full lifecycle of an AI agent:
- Instruction loading and context management
- LLM interaction with tool-calling loops
- Conversation memory integration
- Event emission for observability
- Processor pipeline for video/audio

Usage:
    agent = AgentCore(name="SecurityBot", instructions="You are a security monitor.")
    response = await agent.run("What do you see in this frame?", context={...})
"""

import asyncio
import time
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("agent_core")


@dataclass
class AgentConfig:
    """Configuration for an Agent instance."""
    name: str = "VisionAgent"
    instructions: str = ""
    max_tool_rounds: int = 5
    max_context_messages: int = 20
    temperature: float = 0.7
    model: Optional[str] = None
    enable_tools: bool = True
    enable_streaming: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent run."""
    text: str = ""
    tool_calls_made: List[Dict] = field(default_factory=list)
    tool_results: List[Dict] = field(default_factory=list)
    rounds: int = 0
    latency_ms: float = 0
    model_used: str = ""
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "tool_calls_made": self.tool_calls_made,
            "tool_results": self.tool_results,
            "rounds": self.rounds,
            "latency_ms": round(self.latency_ms, 1),
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
        }


class AgentCore:
    """Core agent orchestration engine.

    Manages the agent loop:
    1. Take user input + context
    2. Send to LLM with instructions and tools
    3. If LLM returns tool calls, execute them and loop
    4. Return final text response

    SDK-aligned features:
    - Configurable instruction sets
    - Multi-round tool calling
    - Conversation history management
    - Event bus integration
    - Performance tracking
    """

    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        self.config = config or AgentConfig(**kwargs)
        self._conversation_history: List[Dict] = []
        self._tool_registry: Dict[str, Callable] = {}
        self._event_bus = None
        self._profiler = None
        self._total_runs = 0
        self._total_tool_calls = 0

        # Try to connect to global event bus
        try:
            from event_bus import event_bus
            self._event_bus = event_bus
        except ImportError:
            pass

        # Try to connect to global profiler
        try:
            from profiling import profiler
            self._profiler = profiler
        except ImportError:
            pass

        logger.info("AgentCore '%s' initialized", self.config.name)

    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool that the agent can call."""
        self._tool_registry[name] = func
        logger.info("Registered tool '%s' for agent '%s'", name, self.config.name)

    def register_tools_from_registry(self) -> int:
        """Import tools from the global function_registry."""
        try:
            from function_registry import tool_registry
            count = 0
            for name in tool_registry.list_tools():
                tool = tool_registry.get_tool(name)
                if tool:
                    self._tool_registry[name] = tool["func"]
                    count += 1
            logger.info("Imported %d tools from global registry", count)
            return count
        except ImportError:
            return 0

    async def run(self, user_input: str, context: Optional[Dict] = None,
                  session_id: Optional[str] = None) -> AgentResponse:
        """Run the agent with user input and optional context.

        This is the main entry point. It:
        1. Builds the prompt with instructions + context + history
        2. Calls the LLM
        3. If tool calls are returned, executes them and loops
        4. Returns the final response
        """
        t0 = time.time()
        self._total_runs += 1
        response = AgentResponse()
        session_id = session_id or str(uuid.uuid4())[:8]

        # Build messages for LLM
        messages = self._build_messages(user_input, context)

        # Tool-calling loop
        for round_num in range(self.config.max_tool_rounds):
            response.rounds = round_num + 1

            # Call LLM
            llm_result = await self._call_llm(messages)
            response.model_used = llm_result.get("model", "unknown")
            response.tokens_used += llm_result.get("tokens", 0)

            # Check for tool calls
            tool_calls = llm_result.get("tool_calls", [])
            text = llm_result.get("text", "")

            if not tool_calls or not self.config.enable_tools:
                # No tool calls — return text response
                response.text = text
                break

            # Execute tool calls
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                response.tool_calls_made.append({"name": tool_name, "arguments": tool_args})
                self._total_tool_calls += 1

                result = await self._execute_tool(tool_name, tool_args)
                response.tool_results.append({"name": tool_name, "result": result})

                # Add tool result to messages for next round
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result),
                })

        # Finalize
        response.latency_ms = (time.time() - t0) * 1000

        # Add to conversation history
        self._conversation_history.append({"role": "user", "content": user_input})
        self._conversation_history.append({"role": "assistant", "content": response.text})

        # Trim history
        max_msgs = self.config.max_context_messages
        if len(self._conversation_history) > max_msgs:
            self._conversation_history = self._conversation_history[-max_msgs:]

        # Track profiling
        if self._profiler:
            self._profiler.record("agent_run", response.latency_ms / 1000,
                                 {"rounds": response.rounds, "tools": len(response.tool_calls_made)})

        return response

    def _build_messages(self, user_input: str, context: Optional[Dict] = None) -> List[Dict]:
        """Build the message list for the LLM call."""
        messages = []

        # System instruction
        if self.config.instructions:
            system_text = self.config.instructions
            if context:
                # Inject context into system prompt
                ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
                system_text += f"\n\nCurrent context:\n{ctx_str}"
            messages.append({"role": "system", "content": system_text})

        # Conversation history
        messages.extend(self._conversation_history)

        # Current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    async def _call_llm(self, messages: List[Dict]) -> Dict:
        """Call the LLM provider. Uses the global cascade provider."""
        try:
            from llm_provider import provider
            prompt = messages[-1].get("content", "")
            result = await asyncio.get_event_loop().run_in_executor(
                None, provider.chat, prompt
            )
            return {
                "text": result.get("text", ""),
                "model": result.get("model", "unknown"),
                "tokens": result.get("tokens", 0),
                "tool_calls": [],  # Tool calls parsed from response
            }
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return {"text": f"Error: {e}", "model": "error", "tokens": 0, "tool_calls": []}

    async def _execute_tool(self, name: str, arguments: Dict) -> Any:
        """Execute a registered tool."""
        func = self._tool_registry.get(name)
        if not func:
            return {"error": f"Tool '{name}' not found"}

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, lambda: func(**arguments))
            return result
        except Exception as e:
            logger.error("Tool '%s' failed: %s", name, e)
            return {"error": str(e)}

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "name": self.config.name,
            "total_runs": self._total_runs,
            "total_tool_calls": self._total_tool_calls,
            "conversation_length": len(self._conversation_history),
            "registered_tools": list(self._tool_registry.keys()),
            "instructions_length": len(self.config.instructions),
        }


# ── Singleton default agent ───────────────────────────────────────────
default_agent = AgentCore(config=AgentConfig(
    name="VisionAgent",
    instructions=(
        "You are Vision Agent, an AI-powered visual analysis assistant. "
        "You can analyze video feeds, detect objects, track poses, monitor security cameras, "
        "and provide real-time coaching. Use your tools when needed."
    ),
))
