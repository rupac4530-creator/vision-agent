# backend/function_registry.py
"""
Function Registry — SDK-Aligned Tool Calling System

Ported from Vision-Agents SDK (agents-core/vision_agents/core/llm/function_registry.py)
Allows registering Python functions as LLM-callable tools with auto-generated
JSON schemas from type hints. Supports async execution and error handling.

Usage:
    registry = FunctionRegistry()

    @registry.register(description="Get current weather for a city")
    async def get_weather(city: str, units: str = "celsius") -> dict:
        return {"city": city, "temp": 22, "units": units}

    # Get OpenAI-compatible tool schemas
    tools = registry.get_tools_schema()

    # Execute a tool call from LLM response
    result = await registry.execute("get_weather", {"city": "London"})
"""

import asyncio
import inspect
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

logger = logging.getLogger("function_registry")


# ── Type mapping for JSON Schema ─────────────────────────────────────
_PY_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_json_schema(py_type: type) -> dict:
    """Convert a Python type hint to a JSON Schema type descriptor."""
    origin = getattr(py_type, "__origin__", None)

    if origin is list:
        args = getattr(py_type, "__args__", (Any,))
        item_type = args[0] if args else Any
        return {"type": "array", "items": _python_type_to_json_schema(item_type)}

    if origin is dict:
        return {"type": "object"}

    # Optional[X] → Union[X, None] → extract X
    if origin is Union:
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])

    if origin is type(None):
        return {"type": "null"}

    # Enum support — SDK-aligned
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        return {"type": "string", "enum": [e.value for e in py_type]}

    json_type = _PY_TO_JSON_TYPE.get(py_type, "string")
    return {"type": json_type}


@dataclass
class RegisteredFunction:
    """Metadata for a registered callable tool."""
    name: str
    description: str
    fn: Callable
    parameters_schema: dict
    required_params: List[str]
    is_async: bool
    registered_at: float = field(default_factory=time.time)
    call_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0


class FunctionRegistry:
    """
    Registry for LLM-callable functions (tools).

    Mirrors the Vision-Agents SDK pattern where functions are registered
    with decorators and automatically exposed as OpenAI-compatible tool schemas.
    """

    def __init__(self):
        self._functions: Dict[str, RegisteredFunction] = {}
        self._middleware: List[Callable] = []
        logger.info("FunctionRegistry initialized")

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        description: str = "",
        name: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a function as an LLM-callable tool.

        @registry.register(description="Get weather for a location")
        async def get_weather(city: str) -> dict:
            ...
        """
        def decorator(fn: Callable) -> Callable:
            func_name = name or fn.__name__
            hints = get_type_hints(fn)
            sig = inspect.signature(fn)

            # Build parameters schema from type hints
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue

                param_type = hints.get(param_name, str)
                if param_type is inspect.Parameter.empty:
                    param_type = str

                # Skip return type
                if param_name == "return":
                    continue

                prop = _python_type_to_json_schema(param_type)

                # Add description from docstring if available
                doc = inspect.getdoc(fn) or ""
                prop_desc = ""
                for line in doc.split("\n"):
                    if param_name in line and ":" in line:
                        prop_desc = line.split(":", 1)[-1].strip()
                        break
                if prop_desc:
                    prop["description"] = prop_desc

                properties[param_name] = prop

                # If no default value, it's required
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            schema = {
                "type": "object",
                "properties": properties,
            }
            if required:
                schema["required"] = required

            reg = RegisteredFunction(
                name=func_name,
                description=description or inspect.getdoc(fn) or f"Call {func_name}",
                fn=fn,
                parameters_schema=schema,
                required_params=required,
                is_async=asyncio.iscoroutinefunction(fn),
            )

            self._functions[func_name] = reg
            logger.info("Registered function: %s (%d params, async=%s)",
                        func_name, len(properties), reg.is_async)
            return fn

        return decorator

    def register_function(self, fn: Callable, description: str = "", name: Optional[str] = None):
        """Imperative registration (non-decorator)."""
        wrapped = self.register(description=description, name=name)
        wrapped(fn)

    # ── Schema Generation ─────────────────────────────────────────────

    def get_tools_schema(self) -> List[dict]:
        """
        Get OpenAI-compatible tools schema for all registered functions.
        Returns list of tool definitions ready for the LLM API.
        """
        tools = []
        for reg in self._functions.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": reg.name,
                    "description": reg.description,
                    "parameters": reg.parameters_schema,
                }
            })
        return tools

    def get_schema(self, name: str) -> Optional[dict]:
        """Get schema for a single function by name."""
        reg = self._functions.get(name)
        if not reg:
            return None
        return {
            "type": "function",
            "function": {
                "name": reg.name,
                "description": reg.description,
                "parameters": reg.parameters_schema,
            }
        }

    def list_functions(self) -> List[dict]:
        """List all registered functions with metadata."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "parameters": list(r.parameters_schema.get("properties", {}).keys()),
                "required": r.required_params,
                "is_async": r.is_async,
                "call_count": r.call_count,
                "error_count": r.error_count,
                "avg_latency_ms": (r.total_latency_ms / r.call_count) if r.call_count > 0 else 0,
            }
            for r in self._functions.values()
        ]

    # ── Execution ─────────────────────────────────────────────────────

    async def execute(self, name: str, arguments: dict) -> dict:
        """
        Execute a registered function by name with given arguments.
        Returns {"ok": True, "result": ...} or {"ok": False, "error": ...}
        """
        reg = self._functions.get(name)
        if not reg:
            return {"ok": False, "error": f"Unknown function: {name}",
                    "available": list(self._functions.keys())}

        start = time.monotonic()
        try:
            # Filter arguments to only include valid parameters
            sig = inspect.signature(reg.fn)
            valid_args = {k: v for k, v in arguments.items() if k in sig.parameters}

            if reg.is_async:
                result = await reg.fn(**valid_args)
            else:
                result = reg.fn(**valid_args)

            elapsed_ms = (time.monotonic() - start) * 1000
            reg.call_count += 1
            reg.total_latency_ms += elapsed_ms

            logger.info("Executed %s in %.1fms", name, elapsed_ms)
            return {"ok": True, "result": result, "latency_ms": round(elapsed_ms, 1)}

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            reg.error_count += 1
            reg.total_latency_ms += elapsed_ms
            tb = traceback.format_exc()
            logger.error("Function %s failed: %s\n%s", name, exc, tb)
            return {"ok": False, "error": str(exc), "latency_ms": round(elapsed_ms, 1)}

    async def execute_tool_call(self, tool_call: dict) -> dict:
        """
        Execute from an OpenAI-style tool_call object.
        Expected format: {"function": {"name": "...", "arguments": "..."}}
        """
        func_info = tool_call.get("function", tool_call)
        name = func_info.get("name", "")
        args_raw = func_info.get("arguments", "{}")

        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except json.JSONDecodeError:
                return {"ok": False, "error": f"Invalid JSON arguments: {args_raw}"}
        else:
            arguments = args_raw

        return await self.execute(name, arguments)

    # ── MCP Tool Registration (explicit schema, no introspection) ─────

    def register_mcp_tool(self, name: str, description: str,
                          parameters_schema: dict, handler: Callable):
        """Register a tool with explicit JSON schema (MCP / external API tools)."""
        reg = RegisteredFunction(
            name=name,
            description=description,
            fn=handler,
            parameters_schema=parameters_schema,
            required_params=parameters_schema.get("required", []),
            is_async=asyncio.iscoroutinefunction(handler),
        )
        self._functions[name] = reg
        logger.info("Registered MCP tool: %s", name)

    def get_tool(self, name: str) -> Optional[dict]:
        """Get a tool's function and metadata by name."""
        reg = self._functions.get(name)
        if not reg:
            return None
        return {"name": reg.name, "func": reg.fn, "schema": reg.parameters_schema}

    def get_callable(self, name: str) -> Optional[Callable]:
        """Get just the callable function by name."""
        reg = self._functions.get(name)
        return reg.fn if reg else None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._functions)

    def has(self, name: str) -> bool:
        return name in self._functions

    def __contains__(self, name: str) -> bool:
        return name in self._functions

    def __len__(self) -> int:
        return len(self._functions)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._functions.keys())


# ══════════════════════════════════════════════════════════════════════
# Global registry singleton + built-in tools
# ══════════════════════════════════════════════════════════════════════

tool_registry = FunctionRegistry()


@tool_registry.register(description="Get the current server time and timezone")
async def get_server_time() -> dict:
    """Returns the current server time in ISO format."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(), "unix": now.timestamp()}


@tool_registry.register(description="Get system health and resource usage")
async def get_system_info() -> dict:
    """Returns CPU, memory, and disk usage of the server."""
    import os
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent if os.name != "nt" else psutil.disk_usage("C:\\").percent,
            "pid": os.getpid(),
        }
    except ImportError:
        return {"cpu_percent": -1, "memory_percent": -1, "note": "psutil not installed"}


@tool_registry.register(description="List all available AI models in the LLM cascade")
async def list_models() -> dict:
    """Returns the current LLM provider cascade configuration."""
    try:
        from llm_provider import provider
        if hasattr(provider, "all_providers_info"):
            return {"providers": provider.all_providers_info(), "active": provider.name}
        return {"active": provider.name, "model": provider.model_id}
    except Exception as e:
        return {"error": str(e)}


@tool_registry.register(description="Search for objects detected in the most recent video analysis")
async def search_detections(object_type: str = "all") -> dict:
    """Search through detected objects from the latest video analysis."""
    import os, json
    analysis_dir = os.path.join(os.path.dirname(__file__), "analysis")
    if not os.path.isdir(analysis_dir):
        return {"results": [], "note": "No analyses found"}

    results = []
    for fname in sorted(os.listdir(analysis_dir), reverse=True)[:5]:
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(analysis_dir, fname)) as f:
                data = json.load(f)
            detections = data.get("detections", [])
            if object_type != "all":
                detections = [d for d in detections if object_type.lower() in d.get("label", "").lower()]
            if detections:
                results.append({"file": fname, "matches": detections[:10]})
        except Exception:
            pass

    return {"query": object_type, "results": results, "files_searched": min(5, len(results))}


logger.info("FunctionRegistry ready with %d built-in tools", tool_registry.count)
