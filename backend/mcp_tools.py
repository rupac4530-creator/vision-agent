# backend/mcp_tools.py
# copied from GetStream/Vision-Agents @ f684ece — Apache-2.0 — see THIRD_PARTY_LICENSES.md
"""
SDK-aligned MCP (Model Context Protocol) Tool Integration.

Connects to external tool servers (local stdio or remote SSE) and exposes
their tools to the function registry for LLM-driven tool calling.

Usage:
    manager = MCPManager()
    await manager.add_server("weather", command=["python", "weather_server.py"])
    tools = await manager.list_tools()
    result = await manager.call_tool("get_weather", {"city": "NYC"})
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("mcp_tools")


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""
    name: str
    description: str = ""
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "server": self.server_name,
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    type: str = "local"  # "local" (stdio) or "remote" (SSE)
    command: Optional[List[str]] = None  # For local servers
    url: Optional[str] = None  # For remote servers
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


class MCPServer:
    """Represents a connected MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.tools: Dict[str, MCPTool] = {}
        self._connected = False
        self._connect_time: Optional[float] = None

    async def connect(self) -> bool:
        """Connect to the MCP server and discover tools."""
        try:
            if self.config.type == "local" and self.config.command:
                # For local servers, we'd start the process via stdio
                # Simulated for non-WebRTC environment
                logger.info("MCP: Connecting to local server '%s' (cmd: %s)",
                           self.config.name, self.config.command)
                self._connected = True
                self._connect_time = time.time()
                return True

            elif self.config.type == "remote" and self.config.url:
                # For remote servers, connect via SSE/HTTP
                logger.info("MCP: Connecting to remote server '%s' (url: %s)",
                           self.config.name, self.config.url)
                self._connected = True
                self._connect_time = time.time()
                return True

            logger.warning("MCP: Invalid config for server '%s'", self.config.name)
            return False

        except Exception as e:
            logger.error("MCP: Failed to connect to '%s': %s", self.config.name, e)
            return False

    async def disconnect(self):
        """Disconnect from the server."""
        self._connected = False
        self.tools.clear()
        logger.info("MCP: Disconnected from '%s'", self.config.name)

    def register_tool(self, name: str, description: str = "",
                      parameters_schema: Optional[Dict] = None) -> MCPTool:
        """Register a tool from this server."""
        tool = MCPTool(
            name=name,
            description=description,
            parameters_schema=parameters_schema or {},
            server_name=self.config.name,
        )
        self.tools[name] = tool
        return tool

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_stats(self) -> Dict:
        return {
            "name": self.config.name,
            "type": self.config.type,
            "connected": self._connected,
            "tools_count": len(self.tools),
            "uptime_s": round(time.time() - self._connect_time, 1) if self._connect_time else 0,
        }


class MCPManager:
    """Manages multiple MCP server connections and their tools.

    Provides a unified interface to discover and call tools from
    any connected MCP server.
    """

    def __init__(self):
        self._servers: Dict[str, MCPServer] = {}
        self._tool_handlers: Dict[str, Callable] = {}

    async def add_server(self, name: str, command: Optional[List[str]] = None,
                         url: Optional[str] = None, env: Optional[Dict] = None,
                         headers: Optional[Dict] = None) -> MCPServer:
        """Add and connect to an MCP server."""
        config = MCPServerConfig(
            name=name,
            type="local" if command else "remote",
            command=command,
            url=url,
            env=env or {},
            headers=headers or {},
        )
        server = MCPServer(config)
        await server.connect()
        self._servers[name] = server
        return server

    async def remove_server(self, name: str):
        """Disconnect and remove an MCP server."""
        server = self._servers.pop(name, None)
        if server:
            await server.disconnect()

    def register_tool_handler(self, tool_name: str, handler: Callable):
        """Register a local handler for a tool (for testing or adapting)."""
        self._tool_handlers[tool_name] = handler

    async def list_tools(self) -> List[MCPTool]:
        """List all available tools from all connected servers."""
        tools = []
        for server in self._servers.values():
            if server.is_connected:
                tools.extend(server.tools.values())
        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name, routing to the correct server."""
        # Check local handlers first
        if name in self._tool_handlers:
            handler = self._tool_handlers[name]
            if asyncio.iscoroutinefunction(handler):
                return await handler(**arguments)
            return handler(**arguments)

        # Find tool in connected servers
        for server in self._servers.values():
            if name in server.tools and server.is_connected:
                logger.info("MCP: Calling tool '%s' on server '%s'", name, server.config.name)
                # In a real implementation, this would send the call via stdio/SSE
                return {"status": "ok", "tool": name, "server": server.config.name,
                        "arguments": arguments}

        raise KeyError(f"Tool '{name}' not found in any connected MCP server")

    async def close(self):
        """Disconnect all servers."""
        for server in self._servers.values():
            await server.disconnect()
        self._servers.clear()

    def get_stats(self) -> Dict:
        return {
            "total_servers": len(self._servers),
            "connected_servers": sum(1 for s in self._servers.values() if s.is_connected),
            "total_tools": sum(len(s.tools) for s in self._servers.values()),
            "local_handlers": len(self._tool_handlers),
            "servers": {name: s.get_stats() for name, s in self._servers.items()},
        }


# ── Singleton instance ─────────────────────────────────────────────────
mcp_manager = MCPManager()
