"""
MCP Server Base Class.
Hosts MCP tools, manages tool registration, exposes discovery metadata,
and executes tool logic securely on the backend.
"""

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.mcp.models import MCPTool

logger = logging.getLogger("app.mcp.server")


class MCPServer:
    """
    Model Context Protocol Server implementation.
    Manages tools, discovery endpoints, and backend-isolated execution.
    """

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self._tools: Dict[str, MCPTool] = {}
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}

    def register_tool(
        self,
        tool: MCPTool,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """
        Registers an MCP tool definition and its executable handler.
        Ensures the tool is associated with this server instance.
        """
        tool.server_name = self.name
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler
        logger.info(f"[{self.name}] Registered tool: '{tool.name}' ({tool.description[:40]}...)")

    def list_tools(self) -> List[MCPTool]:
        """
        Discovery endpoint: Lists all available tools provided by this MCP Server.
        Only metadata and schema are returned; implementation details remain private.
        """
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Returns tool specification by name if present."""
        return self._tools.get(name)

    async def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Executes the registered handler for the requested tool name.
        The LLM does NOT execute this directly; the backend calls this method.
        """
        if name not in self._tools or name not in self._handlers:
            raise KeyError(f"Tool '{name}' is not hosted by MCP server '{self.name}'")

        handler = self._handlers[name]
        logger.debug(f"[{self.name}] Executing tool '{name}' with args keys: {list(arguments.keys())}")

        try:
            # Check if handler accepts context
            sig = inspect.signature(handler)
            if "context" in sig.parameters:
                result = handler(**arguments, context=context)
            else:
                result = handler(**arguments)

            # Await async handlers
            if inspect.isawaitable(result):
                result = await result

            return result

        except Exception as exc:
            logger.error(f"[{self.name}] Tool '{name}' execution failed: {exc}", exc_info=True)
            raise
