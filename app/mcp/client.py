"""
MCP Client Implementation.
Connects to an MCP Server, performs tool discovery, and dispatches tool execution.
"""

import logging
from typing import Any, Dict, List, Optional

from app.mcp.models import MCPTool
from app.mcp.server import MCPServer

logger = logging.getLogger("app.mcp.client")


class MCPClient:
    """
    Model Context Protocol Client.
    Communicates with an MCPServer instance, providing discovery and execution APIs.
    """

    def __init__(self, server: MCPServer):
        self.server = server
        self.server_name = server.name
        self._connected: bool = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Establishes connection / session to the MCP server."""
        self._connected = True
        logger.info(f"[MCPClient] Connected to MCP server '{self.server_name}'")

    async def disconnect(self) -> None:
        """Closes connection to the MCP server."""
        self._connected = False
        logger.info(f"[MCPClient] Disconnected from MCP server '{self.server_name}'")

    async def discover_tools(self) -> List[MCPTool]:
        """
        Queries the connected MCP server for available tools.
        Returns the list of MCPTool specifications.
        """
        if not self._connected:
            await self.connect()

        tools = self.server.list_tools()
        logger.debug(f"[MCPClient:{self.server_name}] Discovered {len(tools)} tools: {[t.name for t in tools]}")
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Dispatches tool invocation to the connected MCP server.
        """
        if not self._connected:
            await self.connect()

        return await self.server.execute_tool(name=name, arguments=arguments, context=context)
