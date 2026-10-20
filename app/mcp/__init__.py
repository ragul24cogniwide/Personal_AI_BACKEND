"""
Model Context Protocol (MCP) Architecture for Personal AI Assistant.
Provides secure, backend-controlled tool integration between Gemini and external services.

Components:
- MCPClient: Client interface connecting to MCP servers.
- MCPServer: Server hosting tools and executing external service actions.
- MCPTool: Standardized MCP tool definition with schema translation for Gemini.
- ToolRouter: Central coordinator managing tool discovery, routing, and execution.
- ToolPermission: Security and authorization engine for tool access control.
- ToolExecution: Execution tracker with safe secret scrubbing and auditing.
"""

from app.mcp.models import (
    ExecutionStatus,
    MCPTool,
    PermissionLevel,
    ToolExecution,
    ToolPermission,
)
from app.mcp.server import MCPServer
from app.mcp.client import MCPClient
from app.mcp.router import ToolRouter, tool_router

__all__ = [
    "ExecutionStatus",
    "MCPClient",
    "MCPServer",
    "MCPTool",
    "PermissionLevel",
    "ToolExecution",
    "ToolPermission",
    "ToolRouter",
    "tool_router",
]
