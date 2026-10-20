"""
MCP Exception Classes.
Defines domain-specific errors for tool routing, permissions, authentication,
availability, and network failures.
"""


class MCPError(Exception):
    """Base exception for all Model Context Protocol operations."""
    def __init__(self, message: str, error_type: str = "mcp_error"):
        super().__init__(message)
        self.message = message
        self.error_type = error_type


class MCPPermissionDeniedError(MCPError):
    """Raised when a tool execution is rejected by access control policies."""
    def __init__(self, message: str = "Tool execution permission denied"):
        super().__init__(message, error_type="permission_denied")


class MCPAuthenticationError(MCPError):
    """Raised when external service credentials/tokens are missing, invalid, or expired."""
    def __init__(self, message: str = "Calendar account not authenticated or token expired"):
        super().__init__(message, error_type="authentication_failure")


class MCPUnavailableError(MCPError):
    """Raised when an MCP server or client is unreachable, stopped, or disconnected."""
    def __init__(self, message: str = "MCP server is offline or unreachable"):
        super().__init__(message, error_type="mcp_unavailable")


class MCPTimeoutError(MCPError):
    """Raised when a tool execution exceeds its execution deadline."""
    def __init__(self, message: str = "Tool execution timed out"):
        super().__init__(message, error_type="timeout")


class MCPAPIFailureError(MCPError):
    """Raised when an upstream external API returns an error or failure response."""
    def __init__(self, message: str = "External API call failed"):
        super().__init__(message, error_type="api_failure")
