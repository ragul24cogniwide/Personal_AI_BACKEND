"""
Model Context Protocol (MCP) Data Models and Abstractions.

Defines:
- MCPTool: Tool definition and schema converter for Gemini
- ToolPermission: Security authorization model
- ToolExecution: Safe audit record with secret scrubbing
"""

import copy
import datetime
from enum import Enum
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

try:
    from google.genai import types
except ImportError:
    types = None  # type: ignore

logger = logging.getLogger("app.mcp")

# Sensitive parameter keys that must ALWAYS be sanitized from logs and Gemini declarations
SENSITIVE_PATTERNS = [
    r"token",
    r"secret",
    r"password",
    r"api[_-]?key",
    r"auth",
    r"credential",
    r"bearer",
    r"cookie",
    r"private[_-]?key",
    r"access[_-]?key",
]
SENSITIVE_REGEX = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


def sanitize_sensitive_data(obj: Any) -> Any:
    """
    Recursively scrubs sensitive keys and values from dictionaries, lists, and primitives.
    Replaces sensitive values with '[REDACTED]' to ensure credentials never leak into logs.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if SENSITIVE_REGEX.search(str(k)):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = sanitize_sensitive_data(v)
        return cleaned
    elif isinstance(obj, list):
        return [sanitize_sensitive_data(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_sensitive_data(item) for item in obj)
    return obj


def json_schema_to_gemini_schema(schema_dict: Optional[Dict[str, Any]]) -> Optional[types.Schema]:
    """
    Converts a standard JSON Schema dictionary into Google GenAI types.Schema.
    Security guarantee: Excludes any backend-only or credential annotations.
    """
    if not schema_dict or types is None:
        return None

    raw_type = schema_dict.get("type", "object").upper()
    gemini_type = getattr(types.Type, raw_type, types.Type.OBJECT)

    properties: Optional[Dict[str, types.Schema]] = None
    if "properties" in schema_dict and isinstance(schema_dict["properties"], dict):
        properties = {}
        for prop_name, prop_def in schema_dict["properties"].items():
            prop_type_str = prop_def.get("type", "string").upper()
            prop_gemini_type = getattr(types.Type, prop_type_str, types.Type.STRING)
            
            # Recursive schema for nested objects
            nested_props = None
            if prop_gemini_type == types.Type.OBJECT and "properties" in prop_def:
                nested_sub = json_schema_to_gemini_schema(prop_def)
                if nested_sub:
                    nested_props = nested_sub.properties

            # Item schema for arrays
            items_schema = None
            if prop_gemini_type == types.Type.ARRAY and "items" in prop_def:
                items_schema = json_schema_to_gemini_schema(prop_def["items"])

            properties[prop_name] = types.Schema(
                type=prop_gemini_type,
                description=prop_def.get("description", ""),
                properties=nested_props,
                items=items_schema,
                nullable=prop_def.get("nullable", False),
            )

    required_fields = schema_dict.get("required") or []

    return types.Schema(
        type=gemini_type,
        description=schema_dict.get("description"),
        properties=properties,
        required=required_fields if required_fields else None,
    )


class PermissionLevel(str, Enum):
    """Permission levels for tool execution access control."""
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class ExecutionStatus(str, Enum):
    """Lifecycle statuses for tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"


class MCPTool(BaseModel):
    """
    Standard Model Context Protocol tool specification.
    Exposes metadata and schema to Gemini while keeping execution logic in backend.
    """
    name: str = Field(..., description="Unique tool identifier, e.g. 'get_calendar_events'")
    display_name: str = Field("", description="Human-readable tool title, e.g. 'Calendar'")
    description: str = Field(..., description="Clear explanation of tool capabilities for the LLM")
    category: str = Field("general", description="Functional category, e.g. 'calendar'")
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="Standard JSON Schema defining tool input parameters",
    )
    server_name: str = Field("default", description="Name of the MCP server hosting this tool")
    required_permissions: List[str] = Field(
        default_factory=list,
        description="Security permissions/scopes needed, e.g. ['calendar.read']",
    )
    is_read_only: bool = Field(True, description="True if tool does not mutate external state")
    is_sensitive: bool = Field(False, description="True if tool requires explicit user authorization")

    def to_gemini_function_declaration(self) -> types.FunctionDeclaration:
        """
        Converts the tool definition into a Gemini FunctionDeclaration.
        Ensures NO credentials, API keys, or backend headers are ever sent to Gemini.
        """
        if types is None:
            raise RuntimeError("google-genai SDK is not installed")

        gemini_schema = json_schema_to_gemini_schema(self.input_schema)
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=gemini_schema,
        )

    def to_gemini_tool(self) -> types.Tool:
        """Wraps the function declaration into a Gemini types.Tool object."""
        if types is None:
            raise RuntimeError("google-genai SDK is not installed")
        return types.Tool(function_declarations=[self.to_gemini_function_declaration()])


class ToolPermission(BaseModel):
    """
    Security authorization rule defining which tools a user or system role may execute.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[UUID] = Field(None, description="Target user ID or None for global rule")
    tool_name: str = Field(..., description="Specific tool name or '*' for all tools")
    permission_level: PermissionLevel = Field(
        default=PermissionLevel.ALLOW,
        description="ALLOW, REQUIRE_CONFIRMATION, or DENY",
    )
    allowed_scopes: List[str] = Field(
        default_factory=list,
        description="Permitted OAuth/API scopes, e.g. ['calendar.read']",
    )
    is_active: bool = Field(True, description="Whether this permission rule is active")

    def allows(self, target_tool: str, required_scopes: Optional[List[str]] = None) -> bool:
        """Checks if this permission rule authorizes execution of the target tool."""
        if not self.is_active:
            return False
        if self.permission_level == PermissionLevel.DENY:
            return False

        # Match tool name or wildcard
        if self.tool_name != "*" and self.tool_name != target_tool:
            return False

        # Check required scopes if any
        if required_scopes:
            for scope in required_scopes:
                if scope not in self.allowed_scopes and "*" not in self.allowed_scopes:
                    return False

        return self.permission_level in (PermissionLevel.ALLOW, PermissionLevel.REQUIRE_CONFIRMATION)


class ToolExecution(BaseModel):
    """
    Audit record tracking a single tool execution lifecycle.
    Implements strict sanitization to never log passwords or credentials.
    """
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    tool_name: str
    server_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: Optional[Any] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    started_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    completed_at: Optional[datetime.datetime] = None
    duration_ms: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        """Sanitizes arguments upon creation to guarantee no secrets remain in memory."""
        self.arguments = sanitize_sensitive_data(self.arguments)

    def complete(self, result: Any, duration_ms: float) -> None:
        """Marks execution as successfully completed."""
        self.status = ExecutionStatus.COMPLETED
        self.result = sanitize_sensitive_data(result)
        self.error_type = None
        self.duration_ms = duration_ms
        self.completed_at = datetime.datetime.now(datetime.timezone.utc)
        self.log_safe()

    def fail(self, error: str, duration_ms: float, error_type: str = "api_failure") -> None:
        """Marks execution as failed with error category."""
        self.status = ExecutionStatus.FAILED
        self.error_message = error
        self.error_type = error_type
        self.duration_ms = duration_ms
        self.completed_at = datetime.datetime.now(datetime.timezone.utc)
        self.log_safe()

    def deny(self, reason: str, error_type: str = "permission_denied") -> None:
        """Marks execution as denied by security permission policy."""
        self.status = ExecutionStatus.DENIED
        self.error_message = f"Permission denied: {reason}"
        self.error_type = error_type
        self.duration_ms = 0.0
        self.completed_at = datetime.datetime.now(datetime.timezone.utc)
        self.log_safe()

    def log_safe(self) -> None:
        """
        Safely logs the tool execution audit event without leaking secrets.
        """
        sanitized_args = sanitize_sensitive_data(self.arguments)
        summary = ""
        if self.status == ExecutionStatus.COMPLETED:
            if isinstance(self.result, dict):
                summary = f"keys={list(self.result.keys())}"
            elif isinstance(self.result, list):
                summary = f"items={len(self.result)}"
            else:
                summary = str(self.result)[:80]

        logger.info(
            f"[ToolExecution:{self.execution_id[:8]}] "
            f"tool='{self.tool_name}' server='{self.server_name}' "
            f"status={self.status.value} duration={self.duration_ms}ms "
            f"args={sanitized_args} summary='{summary}' "
            f"{f'error={self.error_message}' if self.error_message else ''}"
        )
