"""
Model imports for Alembic and application discovery.
"""

from app.database.base import Base
from app.models.user import (
    User,
    UserProfile,
    UserPreference,
    UserGoal,
    UserHabit,
    ConnectedAccount,
)
from app.models.conversation import (
    Conversation,
    ConversationMessage,
)
from app.models.memory import Memory
from app.models.mcp import (
    MCPServer,
    ToolPermission,
)
from app.models.settings import AssistantSetting

__all__ = [
    "Base",
    "User",
    "UserProfile",
    "UserPreference",
    "UserGoal",
    "UserHabit",
    "ConnectedAccount",
    "Conversation",
    "ConversationMessage",
    "Memory",
    "MCPServer",
    "ToolPermission",
    "AssistantSetting",
]
