"""
API v1 Router aggregator.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import auth, conversations, health, mcp, memories, onboarding, preferences, tools, voice

api_router = APIRouter()

# Core health check
api_router.include_router(health.router, tags=["Health"])

# Authentication routes (/api/v1/auth/...)
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Onboarding routes (/api/v1/onboarding/...)
api_router.include_router(onboarding.router, prefix="/onboarding", tags=["Onboarding"])

# Voice routes (/api/v1/voice/...)
api_router.include_router(voice.router, prefix="/voice", tags=["Voice"])

# Conversation routes (/api/v1/conversations/...)
api_router.include_router(conversations.router, prefix="/conversations", tags=["Conversations"])

# Memories routes (/api/v1/memories/...)
api_router.include_router(memories.router, prefix="/memories", tags=["Memories"])

# MCP Tool Discovery routes (/api/v1/mcp/...)
api_router.include_router(mcp.router, prefix="/mcp", tags=["MCP Tools"])

# Connected Tools Management routes (/api/v1/tools/...)
api_router.include_router(tools.router, prefix="/tools", tags=["Connected Tools"])

# User Preferences & Proactivity routes (/api/v1/preferences/...)
api_router.include_router(preferences.preferences_router, prefix="/preferences", tags=["Preferences & Proactivity"])
api_router.include_router(preferences.goals_router, prefix="/goals", tags=["Goals"])
api_router.include_router(preferences.habits_router, prefix="/habits", tags=["Habits"])

