"""
FastAPI Application Entry Point.
Production-ready configuration with CORS, lifespan management, and routers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.api.v1.endpoints.health import router as health_router
from app.core.config import settings
from app.core.redis import close_redis_client, get_redis_client
from app.database.session import async_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Redis connection pool
    try:
        await get_redis_client()
    except Exception as e:
        print(f"[Warning] Could not initialize Redis on startup: {e}")

    # Startup: Initialize MCP tool architecture and default servers
    try:
        from app.mcp import tool_router
        from app.mcp.servers import CalendarMCPServer, ContactsMCPServer
        if "calendar-service" not in tool_router._servers:
            tool_router.register_server(CalendarMCPServer())
        if "contacts-service" not in tool_router._servers:
            tool_router.register_server(ContactsMCPServer())
        await tool_router.discover_all_tools()
    except Exception as e:
        print(f"[Warning] Could not initialize MCP subsystem on startup: {e}")

    yield

    # Shutdown: Cleanly close Redis and DB connection pools
    await close_redis_client()
    await async_engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Direct root health endpoint for load balancers (e.g. Render health check at /health)
app.include_router(health_router, tags=["Health"])

# Direct /auth routes for root POST /auth/... and GET /auth/me
from app.api.v1.endpoints.auth import router as auth_router
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# Direct /onboarding routes
from app.api.v1.endpoints.onboarding import router as onboarding_router
app.include_router(onboarding_router, prefix="/onboarding", tags=["Onboarding"])

# Direct /ws/voice routes for top-level WebSocket connectivity
from app.api.v1.endpoints.voice import router as voice_router
app.include_router(voice_router, prefix="/ws/voice", tags=["Voice WebSocket"])
app.include_router(voice_router, prefix="/voice", tags=["Voice"])

# Direct /conversations routes
from app.api.v1.endpoints.conversations import router as conversations_router
app.include_router(conversations_router, prefix="/conversations", tags=["Conversations"])

# Direct /memories routes
from app.api.v1.endpoints.memories import router as memories_router
app.include_router(memories_router, prefix="/memories", tags=["Memories"])

# Direct /mcp routes
from app.api.v1.endpoints.mcp import router as mcp_router
app.include_router(mcp_router, prefix="/mcp", tags=["MCP Tools"])

# Direct /tools routes
from app.api.v1.endpoints.tools import router as tools_router
app.include_router(tools_router, prefix="/tools", tags=["Connected Tools"])

# Direct /greeting endpoint
from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.services.greeting import greeting_service
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

@app.get("/greeting", tags=["Greeting"])
async def root_greeting(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await greeting_service.generate_greeting(db=db, user_id=current_user.id)

# Direct /preferences, /goals, and /habits routes
from app.api.v1.endpoints.preferences import (
    preferences_router,
    goals_router,
    habits_router,
)
app.include_router(preferences_router, prefix="/preferences", tags=["Preferences & Proactivity"])
app.include_router(goals_router, prefix="/goals", tags=["Goals"])
app.include_router(habits_router, prefix="/habits", tags=["Habits"])

# API v1 Router (/api/v1/...)
app.include_router(api_router, prefix=settings.API_V1_STR)



@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": "0.1.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }
