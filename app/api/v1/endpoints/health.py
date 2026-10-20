"""
Health check endpoint.
Tests live PostgreSQL and Redis connectivity.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, status
from app.core.config import settings
from app.core.redis import ping_redis
from app.database.session import check_db_connection
from app.schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check with live database and Redis verification",
)
async def get_health():
    db_connected = await check_db_connection()
    redis_connected = await ping_redis()

    all_healthy = db_connected and redis_connected
    partially_healthy = db_connected or redis_connected

    overall_status = "ok" if all_healthy else ("degraded" if partially_healthy else "error")

    details = {}
    if not db_connected:
        details["database"] = "Unable to connect to PostgreSQL instance"
    if not redis_connected:
        details["redis"] = "Unable to connect to Redis instance"

    return HealthResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        database="connected" if db_connected else "disconnected",
        redis="connected" if redis_connected else "disconnected",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        details=details,
    )
