"""
Health check response schema.
"""

from datetime import datetime, timezone
from typing import Dict, Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"] = "ok"
    environment: str
    database: Literal["connected", "disconnected"]
    redis: Literal["connected", "disconnected"]
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, str] = Field(default_factory=dict)
