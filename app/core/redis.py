"""
Redis Client Manager.
Handles caching, temporary session state, realtime voice state, and rate limiting.
"""

import json
from typing import Any, Optional
import redis.asyncio as aioredis
from app.core.config import settings

_redis_client: Optional[aioredis.Redis] = None


async def get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            health_check_interval=30,
        )
    return _redis_client


async def close_redis_client():
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


async def ping_redis() -> bool:
    try:
        client = await get_redis_client()
        return await client.ping()
    except Exception:
        return False


# Real-time Voice State Management
async def set_realtime_voice_state(
    user_id: str, state_data: dict, ttl_seconds: int = 300
) -> bool:
    """Store active voice session metadata (state, audio format, latency metrics)."""
    try:
        client = await get_redis_client()
        key = f"voice:state:{user_id}"
        await client.set(key, json.dumps(state_data), ex=ttl_seconds)
        return True
    except Exception:
        return False


async def get_realtime_voice_state(user_id: str) -> Optional[dict]:
    try:
        client = await get_redis_client()
        key = f"voice:state:{user_id}"
        val = await client.get(key)
        if val:
            return json.loads(val)
        return None
    except Exception:
        return None


# General Caching
async def cache_set(key: str, value: Any, ttl_seconds: int = 3600) -> bool:
    try:
        client = await get_redis_client()
        serialized = json.dumps(value) if not isinstance(value, str) else value
        await client.set(f"cache:{key}", serialized, ex=ttl_seconds)
        return True
    except Exception:
        return False


async def cache_get(key: str) -> Optional[Any]:
    try:
        client = await get_redis_client()
        val = await client.get(f"cache:{key}")
        if val is None:
            return None
        try:
            return json.loads(val)
        except (ValueError, TypeError):
            return val
    except Exception:
        return None


# Sliding Window Rate Limiting
async def check_rate_limit(
    identifier: str, limit: int = 120, window_seconds: int = 60
) -> bool:
    """Returns True if request is allowed, False if limit exceeded."""
    try:
        client = await get_redis_client()
        key = f"ratelimit:{identifier}"
        current = await client.incr(key)
        if current == 1:
            await client.expire(key, window_seconds)
        return current <= limit
    except Exception:
        # Fail-open if Redis is temporarily unreachable
        return True
