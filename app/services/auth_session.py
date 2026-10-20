"""
Redis-backed authentication session and token revocation service.
Manages active refresh tokens and blocklisted access tokens.
"""

from typing import Optional
from app.core.redis import get_redis_client


async def store_refresh_token(user_id: str, jti: str, ttl_seconds: int) -> bool:
    """Store active refresh token JTI in Redis with TTL."""
    try:
        redis = await get_redis_client()
        key = f"auth:refresh:{user_id}:{jti}"
        await redis.set(key, "1", ex=ttl_seconds)
        return True
    except Exception as e:
        print(f"[Warning] Redis store_refresh_token failed: {e}")
        return True  # Fallback gracefully if Redis is unavailable


async def is_refresh_token_valid(user_id: str, jti: str) -> bool:
    """Check if refresh token JTI is active and not revoked."""
    try:
        redis = await get_redis_client()
        key = f"auth:refresh:{user_id}:{jti}"
        val = await redis.get(key)
        return val is not None
    except Exception as e:
        print(f"[Warning] Redis is_refresh_token_valid failed: {e}")
        return True  # Fallback to JWT cryptographic verification


async def revoke_refresh_token(user_id: str, jti: str) -> bool:
    """Revoke a specific refresh token."""
    try:
        redis = await get_redis_client()
        key = f"auth:refresh:{user_id}:{jti}"
        await redis.delete(key)
        return True
    except Exception as e:
        print(f"[Warning] Redis revoke_refresh_token failed: {e}")
        return False


async def revoke_all_user_refresh_tokens(user_id: str) -> bool:
    """Revoke all active refresh tokens for a user (e.g. security breach/password change)."""
    try:
        redis = await get_redis_client()
        cursor = "0"
        match_pattern = f"auth:refresh:{user_id}:*"
        while True:
            cursor, keys = await redis.scan(cursor=cursor, match=match_pattern, count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0 or cursor == "0":
                break
        return True
    except Exception as e:
        print(f"[Warning] Redis revoke_all_user_refresh_tokens failed: {e}")
        return False


async def blocklist_access_token(jti: str, ttl_seconds: int) -> bool:
    """Add access token JTI to blocklist on logout until its expiration."""
    try:
        redis = await get_redis_client()
        key = f"auth:blocklist:{jti}"
        await redis.set(key, "1", ex=ttl_seconds)
        return True
    except Exception as e:
        print(f"[Warning] Redis blocklist_access_token failed: {e}")
        return False


async def is_access_token_blocklisted(jti: str) -> bool:
    """Check if access token has been blocklisted via logout."""
    try:
        redis = await get_redis_client()
        key = f"auth:blocklist:{jti}"
        return await redis.exists(key) > 0
    except Exception as e:
        print(f"[Warning] Redis is_access_token_blocklisted failed: {e}")
        return False
