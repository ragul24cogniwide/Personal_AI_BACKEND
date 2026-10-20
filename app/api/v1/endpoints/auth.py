"""
Authentication Endpoints: register, login, refresh, logout, me.
Supports short-lived access tokens, refresh tokens, Argon2 hashing, and Redis revocation.
"""

from datetime import datetime, timezone
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.deps import get_current_user, get_current_user_and_token, get_db
from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from app.models.settings import AssistantSetting
from app.models.user import User, UserPreference, UserProfile
from app.schemas.auth import (
    LogoutRequest,
    MessageResponse,
    TokenRefresh,
    TokenRefreshResponse,
    TokenResponse,
    UserLogin,
    UserRead,
    UserRegister,
)
from app.services.auth_session import (
    blocklist_access_token,
    is_refresh_token_valid,
    revoke_all_user_refresh_tokens,
    revoke_refresh_token,
    store_refresh_token,
)

router = APIRouter()


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    data: UserRegister,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user with email and password.
    Hashes password using Argon2id, generates user profile and default preferences,
    and returns initial short-lived access token and refresh token.
    """
    normalized_email = data.email.lower().strip()

    # Check if user already exists
    existing = await db.execute(select(User).where(User.email == normalized_email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists.",
        )

    # Hash password with Argon2id
    password_hash = get_password_hash(data.password)

    # Create root user
    user = User(
        email=normalized_email,
        password_hash=password_hash,
    )
    db.add(user)
    await db.flush()  # Obtain user.id

    # Create user profile
    preferred_name = data.name.strip() if data.name else normalized_email.split("@")[0]
    profile = UserProfile(
        user_id=user.id,
        name=data.name.strip() if data.name else preferred_name,
        preferred_name=preferred_name,
        communication_style="concise, structured, and insightful",
        personality_preferences={"verbosity": "concise", "empathy": "high"},
    )
    db.add(profile)

    # Create user preferences
    preferences = UserPreference(
        user_id=user.id,
        theme="dark",
        language="en",
        voice="alloy",
        assistant_tone="friendly",
        verbosity="balanced",
        proactivity="medium",
        timezone="UTC",
    )
    db.add(preferences)

    # Create assistant settings
    assistant_settings = AssistantSetting(
        user_id=user.id,
        personality="helpful, empathetic, concise, and adaptive",
        voice_settings={"voice_id": "alloy", "rate": 1.0, "pitch": 0.0},
        proactivity_settings={"suggestions_enabled": True, "daily_briefing": True},
        memory_settings={"auto_store": True, "privacy_mode": False},
    )
    db.add(assistant_settings)

    await db.commit()

    user.profile = profile
    user.preferences = preferences
    user.assistant_settings = assistant_settings

    # Issue tokens
    access_token, _, expires_in = create_access_token(subject=str(user.id))
    refresh_token, refresh_jti, refresh_exp = create_refresh_token(subject=str(user.id))

    # Persist active refresh token in Redis
    await store_refresh_token(str(user.id), refresh_jti, refresh_exp)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate with email and password",
)
async def login(
    data: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user, verify Argon2/bcrypt password hash,
    and return short-lived access token and refresh token.
    """
    normalized_email = data.email.lower().strip()

    stmt = (
        select(User)
        .where(User.email == normalized_email)
        .options(
            selectinload(User.profile),
            selectinload(User.preferences),
            selectinload(User.assistant_settings),
        )
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Issue tokens
    access_token, _, expires_in = create_access_token(subject=str(user.id))
    refresh_token, refresh_jti, refresh_exp = create_refresh_token(subject=str(user.id))

    # Store refresh token in Redis
    await store_refresh_token(str(user.id), refresh_jti, refresh_exp)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        user=user,
    )


@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    summary="Refresh access token using refresh token",
)
async def refresh_tokens(
    data: TokenRefresh,
    db: AsyncSession = Depends(get_db),
):
    """
    Validate refresh token, check revocation status in Redis,
    and issue a new short-lived access token and rotated refresh token.
    """
    try:
        payload = decode_token(data.refresh_token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Ensure token type is refresh
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type: refresh token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("sub")
    old_jti = payload.get("jti")

    if not user_id_str or not old_jti:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token structure.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check revocation in Redis
    if not await is_refresh_token_valid(user_id_str, old_jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user exists
    try:
        user_uuid = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed user ID.",
        )

    user = (await db.execute(select(User).where(User.id == user_uuid))).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
        )

    # Revoke old refresh token (Token rotation)
    await revoke_refresh_token(user_id_str, old_jti)

    # Issue new access token and rotated refresh token
    new_access_token, _, expires_in = create_access_token(subject=user_id_str)
    new_refresh_token, new_jti, new_refresh_exp = create_refresh_token(subject=user_id_str)

    # Store new refresh token in Redis
    await store_refresh_token(user_id_str, new_jti, new_refresh_exp)

    return TokenRefreshResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Log out and invalidate session tokens",
)
async def logout(
    data: Optional[LogoutRequest] = None,
    user_and_token: tuple = Depends(get_current_user_and_token),
):
    """
    Log out the user. Blocklists the access token JTI and invalidates
    the refresh token in Redis so it cannot be reused.
    """
    user, payload = user_and_token
    user_id_str = str(user.id)

    # 1. Blocklist current access token until it expires
    access_jti = payload.get("jti")
    exp = payload.get("exp", 0)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    remaining_seconds = max(exp - now_ts, 1)

    if access_jti:
        await blocklist_access_token(access_jti, remaining_seconds)

    # 2. Invalidate refresh token
    if data and data.refresh_token:
        try:
            refresh_payload = decode_token(data.refresh_token)
            refresh_jti = refresh_payload.get("jti")
            if refresh_jti:
                await revoke_refresh_token(user_id_str, refresh_jti)
        except Exception:
            pass
    else:
        # If no specific refresh token is provided, invalidate all user refresh tokens
        await revoke_all_user_refresh_tokens(user_id_str)

    return MessageResponse(message="Successfully logged out.")


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get current authenticated user profile and settings",
)
async def get_me(
    current_user: User = Depends(get_current_user),
):
    """
    Protected endpoint: Returns current authenticated user with profile,
    preferences, and assistant settings.
    """
    return current_user
