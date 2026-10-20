"""
Security, hashing, and token utilities using Argon2 and PyJWT.
Never store passwords in plain text.
Provides short-lived access tokens and refresh tokens with JTI tracking.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple, Union
import uuid
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
import jwt
from app.core.config import settings

# Initialize Argon2id password hasher (OWASP recommended memory-hard hashing)
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def get_password_hash(password: str) -> str:
    """Hash password using Argon2id."""
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against Argon2id hash with fallback for bcrypt hashes."""
    if not hashed_password or not plain_password:
        return False

    # Check for Argon2 hash
    if hashed_password.startswith("$argon2"):
        try:
            return _hasher.verify(hashed_password, plain_password)
        except (VerifyMismatchError, VerificationError):
            return False
        except Exception:
            return False

    # Fallback: check for bcrypt hash
    if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            import bcrypt
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception:
            return False

    return False


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[dict] = None,
) -> Tuple[str, str, int]:
    """
    Create a short-lived JWT access token.
    Returns: (token_str, jti, expires_in_seconds)
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
        expires_in = int(expires_delta.total_seconds())
    else:
        expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        expire = now + timedelta(seconds=expires_in)

    jti = uuid.uuid4().hex
    payload = {
        "sub": str(subject),
        "type": "access",
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": expire,
    }
    if extra_claims:
        payload.update(extra_claims)

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token, jti, expires_in


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, str, int]:
    """
    Create a long-lived JWT refresh token.
    Returns: (token_str, jti, expires_in_seconds)
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
        expires_in = int(expires_delta.total_seconds())
    else:
        expires_in = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
        expire = now + timedelta(seconds=expires_in)

    jti = uuid.uuid4().hex
    payload = {
        "sub": str(subject),
        "type": "refresh",
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": expire,
    }

    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return token, jti, expires_in


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
        options={"require": ["exp", "sub", "type", "jti"]},
    )
