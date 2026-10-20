"""
Database session management.
Provides async engine and session factory for FastAPI endpoints,
plus sync engine for Alembic and migrations.
"""

from typing import AsyncGenerator
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Async Engine (for FastAPI request handlers)
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=(settings.DEBUG and settings.ENVIRONMENT != "production"),
    future=True,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,
    pool_timeout=30,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync Engine (for Alembic migrations and sync scripts)
sync_engine = create_engine(
    settings.SYNC_DATABASE_URL,
    echo=(settings.DEBUG and settings.ENVIRONMENT != "production"),
    pool_pre_ping=True,
    pool_recycle=1800,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for obtaining an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """Tests database connectivity by executing SELECT 1."""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception:
        return False
