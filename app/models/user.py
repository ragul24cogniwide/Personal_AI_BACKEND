"""
User, UserProfile, UserPreferences, UserGoals, UserHabits, ConnectedAccounts domain models.
Tables: users, user_profiles, user_preferences, user_goals, user_habits, connected_accounts
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_onboarded = Column(Boolean, default=False, nullable=False)
    onboarding_completed_at = Column(DateTime(timezone=True), nullable=True)

    # 1:1 Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
    assistant_settings = relationship("AssistantSetting", back_populates="user", uselist=False, cascade="all, delete-orphan")

    # 1:N Relationships
    goals = relationship("UserGoal", back_populates="user", cascade="all, delete-orphan")
    habits = relationship("UserHabit", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    connected_accounts = relationship("ConnectedAccount", back_populates="user", cascade="all, delete-orphan")
    mcp_servers = relationship("MCPServer", back_populates="user", cascade="all, delete-orphan")
    tool_permissions = relationship("ToolPermission", back_populates="user", cascade="all, delete-orphan")


class UserProfile(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "user_profiles"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    name = Column(String(255), nullable=True)
    preferred_name = Column(String(100), nullable=True)
    occupation = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    communication_style = Column(String(100), nullable=True)
    personality_preferences = Column(JSONB, default=dict, nullable=False)
    interests = Column(JSONB, default=list, nullable=False)
    hobbies = Column(JSONB, default=list, nullable=False)
    fitness_interests = Column(JSONB, default=list, nullable=False)
    learning_interests = Column(JSONB, default=list, nullable=False)
    work_interests = Column(JSONB, default=list, nullable=False)
    daily_routine = Column(JSONB, default=dict, nullable=False)

    user = relationship("User", back_populates="profile")


class UserPreference(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "user_preferences"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    theme = Column(String(50), default="dark", nullable=False)
    language = Column(String(50), default="en", nullable=False)
    voice = Column(String(50), default="alloy", nullable=False)
    assistant_tone = Column(String(50), default="friendly", nullable=False)
    verbosity = Column(String(50), default="balanced", nullable=False)
    proactivity = Column(String(50), default="medium", nullable=False)
    timezone = Column(String(100), default="UTC", nullable=False)

    user = relationship("User", back_populates="preferences")


class UserGoal(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "user_goals"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    category = Column(String(100), default="general", index=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    priority = Column(String(50), default="medium", nullable=False)
    status = Column(String(50), default="in_progress", nullable=False)

    user = relationship("User", back_populates="goals")

    __table_args__ = (
        Index("ix_user_goals_user_id_category", "user_id", "category"),
    )


class UserHabit(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "user_habits"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    category = Column(String(100), default="general", index=True, nullable=False)
    name = Column(String(255), nullable=False)
    frequency = Column(String(50), default="daily", nullable=False)
    description = Column(Text, nullable=True)

    user = relationship("User", back_populates="habits")

    __table_args__ = (
        Index("ix_user_habits_user_id_category", "user_id", "category"),
    )


class ConnectedAccount(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "connected_accounts"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    provider = Column(String(50), nullable=False)
    provider_user_id = Column(String(255), nullable=False)
    encrypted_credentials = Column(Text, nullable=True)
    status = Column(String(50), default="active", nullable=False)

    user = relationship("User", back_populates="connected_accounts")

    __table_args__ = (
        Index("ix_connected_accounts_user_provider", "user_id", "provider"),
    )
