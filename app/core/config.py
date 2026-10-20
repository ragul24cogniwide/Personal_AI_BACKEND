"""
Core configuration and settings using Pydantic Settings.
Normalizes database URLs for asyncpg and sync psycopg2, supporting Render PostgreSQL.
"""

from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Sage"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Security
    SECRET_KEY: str = "super-secret-production-quality-ai-assistant-jwt-signing-key-32bytes-min"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15  # Short-lived access token (15 mins)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7     # Refresh token (7 days)

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/personal_assistant_db"
    SYNC_DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/personal_assistant_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    RATE_LIMIT_PER_MINUTE: int = 120

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Voice Layer - Providers Configuration
    # Supported STT: "sarvam", "deepgram", "openai"
    STT_PROVIDER: str = "sarvam"
    # Supported TTS: "sarvam", "elevenlabs", "openai"
    TTS_PROVIDER: str = "sarvam"

    # Sarvam AI Configuration (Speech-to-Text & Text-to-Speech)
    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_STT_MODEL: str = "saaras:v3"  # saaras:v3, saaras:v4, saarika:v2.5
    SARVAM_TTS_MODEL: str = "bulbul:v3"  # bulbul:v3, bulbul:v4-flash
    SARVAM_TTS_SPEAKER: str = "shubh"    # shubh, aditya, ritu, priya, etc.
    SARVAM_LANGUAGE_CODE: str = "en-IN"  # en-IN, hi-IN, etc.
    SARVAM_TTS_PACE: float = 1.0         # 0.5 to 2.0
    SARVAM_TTS_PITCH: float = 0.0        # -1.0 to 1.0
    SARVAM_TTS_LOUDNESS: float = 1.0     # 0.5 to 2.0
    SARVAM_TTS_SAMPLE_RATE: int = 16000  # 8000, 16000, 22050
    SARVAM_TTS_MAX_CHUNK_CHARS: int = 500 # Sentence split threshold for smooth streaming

    # AI / LLM Brain Layer Configuration (Gemini as Primary LLM)
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_MAX_OUTPUT_TOKENS: int = 2048
    GEMINI_TIMEOUT_SECONDS: float = 30.0

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_async_db_url(cls, v: str) -> str:
        if not v:
            return v
        # Protocol normalization
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+asyncpg://", 1)
        elif v.startswith("postgresql://") and not v.startswith("postgresql+asyncpg://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Clean query parameters for asyncpg (asyncpg expects ssl=require, not sslmode=require)
        try:
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            parsed = urlparse(v)
            if parsed.query:
                query_params = parse_qs(parsed.query)
                if "sslmode" in query_params:
                    ssl_val = query_params.pop("sslmode")[0]
                    query_params["ssl"] = ["require" if ssl_val in ("require", "verify-ca", "verify-full") else ssl_val]
                for unsupported in ["channel_binding", "gssencmode", "target_session_attrs"]:
                    query_params.pop(unsupported, None)
                cleaned_query = urlencode(query_params, doseq=True)
                v = urlunparse(parsed._replace(query=cleaned_query))
        except Exception:
            pass
        return v

    @field_validator("SYNC_DATABASE_URL", mode="before")
    @classmethod
    def assemble_sync_db_url(cls, v: str, info) -> str:
        if not v:
            raw_db = info.data.get("DATABASE_URL", "")
            if raw_db:
                v = raw_db.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
            else:
                return v
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql+psycopg2://", 1)
        elif v.startswith("postgresql+asyncpg://"):
            v = v.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
        elif v.startswith("postgresql://") and not v.startswith("postgresql+psycopg2://"):
            v = v.replace("postgresql://", "postgresql+psycopg2://", 1)
        return v
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return self.DATABASE_URL

settings = Settings()
