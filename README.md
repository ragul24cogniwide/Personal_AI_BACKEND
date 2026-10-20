# Personal AI Voice Assistant — Backend

Production-ready backend built with **FastAPI**, **PostgreSQL**, **SQLAlchemy**, **Alembic**, **Pydantic**, and **Redis**.

## Architecture

```text
backend/
├── app/
│   ├── main.py                  # FastAPI application entry point & lifecycle
│   ├── api/
│   │   └── v1/
│   │       ├── api.py           # V1 Router aggregator
│   │       └── endpoints/
│   │           └── health.py    # Health check with live DB & Redis ping
│   ├── core/
│   │   ├── config.py            # Pydantic Settings & Render URL normalization
│   │   ├── security.py          # Password hashing and JWT tokens
│   │   └── redis.py             # Redis client, caching, voice state & rate limiting
│   ├── database/
│   │   ├── base.py              # Base model with UUID PKs & UTC timestamps
│   │   └── session.py           # Async and Sync session factories
│   ├── models/                  # 12 SQLAlchemy ORM models
│   │   ├── user.py              # User, Profile, Preference, Goal, Habit, Accounts
│   │   ├── conversation.py      # Conversation, Message
│   │   ├── memory.py            # Long-term & Semantic Memory
│   │   ├── mcp.py               # MCP Servers & Tool Permissions
│   │   └── settings.py          # Assistant Settings
│   └── schemas/                 # Pydantic Request/Response DTOs
│       ├── health.py
│       ├── user.py
│       ├── conversation.py
│       └── memory.py
├── alembic/                     # Database migrations
│   └── versions/
│       └── 0001_initial_schema.py
├── docker-compose.yml           # Local dev orchestrator (Postgres + Redis + App)
├── Dockerfile
└── requirements.txt
```

## Quick Start (Docker)

```bash
cd backend
docker compose up -d
```

Check logs:
```bash
docker compose logs -f backend
```

## Local Development (Python Virtual Environment)

1. **Activate virtualenv**:
   ```bash
   source backend/.venv/bin/activate
   ```

2. **Start PostgreSQL & Redis**:
   ```bash
   docker compose up -d postgres redis
   ```

3. **Run database migrations**:
   ```bash
   cd backend
   alembic upgrade head
   ```

4. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **API Documentation**:
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health check: [http://localhost:8000/health](http://localhost:8000/health)

## Render PostgreSQL Setup

To connect to Render:
1. Copy your Render database connection URL (External or Internal URL).
2. Open `backend/.env` and paste it into `DATABASE_URL`:
   ```env
   DATABASE_URL=postgresql://user:password@dpg-xxxxxx.oregon-postgres.render.com/dbname
   ```
   *(The backend automatically converts `postgres://` or `postgresql://` into `postgresql+asyncpg://` for async queries and `postgresql://` for Alembic migrations)*.
3. Apply migrations to Render:
   ```bash
   alembic upgrade head
   ```
# Personal_AI_BACKEND
