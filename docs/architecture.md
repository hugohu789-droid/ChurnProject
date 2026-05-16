# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Browser (Vue 3 SPA)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Login/  │  │Dashboard │  │ Training │  │  Models / Predict │   │
│  │ Register │  │  View    │  │  View    │  │      Views        │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│       │              │              │                  │             │
│  ┌────▼──────────────▼──────────────▼──────────────────▼──────┐    │
│  │              Pinia Stores  (auth, etc.)                      │    │
│  └───────────────────────────┬──────────────────────────────────┘   │
│                               │                                      │
│  ┌────────────────────────────▼──────────────────────────────────┐  │
│  │        Axios Client (token injection + auto-refresh)           │  │
│  └────────────────────────────┬──────────────────────────────────┘  │
└───────────────────────────────┼─────────────────────────────────────┘
                                │ HTTPS
┌───────────────────────────────▼─────────────────────────────────────┐
│                         FastAPI Backend                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                       API v1 Router                          │    │
│  │  /auth  /dashboard  /datasets  /training  /models  /predict  │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                                │                                      │
│  ┌──────────────┐  ┌───────────▼────────────┐  ┌────────────────┐  │
│  │  Auth Service│  │     ML Service         │  │   Depends      │  │
│  │  (JWT, bcrypt│  │  (train / infer bg job)│  │  (get_db,      │  │
│  │   CRUD user) │  │                        │  │  current_user) │  │
│  └──────┬───────┘  └───────────┬────────────┘  └────────────────┘  │
│         │                       │                                     │
│  ┌──────▼───────────────────────▼────────────────────────────────┐  │
│  │            SQLAlchemy async ORM (asyncpg / aiosqlite)          │  │
│  └───────────────────────────────┬───────────────────────────────┘  │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │          Database            │
                    │  SQLite (dev) / PostgreSQL   │
                    │  (prod — Neon, Supabase,     │
                    │   Railway, or self-hosted)   │
                    └─────────────────────────────┘
```

## Key Design Decisions

### 1. Async-first backend
All database operations use `async/await` via SQLAlchemy's async interface.  
ML training runs in a thread-pool via FastAPI `BackgroundTasks` so it never blocks the event loop.

### 2. Single `DATABASE_URL` for both environments
The driver prefix (`+aiosqlite` vs `+asyncpg`) switches the entire stack.  
Alembic generates migrations from the same ORM models, keeping dev and prod in sync.

### 3. JWT with refresh rotation
- Short-lived access tokens (30 min) protect resources.
- Long-lived refresh tokens (7 days) allow seamless re-auth without re-login.
- The Axios interceptor silently refreshes on 401 and retries the original request.

### 4. Layered backend architecture

```
app/
  api/v1/       ← HTTP layer only (validate input, call service, return response)
  services/     ← Business logic (no HTTP concepts)
  db/models/    ← SQLAlchemy ORM (no business logic)
  schemas/      ← Pydantic I/O contracts
  core/         ← Cross-cutting concerns (config, DB engine, security)
  dependencies/ ← FastAPI DI wiring
```

This separation makes unit testing services trivial — inject a test DB session, no HTTP needed.

### 5. Database migration workflow

```bash
# After changing an ORM model:
alembic revision --autogenerate -m "add user table"
alembic upgrade head
```

Migrations are stored in `backend/migrations/versions/` and committed to git, ensuring reproducible schema evolution across environments.

## Data Flow: Model Training

```
User uploads CSV
      │
  POST /api/v1/datasets/upload
      │  ← file saved to disk, FileUpload row created (status=uploaded)
      ▼
  POST /api/v1/training/train  { id, model_name }
      │  ← FileUpload.status → "training"
      │  ← background task enqueued
      ▼
  BackgroundTasks (thread pool)
      │  ← modeltrain.train_model(file_path, model_save_path)
      │  ← TrainModel row created with metrics
      │  ← FileUpload.status → "trained"
      ▼
  GET /api/v1/models/list  ← user polls or refreshes
```

## Recommended Production Database: Neon (Serverless PostgreSQL)

[Neon](https://neon.tech) provides serverless PostgreSQL with:
- **Free tier** — perfect for portfolio projects
- **Branching** — create a DB branch per PR (like git branches)
- **Auto-suspend** — scales to zero when idle
- **Connection pooling** built-in

Connection string format:
```
postgresql+asyncpg://user:password@ep-xxx.us-east-1.aws.neon.tech/ml_platform?sslmode=require
```
