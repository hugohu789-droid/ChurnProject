# ML Platform

A full-stack, self-service machine learning platform for training churn-prediction models and running batch inference — built to production standards.

![CI](https://github.com/hugohu789-droid/ml-platform/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121-009688?logo=fastapi)
![Vue](https://img.shields.io/badge/Vue-3.5-4FC08D?logo=vue.js)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)

---

## Features

| Area | Details |
|---|---|
| **Auth** | JWT access + refresh tokens, bcrypt passwords, route guards |
| **Dataset management** | CSV upload, versioned storage, metadata tracking |
| **Model training** | Async background jobs, scikit-learn / XGBoost / LightGBM, Optuna HPO |
| **Model registry** | Persist trained models with accuracy / recall / precision metrics |
| **Batch inference** | Upload new data → predict → download results CSV |
| **Dashboard** | Live stats: files uploaded, models trained, avg accuracy, recent activity |
| **API** | Versioned REST API (`/api/v1`), OpenAPI docs at `/docs` |
| **Database** | SQLite (dev) → PostgreSQL (prod) via async SQLAlchemy + Alembic |
| **CI/CD** | GitHub Actions: lint → test → build → deploy to EC2 |

---

## Tech Stack

### Backend
- **FastAPI** — async Python web framework
- **SQLAlchemy 2 (async)** — ORM with `asyncpg` (PostgreSQL) or `aiosqlite` (SQLite)
- **Alembic** — database schema migrations
- **python-jose** + **passlib** — JWT auth and bcrypt password hashing
- **pydantic-settings** — type-safe environment configuration
- **scikit-learn**, **XGBoost**, **LightGBM**, **Optuna** — ML pipeline

### Frontend
- **Vue 3** + **TypeScript** — reactive SPA
- **Pinia** — state management (auth store, etc.)
- **Vue Router 4** — client-side routing with navigation guards
- **Axios** — HTTP client with automatic token refresh interceptor
- **Element Plus** — UI component library (dark-themed)
- **Vite** — bundler

### Infrastructure
- **Docker Compose** — local dev and production orchestration
- **PostgreSQL 16** — production database
- **Nginx** — reverse proxy / static file serving
- **AWS EC2** — compute

---

## Project Structure

```
ml-platform/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # Route handlers (auth, datasets, training, models, predictions)
│   │   ├── core/            # Config, database engine, security (JWT)
│   │   ├── db/models/       # SQLAlchemy ORM models
│   │   ├── schemas/         # Pydantic request / response models
│   │   ├── services/        # Business logic (auth, ML training)
│   │   ├── dependencies.py  # FastAPI dependency injection (get_db, get_current_user)
│   │   └── main.py          # App factory with lifespan hooks
│   ├── migrations/          # Alembic migrations
│   ├── tests/               # pytest async test suite
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/             # Axios client + per-resource API modules
│   │   ├── composables/     # Reusable Vue composables (useAuth)
│   │   ├── layouts/         # App / Auth layout wrappers
│   │   ├── router/          # Vue Router with auth guards
│   │   ├── stores/          # Pinia stores (auth)
│   │   ├── types/           # TypeScript interfaces
│   │   └── views/           # Page components (auth/LoginView, auth/RegisterView, …)
│   └── Dockerfile
├── deploy/
│   ├── docker-compose.yml   # Production compose
│   └── nginx.conf
├── docs/
│   └── architecture.md      # System design overview
├── docker-compose.yml        # Local dev (PostgreSQL + backend + frontend)
└── .env.example
```

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Node.js ≥ 20 (for frontend-only dev)
- Python ≥ 3.11 (for backend-only dev)

### 1. Clone and configure

```bash
git clone https://github.com/hugohu789-droid/ml-platform.git
cd ml-platform
cp .env.example .env
cp backend/.env.example backend/.env
```

Edit `backend/.env` — set a strong `SECRET_KEY` and your `DATABASE_URL`.

### 2. Run with Docker Compose (recommended)

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

### 3. Backend only (local Python)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

alembic upgrade head
uvicorn app.main:app --reload
```

### 4. Frontend only

```bash
cd frontend
npm install
npm run dev
```

---

## Authentication

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/auth/register` | POST | Create a new account |
| `/api/v1/auth/login` | POST | Get access + refresh tokens |
| `/api/v1/auth/refresh` | POST | Rotate tokens |
| `/api/v1/auth/me` | GET | Get current user info |

All other endpoints require `Authorization: Bearer <access_token>`.

---

## Database

The project works with both **SQLite** (dev, zero-config) and **PostgreSQL** (prod) via a single env var:

```bash
# Local dev
DATABASE_URL=sqlite+aiosqlite:///./ml_platform.db

# Production (Neon / Supabase / Railway / self-hosted)
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/ml_platform
```

Run migrations:

```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## CI/CD

GitHub Actions pipeline:

1. **build-backend** — Flake8 lint + pytest (SQLite in CI)
2. **build-frontend** — TypeScript type-check + Vitest + Vite build
3. **deploy** *(main branch only)* — SCP to EC2, restart Docker Compose

Required GitHub Secrets: `EC2_HOST`, `EC2_USERNAME`, `EC2_SSH_KEY`.

---

## License

MIT
