from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "ML Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "change-me-in-production-use-a-long-random-string"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    # Supports both PostgreSQL (production) and SQLite (local dev)
    # PostgreSQL: postgresql+asyncpg://user:password@host:5432/dbname
    # SQLite:     sqlite+aiosqlite:///./ml_platform.db
    DATABASE_URL: str = "sqlite+aiosqlite:///./ml_platform.db"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    # File storage
    UPLOAD_DIR: str = "uploadfiles"
    MODELS_DIR: str = "trained_models"
    PREDICT_DIR: str = "predictfiles"
    RESULTS_DIR: str = "predictresults"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
