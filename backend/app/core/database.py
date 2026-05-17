from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    pass


# create_async_engine works with both asyncpg (PostgreSQL) and aiosqlite (SQLite)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    # pool settings only apply to PostgreSQL; SQLite ignores them
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create all tables. Called on application startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
