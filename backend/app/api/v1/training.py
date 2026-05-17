from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.dataset import FileUpload
from app.dependencies import get_current_user, get_db
from app.db.models.user import User
from app.schemas.common import PageRequest, PagedResponse
from app.schemas.dataset import FileUploadResponse
from app.schemas.ml_model import TrainRequest
from app.services.ml_service import train_model_background

router = APIRouter(prefix="/training", tags=["training"])


def _sync_session_factory():
    """Returns a sync session factory for use in background threads."""
    from sqlalchemy import create_engine as _ce
    from sqlalchemy.orm import sessionmaker as _sm
    from app.core.config import settings

    sync_url = settings.DATABASE_URL.replace("+asyncpg", "").replace("+aiosqlite", "")
    _engine = _ce(sync_url)
    return _sm(autocommit=False, autoflush=False, bind=_engine)


@router.post("/train")
async def train(
    body: TrainRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(FileUpload).where(FileUpload.id == body.id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found")

    record.status = "training"
    await db.commit()

    background_tasks.add_task(
        train_model_background,
        _sync_session_factory(),
        body.id,
        body.model_name,
    )
    return {"id": body.id, "model_name": body.model_name, "status": "training"}


@router.post("/list", response_model=PagedResponse[FileUploadResponse])
async def list_training_files(
    body: PageRequest,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    total_result = await db.execute(select(func.count()).select_from(FileUpload))
    total = total_result.scalar_one()

    offset = (body.page - 1) * body.page_size
    result = await db.execute(
        select(FileUpload).order_by(FileUpload.upload_time.desc()).offset(offset).limit(body.page_size)
    )
    records = result.scalars().all()

    return PagedResponse(page=body.page, page_size=body.page_size, total=total, records=records)
