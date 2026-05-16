import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.ml_model import TrainModel
from app.db.models.prediction import PredictionHistory
from app.dependencies import get_current_user, get_db
from app.db.models.user import User
from app.schemas.common import PageRequest, PagedResponse
from app.schemas.prediction import PredictionResponse
from app.services.ml_service import predict_model_background

router = APIRouter(prefix="/predictions", tags=["predictions"])

_SAFE_DIR = Path.cwd().resolve()


@router.post("/run")
async def run_prediction(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_id: int = Form(..., alias="modelId"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    date_dir = datetime.now().strftime("%Y%m%d")
    upload_dir = Path(settings.PREDICT_DIR) / date_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    base, ext = os.path.splitext(file.filename)
    dest = upload_dir / file.filename
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{base}_{counter}{ext}"
        counter += 1

    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    model_result = await db.execute(select(TrainModel).where(TrainModel.id == model_id))
    model = model_result.scalar_one_or_none()

    record = PredictionHistory(
        original_filename=file.filename,
        saved_filename=dest.name,
        train_file_path=str(dest),
        train_model_id=model_id,
        train_model_name=model.model_name if model else "",
        status="predicting",
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    from app.api.v1.training import _sync_session_factory
    background_tasks.add_task(
        predict_model_background,
        _sync_session_factory(),
        record.id,
        str(dest),
        model_id,
    )

    return {"id": record.id, "status": "predicting"}


@router.post("/list", response_model=PagedResponse[PredictionResponse])
async def list_predictions(
    body: PageRequest,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    total_result = await db.execute(select(func.count()).select_from(PredictionHistory))
    total = total_result.scalar_one()

    offset = (body.page - 1) * body.page_size
    result = await db.execute(
        select(PredictionHistory)
        .order_by(PredictionHistory.predict_date.desc())
        .offset(offset)
        .limit(body.page_size)
    )
    records = result.scalars().all()

    return PagedResponse(page=body.page, page_size=body.page_size, total=total, records=records)


@router.get("/download")
async def download_result(
    file_path: str = Query(..., description="Relative path to the result CSV"),
    _: User = Depends(get_current_user),
):
    relative = file_path.lstrip(".\\/")
    full_path = (_SAFE_DIR / relative).resolve()

    if not full_path.is_relative_to(_SAFE_DIR):
        raise HTTPException(status_code=403, detail="Access forbidden")
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=full_path, media_type="text/csv", filename=full_path.name)
