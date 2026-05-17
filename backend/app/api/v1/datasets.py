import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.dataset import FileUpload
from app.dependencies import get_current_user, get_db
from app.db.models.user import User
from app.schemas.common import PageRequest, PagedResponse
from app.schemas.dataset import FileUploadResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    date_dir = datetime.now().strftime("%Y%m%d")
    upload_dir = Path(settings.UPLOAD_DIR) / date_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    base, ext = os.path.splitext(file.filename)
    dest = upload_dir / file.filename
    counter = 1
    while dest.exists():
        dest = upload_dir / f"{base}_{counter}{ext}"
        counter += 1

    with dest.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    record = FileUpload(
        original_filename=file.filename,
        saved_filename=dest.name,
        file_path=str(dest),
        file_size=dest.stat().st_size,
        status="uploaded",
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


@router.post("/list", response_model=PagedResponse[FileUploadResponse])
async def list_datasets(
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


@router.delete("/{file_id}")
async def delete_dataset(
    file_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(FileUpload).where(FileUpload.id == file_id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="File not found")

    if os.path.exists(record.file_path):
        os.remove(record.file_path)

    await db.delete(record)
    await db.commit()
    return {"message": f"File {file_id} deleted"}
