from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ml_model import TrainModel
from app.dependencies import get_current_user, get_db
from app.db.models.user import User
from app.schemas.common import PageRequest, PagedResponse
from app.schemas.ml_model import TrainModelResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/{model_id}", response_model=TrainModelResponse)
async def get_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(TrainModel).where(TrainModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/list", response_model=PagedResponse[TrainModelResponse])
async def list_models(
    body: PageRequest,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    total_result = await db.execute(select(func.count()).select_from(TrainModel))
    total = total_result.scalar_one()

    offset = (body.page - 1) * body.page_size
    result = await db.execute(
        select(TrainModel).order_by(TrainModel.train_date.desc()).offset(offset).limit(body.page_size)
    )
    records = result.scalars().all()

    return PagedResponse(page=body.page, page_size=body.page_size, total=total, records=records)
