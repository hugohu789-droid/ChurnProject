from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.dataset import FileUpload
from app.db.models.ml_model import TrainModel
from app.db.models.prediction import PredictionHistory
from app.dependencies import get_current_user, get_db
from app.db.models.user import User

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
async def dashboard_stats(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    total_files = (await db.execute(select(func.count()).select_from(FileUpload))).scalar_one()
    total_models = (await db.execute(select(func.count()).select_from(TrainModel))).scalar_one()
    total_predictions = (await db.execute(select(func.count()).select_from(PredictionHistory))).scalar_one()

    avg_result = await db.execute(select(func.avg(TrainModel.accuracy)))
    avg_accuracy = avg_result.scalar_one() or 0.0

    recent_result = await db.execute(
        select(TrainModel).order_by(TrainModel.train_date.desc()).limit(5)
    )
    recent_models = [
        {
            "id": m.id,
            "model_name": m.model_name,
            "accuracy": float(m.accuracy) if m.accuracy else 0.0,
            "precision": float(m.precision) if m.precision else 0.0,
            "train_date": m.train_date,
        }
        for m in recent_result.scalars().all()
    ]

    return {
        "total_files": total_files,
        "total_models": total_models,
        "total_predictions": total_predictions,
        "average_accuracy": float(avg_accuracy),
        "recent_models": recent_models,
    }
