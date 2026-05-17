from fastapi import APIRouter

from app.api.v1 import auth, dashboard, datasets, ml_models, predictions, training

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(dashboard.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(ml_models.router)
api_router.include_router(predictions.router)
