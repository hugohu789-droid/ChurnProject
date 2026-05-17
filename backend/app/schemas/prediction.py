from datetime import datetime
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    id: int
    train_model_id: int
    train_model_name: str
    result1_path: str
    result2_path: str
    predict_date: datetime
    status: str

    model_config = {"from_attributes": True}
