from datetime import datetime
from pydantic import BaseModel


class TrainRequest(BaseModel):
    id: int
    model_name: str


class TrainModelResponse(BaseModel):
    id: int
    file_id: int
    record_number: int
    accuracy: float | None
    recall_rate: float | None
    precision: float | None
    model_name: str
    train_date: datetime

    model_config = {"from_attributes": True}
