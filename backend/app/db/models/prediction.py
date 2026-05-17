from datetime import datetime, timezone
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PredictionHistory(Base):
    __tablename__ = "prediction_histories"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    saved_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    train_file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    train_model_id: Mapped[int] = mapped_column(Integer, nullable=False)
    train_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    result1_path: Mapped[str] = mapped_column(String(512), default="")
    result2_path: Mapped[str] = mapped_column(String(512), default="")
    predict_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    status: Mapped[str] = mapped_column(String(50), default="predicting")
