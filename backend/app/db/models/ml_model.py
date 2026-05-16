from datetime import datetime, timezone
from sqlalchemy import DateTime, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TrainModel(Base):
    __tablename__ = "train_models"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    file_id: Mapped[int] = mapped_column(Integer, nullable=False)
    record_number: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float] = mapped_column(Numeric(precision=8, scale=6), nullable=True)
    recall_rate: Mapped[float] = mapped_column(Numeric(precision=8, scale=6), nullable=True)
    precision: Mapped[float] = mapped_column(Numeric(precision=8, scale=6), nullable=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model1_path: Mapped[str] = mapped_column(String(512), nullable=True)
    model2_path: Mapped[str] = mapped_column(String(512), nullable=True)
    train_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
