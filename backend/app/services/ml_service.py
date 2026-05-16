"""ML training and inference background task logic."""

import os
from datetime import datetime
from sqlalchemy.orm import Session

from app.db.models.dataset import FileUpload
from app.db.models.ml_model import TrainModel
from app.db.models.prediction import PredictionHistory

# Lazy import to avoid loading heavy ML libs on startup
import importlib


def _get_modeltrain():
    return importlib.import_module("modeltrain")


def train_model_background(session_factory, id_value: int, model_name: str) -> None:
    """Run model training in a background thread and persist results."""
    db: Session = session_factory()
    try:
        file_record = db.query(FileUpload).filter(FileUpload.id == id_value).first()
        if not file_record:
            return

        today = datetime.now()
        models_dir = os.path.join("trained_models", today.strftime("%Y%m%d"))
        os.makedirs(models_dir, exist_ok=True)

        base_name = os.path.splitext(file_record.saved_filename)[0]
        model_path = os.path.join(models_dir, f"{base_name}_churn_model.joblib")

        modeltrain = _get_modeltrain()
        metrics = modeltrain.train_model(file_record.file_path, model_save_path=model_path)

        db.add(
            TrainModel(
                file_id=id_value,
                record_number=0,
                accuracy=metrics["accuracy"],
                recall_rate=metrics["recall"],
                precision=metrics["precision"],
                model1_path=model_path,
                model_name=model_name,
            )
        )
        file_record.status = "trained"
        db.commit()
    except Exception as exc:
        db.rollback()
        print(f"[train_model_background] error: {exc}")
    finally:
        db.close()


def predict_model_background(session_factory, predict_id: int, file_path: str, model_id: int) -> None:
    """Run batch inference in a background thread and save results CSV."""
    db: Session = session_factory()
    try:
        predict_record = db.query(PredictionHistory).filter(PredictionHistory.id == predict_id).first()
        if not predict_record:
            return

        today = datetime.now()
        results_dir = os.path.join("predictresults", today.strftime("%Y%m%d"))
        os.makedirs(results_dir, exist_ok=True)

        result_path = os.path.join(results_dir, f"predict_result_{predict_id}.csv")
        predict_record.result1_path = result_path

        model = db.query(TrainModel).filter(TrainModel.id == model_id).first()
        if model:
            modeltrain = _get_modeltrain()
            modeltrain.predict(
                data_path=file_path,
                model_path=model.model1_path,
                output_path=result_path,
            )

        predict_record.status = "completed"
        db.commit()
    except Exception as exc:
        print(f"[predict_model_background] error: {exc}")
    finally:
        db.close()
