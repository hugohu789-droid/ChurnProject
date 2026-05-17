"""
Churn prediction ML pipeline.

Provides two public functions used by the training and inference services:
  - train_model(file_path, model_save_path) -> dict[str, float]
  - predict(model_path, data_path, output_path) -> str
"""

import logging
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "churn_model_rf.joblib"
DEFAULT_OUTPUT_PATH = "prediction_results.csv"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_csv(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    logger.info("Loaded %s — shape %s", file_path, df.shape)
    return df


def _clean(df: pd.DataFrame, *, is_training: bool = True) -> pd.DataFrame:
    """Standardise the Telco Churn dataset for training or inference."""
    out = df.copy()

    # TotalCharges arrives as strings with spaces → coerce to float
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0)

    # customerID is an identifier, not a feature
    out = out.drop(columns=["customerID"], errors="ignore")

    if is_training and "Churn" in out.columns:
        out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def train_model(file_path: str, model_save_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Train a Random Forest classifier on the Telco Churn dataset.

    Returns a dict with keys: accuracy, precision, recall, roc_auc.
    Saves the model + feature list to *model_save_path* (.joblib).
    """
    logger.info("Training pipeline started — input: %s", file_path)

    df = _load_csv(file_path)
    df = _clean(df, is_training=True)

    X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight='balanced' compensates for the natural churn-class imbalance
    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced", max_depth=10
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    logger.info(
        "Metrics — acc: %.4f  auc: %.4f  recall: %.4f  precision: %.4f",
        metrics["accuracy"], metrics["roc_auc"], metrics["recall"], metrics["precision"],
    )

    joblib.dump({"model": clf, "features": feature_names}, model_save_path)
    logger.info("Model saved to %s", model_save_path)

    return metrics


def predict(model_path: str, data_path: str, output_path: str = DEFAULT_OUTPUT_PATH) -> str:
    """
    Run batch inference with a saved model.

    Returns the absolute path to the results CSV.
    """
    logger.info("Inference pipeline started — model: %s  data: %s", model_path, data_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    artifact = joblib.load(model_path)
    clf = artifact["model"]
    expected_features: list[str] = artifact["features"]

    df_raw = _load_csv(data_path)
    df_clean = _clean(df_raw, is_training=False)
    df_encoded = pd.get_dummies(df_clean, drop_first=True)

    # Align columns to exactly what the model was trained on
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_final = df_encoded[expected_features]

    predictions = clf.predict(df_final)
    probabilities = clf.predict_proba(df_final)[:, 1]

    results = df_raw.copy()
    results["Predicted_Churn"] = predictions
    results["Churn_Probability"] = probabilities
    results.to_csv(output_path, index=False)

    abs_path = os.path.abspath(output_path)
    logger.info("Results saved to %s", abs_path)
    return abs_path
