"""
Inference layer. This is the ONLY place that loads the trained pipeline and
runs predictions -- the UI, database layer, and any future API must call
through here rather than re-loading joblib files or duplicating feature
logic themselves. This is what keeps prediction behavior identical across
the whole application.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src.data.feature_engineering import clean_and_engineer, get_feature_lists
from src.ml.risk import get_risk_band
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when a prediction is requested but no trained pipeline exists yet."""


@lru_cache(maxsize=1)
def _load_pipeline():
    if not settings.MODEL_PATH.exists():
        message = (
            f"No trained model found at {settings.MODEL_PATH}. "
            f"Run training first: python -m src.ml.train"
        )
        logger.error(message)
        raise ModelNotTrainedError(message)
    logger.info("Loading trained pipeline from %s", settings.MODEL_PATH)
    return joblib.load(settings.MODEL_PATH)


@lru_cache(maxsize=1)
def _load_feature_metadata() -> dict:
    if not settings.FEATURE_METADATA_PATH.exists():
        message = f"No feature metadata found at {settings.FEATURE_METADATA_PATH}."
        logger.error(message)
        raise ModelNotTrainedError(message)
    with open(settings.FEATURE_METADATA_PATH) as f:
        return json.load(f)


def clear_model_cache() -> None:
    """Call after retraining so the app picks up the new artifacts without a restart."""
    _load_pipeline.cache_clear()
    _load_feature_metadata.cache_clear()


def is_model_available() -> bool:
    return settings.MODEL_PATH.exists() and settings.FEATURE_METADATA_PATH.exists()


def get_feature_metadata() -> dict:
    return _load_feature_metadata()


def _prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    numeric_features, categorical_features = get_feature_lists()
    engineered = clean_and_engineer(raw_df, require_target=False)
    return engineered[numeric_features + categorical_features]


def predict_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs inference on a DataFrame of applicants (any subset of raw Home
    Credit columns -- missing ones are treated as unknown/NaN).

    Returns a DataFrame with columns: probability, risk_band (plus SK_ID_CURR
    if it was present in the input).
    """
    pipeline = _load_pipeline()
    X = _prepare_features(raw_df)
    probabilities = pipeline.predict_proba(X)[:, 1]
    risk_bands = [get_risk_band(float(p)) for p in probabilities]

    result = pd.DataFrame({
        "probability": probabilities,
        "risk_band": risk_bands,
    })
    if "SK_ID_CURR" in raw_df.columns:
        result.insert(0, "SK_ID_CURR", raw_df["SK_ID_CURR"].values)
    return result


def predict_single(applicant: dict) -> dict:
    """
    Runs inference for one applicant supplied as a dict of raw field values
    (e.g. from a Streamlit form). Unsupplied fields default to unknown (NaN)
    and are handled by the pipeline's imputers -- the same way missing values
    are handled during training, so behavior stays consistent.

    Returns:
        {"probability": float, "risk_band": str}
    """
    raw_df = pd.DataFrame([applicant])
    result = predict_batch(raw_df)
    row = result.iloc[0]
    return {
        "probability": float(row["probability"]),
        "risk_band": str(row["risk_band"]),
    }
