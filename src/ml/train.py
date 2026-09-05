"""
Trains and compares candidate credit-risk models, selects the best one by
evidence (not assumption), and saves the winning pipeline + feature metadata
for use by inference, the UI, and explainability.

Usage:
    python -m src.ml.train

Reads settings.RAW_DATA_PATH / settings.DATA_SAMPLE_SIZE, writes:
    settings.MODEL_PATH               (trained sklearn Pipeline, joblib)
    settings.FEATURE_METADATA_PATH    (JSON: feature lists, chosen model, metrics)
    <MODEL_DIR>/model_comparison.json (JSON: all 3 candidates' metrics, for README/UI)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.feature_engineering import clean_and_engineer, get_feature_lists
from src.data.loader import DatasetNotFoundError, DatasetValidationError, load_raw_data
from src.ml.evaluate import compute_classification_metrics, select_best_model, time_inference
from src.ml.pipeline import build_model_pipeline
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_candidates(y_train: np.ndarray) -> dict[str, object]:
    """
    Builds the 3 candidate classifiers with imbalance-aware settings.
    class_weight='balanced' / scale_pos_weight handle imbalance without
    resampling, avoiding any risk of train/test leakage from oversampling.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=settings.RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=settings.RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=settings.RANDOM_SEED,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def train_and_select_model(df: pd.DataFrame) -> dict:
    """
    Runs the full train/compare/select workflow on an already-loaded raw
    DataFrame. Separated from main() so tests can call it directly on a
    small in-memory DataFrame without touching disk.
    """
    engineered = clean_and_engineer(df, require_target=True)
    numeric_features, categorical_features = get_feature_lists()

    X = engineered[numeric_features + categorical_features]
    y = engineered["TARGET"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=settings.RANDOM_SEED,
    )
    logger.info(
        "Train/test split: %d train rows (%.2f%% positive), %d test rows (%.2f%% positive)",
        len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean(),
    )

    candidates = _build_candidates(y_train.to_numpy())
    results: dict[str, dict] = {}
    fitted_pipelines: dict[str, object] = {}

    for name, classifier in candidates.items():
        logger.info("Training candidate model: %s", name)
        pipeline = build_model_pipeline(classifier, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test.to_numpy(), y_prob)
        metrics["inference_ms_per_row"] = time_inference(pipeline, X_test.iloc[:100])

        results[name] = metrics
        fitted_pipelines[name] = pipeline
        logger.info("%s metrics: %s", name, metrics)

    best_name = select_best_model(results)
    best_pipeline = fitted_pipelines[best_name]

    # Log the predicted-probability distribution for the chosen model, to
    # document (not assume) that the configured risk thresholds are sensible.
    best_probs = best_pipeline.predict_proba(X_test)[:, 1]
    percentiles = {p: round(float(np.percentile(best_probs, p)), 4) for p in (50, 75, 90, 95, 99)}
    logger.info(
        "Selected model '%s' predicted-probability percentiles on test set: %s "
        "(compare against MEDIUM_RISK_THRESHOLD=%.2f, HIGH_RISK_THRESHOLD=%.2f)",
        best_name, percentiles, settings.MEDIUM_RISK_THRESHOLD, settings.HIGH_RISK_THRESHOLD,
    )

    return {
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "all_results": results,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "probability_percentiles": percentiles,
        "n_train_rows": len(X_train),
        "n_test_rows": len(X_test),
    }


def save_artifacts(training_output: dict) -> None:
    """Persists the winning pipeline and feature/metrics metadata to disk."""
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(training_output["best_pipeline"], settings.MODEL_PATH)
    logger.info("Saved trained pipeline to %s", settings.MODEL_PATH)

    feature_metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model_name": training_output["best_model_name"],
        "numeric_features": training_output["numeric_features"],
        "categorical_features": training_output["categorical_features"],
        "n_train_rows": training_output["n_train_rows"],
        "n_test_rows": training_output["n_test_rows"],
        "test_metrics": training_output["all_results"][training_output["best_model_name"]],
        "probability_percentiles": training_output["probability_percentiles"],
        "risk_thresholds": {
            "medium": settings.MEDIUM_RISK_THRESHOLD,
            "high": settings.HIGH_RISK_THRESHOLD,
        },
        "random_seed": settings.RANDOM_SEED,
    }
    with open(settings.FEATURE_METADATA_PATH, "w") as f:
        json.dump(feature_metadata, f, indent=2)
    logger.info("Saved feature metadata to %s", settings.FEATURE_METADATA_PATH)

    comparison_path = settings.MODEL_DIR / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(training_output["all_results"], f, indent=2)
    logger.info("Saved model comparison results to %s", comparison_path)


def main() -> None:
    logger.info("Starting training run (DATA_SAMPLE_SIZE=%s)", settings.DATA_SAMPLE_SIZE)
    try:
        df = load_raw_data()
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        logger.error("Training aborted: %s", exc)
        print(f"Training could not start: {exc}")
        sys.exit(1)

    training_output = train_and_select_model(df)
    save_artifacts(training_output)

    print(f"\nTraining complete. Best model: {training_output['best_model_name']}")
    print(f"Test metrics: {training_output['all_results'][training_output['best_model_name']]}")
    print(f"Artifacts saved to: {settings.MODEL_DIR}")


if __name__ == "__main__":
    main()
