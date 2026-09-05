"""
Tests for the ML training, risk classification, and inference pipeline.

Run with: pytest tests/test_ml.py -v

Note: test_train_and_select_model_end_to_end trains real (small) models on
synthetic data -- it's slower than a unit test but is the most meaningful
verification that training, evaluation, and model selection genuinely work
together end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.evaluate import compute_classification_metrics, select_best_model
from src.ml.risk import HIGH_RISK, LOW_RISK, MEDIUM_RISK, get_risk_band
from src.ml.train import train_and_select_model
from tests.generate_sample_data import generate_sample_dataframe


# -----------------------------------------------------------------------------
# Risk band threshold tests
# -----------------------------------------------------------------------------

def test_risk_band_low():
    assert get_risk_band(0.0) == LOW_RISK
    assert get_risk_band(0.10) == LOW_RISK
    assert get_risk_band(0.299) == LOW_RISK


def test_risk_band_medium():
    assert get_risk_band(0.30) == MEDIUM_RISK
    assert get_risk_band(0.45) == MEDIUM_RISK
    assert get_risk_band(0.599) == MEDIUM_RISK


def test_risk_band_high():
    assert get_risk_band(0.60) == HIGH_RISK
    assert get_risk_band(0.85) == HIGH_RISK
    assert get_risk_band(1.0) == HIGH_RISK


def test_risk_band_rejects_out_of_range():
    with pytest.raises(ValueError):
        get_risk_band(1.5)
    with pytest.raises(ValueError):
        get_risk_band(-0.1)


def test_risk_band_rejects_nan():
    with pytest.raises(ValueError):
        get_risk_band(float("nan"))


# -----------------------------------------------------------------------------
# Evaluation metric tests
# -----------------------------------------------------------------------------

def test_compute_classification_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.9, 0.95, 0.99])
    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_select_best_model_prefers_higher_pr_auc():
    results = {
        "model_a": {"pr_auc": 0.20, "recall": 0.5, "inference_ms_per_row": 1.0},
        "model_b": {"pr_auc": 0.35, "recall": 0.4, "inference_ms_per_row": 2.0},
    }
    assert select_best_model(results) == "model_b"


def test_select_best_model_tiebreaks_on_recall():
    results = {
        "model_a": {"pr_auc": 0.30, "recall": 0.4, "inference_ms_per_row": 1.0},
        "model_b": {"pr_auc": 0.30, "recall": 0.6, "inference_ms_per_row": 1.0},
    }
    assert select_best_model(results) == "model_b"


# -----------------------------------------------------------------------------
# End-to-end training test (slower, but verifies real integration)
# -----------------------------------------------------------------------------

def test_train_and_select_model_end_to_end():
    df = generate_sample_dataframe(n_rows=800, random_seed=7)
    output = train_and_select_model(df)

    assert output["best_model_name"] in ("logistic_regression", "random_forest", "xgboost")
    assert output["best_pipeline"] is not None
    assert 0.0 <= output["all_results"][output["best_model_name"]]["roc_auc"] <= 1.0

    # The fitted pipeline must actually be usable for prediction.
    from src.data.feature_engineering import clean_and_engineer, get_feature_lists
    numeric_features, categorical_features = get_feature_lists()
    engineered = clean_and_engineer(df, require_target=True)
    X = engineered[numeric_features + categorical_features].head(5)
    probs = output["best_pipeline"].predict_proba(X)[:, 1]
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)
