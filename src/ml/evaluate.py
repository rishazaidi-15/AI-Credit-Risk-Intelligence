"""
Evaluation metrics for the credit-risk binary classifier.

Because the target is heavily imbalanced (most applicants do not default),
accuracy is intentionally excluded from the primary decision metrics --
ROC-AUC and PR-AUC (more informative under imbalance) drive model selection,
alongside minority-class recall and inference speed.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DECISION_THRESHOLD = 0.5


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> dict:
    """
    Computes the full metric set for one model's predictions on a held-out set.

    Args:
        y_true: ground-truth binary labels.
        y_prob: predicted probability of the positive class (default).
        decision_threshold: threshold for converting probabilities to hard
            labels for precision/recall/F1/confusion matrix. ROC-AUC and
            PR-AUC are threshold-independent.
    """
    y_pred = (y_prob >= decision_threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "decision_threshold": decision_threshold,
    }


def time_inference(pipeline, X_sample) -> float:
    """Returns average per-row inference latency in milliseconds, for model comparison."""
    start = time.perf_counter()
    pipeline.predict_proba(X_sample)
    elapsed = time.perf_counter() - start
    return round((elapsed / max(len(X_sample), 1)) * 1000, 4)


def select_best_model(results: dict[str, dict]) -> str:
    """
    Chooses the best model from a {model_name: metrics_dict} mapping.

    Selection rule (documented, not assumed): primary criterion is PR-AUC
    (most informative under class imbalance), tie-broken by minority-class
    recall, then by inference speed (lower is better).
    """
    ranked = sorted(
        results.items(),
        key=lambda item: (
            -item[1]["pr_auc"],
            -item[1]["recall"],
            item[1].get("inference_ms_per_row", 0.0),
        ),
    )
    best_name = ranked[0][0]
    logger.info(
        "Model selection ranking (by PR-AUC, then recall, then speed): %s -> selected: %s",
        [name for name, _ in ranked], best_name,
    )
    return best_name
