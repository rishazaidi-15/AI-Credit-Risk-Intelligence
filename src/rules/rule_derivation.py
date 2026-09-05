"""
Business rule derivation: extracts human-readable decision rules that
approximate the trained model's behavior, using a shallow surrogate decision
tree. These are explicitly documented as MODEL-DERIVED INSIGHTS, not
official credit policy -- the module never claims otherwise.

Approach: train a small (max_depth<=4) DecisionTreeClassifier on the
NUMERIC subset of the same engineered features, predicting the model's own
risk band (LOW/MEDIUM/HIGH) rather than the raw TARGET. This explains "what
does the model tend to do" in simple, auditable IF-THEN language grounded
in real feature thresholds -- not the ground-truth default outcome itself.
Restricting rule extraction to numeric features keeps the extracted
conditions short and business-readable; categorical drivers are already
covered by SHAP local/global explanations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.data.feature_engineering import clean_and_engineer, get_feature_lists
from src.ml.predict import predict_batch
from src.ml.risk import HIGH_RISK, LOW_RISK, MEDIUM_RISK, RISK_BAND_ORDER
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ORDINAL_TO_BAND = {v: k for k, v in RISK_BAND_ORDER.items()}

MAX_TREE_DEPTH = 3
MIN_SAMPLES_PER_RULE = 20


class RuleDerivationError(RuntimeError):
    """Raised when rule extraction cannot be completed."""


def _extract_rules_from_tree(tree_model: DecisionTreeClassifier, feature_names: list[str]) -> list[dict]:
    """Walks every root-to-leaf path in a fitted sklearn tree and returns a readable rule per leaf."""
    tree_ = tree_model.tree_
    rules: list[dict] = []

    def recurse(node: int, conditions: list[str]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = round(float(tree_.threshold[node]), 2)

            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold}"])
        else:
            sample_count = int(tree_.n_node_samples[node])
            if sample_count < MIN_SAMPLES_PER_RULE:
                return  # skip leaves with too little support to be a trustworthy rule

            class_counts = tree_.value[node][0]
            predicted_ordinal = int(np.argmax(class_counts))
            predicted_band = _ORDINAL_TO_BAND.get(predicted_ordinal, "UNKNOWN")
            purity = round(float(class_counts[predicted_ordinal] / class_counts.sum()), 3)

            rules.append({
                "conditions": conditions,
                "predicted_risk_band": predicted_band,
                "supporting_samples": sample_count,
                "purity": purity,
            })

    recurse(0, [])
    # Most useful rules first: high sample support and high purity.
    rules.sort(key=lambda r: (r["supporting_samples"] * r["purity"]), reverse=True)
    return rules


def derive_rules(raw_df: pd.DataFrame, max_rules: int = 8) -> list[dict]:
    """
    Trains a surrogate decision tree on the model's own predicted risk bands
    and extracts business-readable rules from it.

    Args:
        raw_df: raw applicant data (e.g. a sample of the dataset) to derive
            rules from. Must be large enough to produce leaves with
            MIN_SAMPLES_PER_RULE+ support.
        max_rules: maximum number of rules to return.

    Returns:
        List of rule dicts: conditions, predicted_risk_band, supporting_samples,
        purity, and a rendered human-readable "rule_text".
    """
    try:
        numeric_features, _ = get_feature_lists()
        engineered = clean_and_engineer(raw_df, require_target=False)
        X_numeric = engineered[numeric_features].fillna(engineered[numeric_features].median())

        predictions = predict_batch(raw_df)
        risk_ordinal = predictions["risk_band"].map(RISK_BAND_ORDER)

        surrogate = DecisionTreeClassifier(
            max_depth=MAX_TREE_DEPTH,
            min_samples_leaf=MIN_SAMPLES_PER_RULE,
            random_state=settings.RANDOM_SEED,
        )
        surrogate.fit(X_numeric, risk_ordinal)

        rules = _extract_rules_from_tree(surrogate, numeric_features)[:max_rules]
        for rule in rules:
            condition_text = " AND ".join(rule["conditions"]) if rule["conditions"] else "(no conditions -- root)"
            rule["rule_text"] = (
                f"IF {condition_text} "
                f"THEN applicant is more likely to fall into {rule['predicted_risk_band']} RISK "
                f"(based on {rule['supporting_samples']} similar applicants, "
                f"{rule['purity'] * 100:.0f}% consistent)"
            )

        logger.info("Derived %d business rules from surrogate tree.", len(rules))
        return rules
    except Exception as exc:  # noqa: BLE001
        logger.error("Rule derivation failed: %s", exc)
        raise RuleDerivationError("Business rules could not be derived at this time.") from exc


RULE_DISCLAIMER = (
    "These are data-driven, model-derived decision insights extracted from a "
    "simplified surrogate model. They approximate -- but do not exactly "
    "reproduce -- the full model's behavior, and they are NOT official "
    "credit policy. They should support, not replace, human underwriting judgment."
)
