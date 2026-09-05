"""
Explainable AI layer: SHAP-based global and local explanations for the
trained credit-risk pipeline, translated into business-readable language.

All explanations are computed from the ACTUAL trained pipeline and the
ACTUAL processed features for the applicant in question -- there is no
fabricated or randomly generated explanation content anywhere in this file.
If SHAP computation fails for any reason (model/version incompatibility,
unexpected input), we fail safely with a clear, loggable error rather than
inventing an explanation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from src.data.feature_engineering import clean_and_engineer, get_feature_lists
from src.ml.predict import _load_pipeline  # reuse the single cached model loader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplanationUnavailableError(RuntimeError):
    """Raised when SHAP explanation cannot be computed; caller should show a clean fallback message."""


# Business-readable labels for the raw/engineered feature names. Falls back
# to a lightly formatted version of the raw name if not listed here.
_BUSINESS_LABELS: dict[str, str] = {
    "AMT_INCOME_TOTAL": "Total income",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_ANNUITY": "Loan annuity (installment amount)",
    "AMT_GOODS_PRICE": "Price of goods being financed",
    "CNT_CHILDREN": "Number of children",
    "CNT_FAM_MEMBERS": "Family size",
    "DAYS_BIRTH": "Applicant age",
    "AGE_YEARS": "Applicant age (years)",
    "DAYS_EMPLOYED": "Employment tenure",
    "EMPLOYED_YEARS": "Employment tenure (years)",
    "IS_RETIRED_OR_UNEMPLOYED_FLAG": "Retired / not currently employed",
    "CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-income ratio",
    "CREDIT_TERM": "Loan term (annuity / credit)",
    "EXT_SOURCE_1": "External credit score #1",
    "EXT_SOURCE_2": "External credit score #2",
    "EXT_SOURCE_3": "External credit score #3",
    "EXT_SOURCE_MEAN": "Average external credit score",
    "REGION_RATING_CLIENT": "Region risk rating",
    "REGION_POPULATION_RELATIVE": "Region population density",
    "OWN_CAR_AGE": "Age of owned car",
    "OBS_30_CNT_SOCIAL_CIRCLE": "Social circle observations (30-day)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Social circle defaults (30-day)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit bureau inquiries (past year)",
    "DAYS_LAST_PHONE_CHANGE": "Days since last phone number change",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns real estate",
    "NAME_CONTRACT_TYPE": "Contract type",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_HOUSING_TYPE": "Housing situation",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer type",
    "NAME_TYPE_SUITE": "Accompanied by",
    "WEEKDAY_APPR_PROCESS_START": "Application weekday",
}


def _humanize_feature_name(transformed_name: str) -> str:
    """
    Converts a post-ColumnTransformer feature name (e.g.
    'numeric__AMT_INCOME_TOTAL' or 'categorical__CODE_GENDER_F') into a
    business-readable label.
    """
    raw_name = transformed_name.split("__", 1)[-1]

    # One-hot encoded categorical features look like "CODE_GENDER_F"
    for base_col, label in _BUSINESS_LABELS.items():
        if raw_name == base_col:
            return label
        if raw_name.startswith(base_col + "_"):
            category_value = raw_name[len(base_col) + 1:]
            return f"{label} = {category_value}"

    return raw_name.replace("_", " ").title()


def _get_shap_explainer(pipeline):
    """Selects an appropriate SHAP explainer type based on the underlying model."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    background_transform_fn = preprocessor.transform

    if isinstance(classifier, LogisticRegression):
        # LinearExplainer needs a background distribution; we supply it at call time.
        return "linear", classifier
    else:
        # Tree-based models (RandomForest, XGBoost) support the fast, exact TreeExplainer.
        return "tree", classifier


def global_explanation(background_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Computes global feature importance (mean |SHAP value|) across a
    representative sample of applicants, using the actual trained pipeline.

    Args:
        background_df: raw applicant DataFrame (e.g. a sample of the training
            or full dataset) used as the basis for global importance.
        top_n: number of top features to return.

    Returns:
        List of {feature, business_label, mean_abs_shap_value}, sorted descending.
    """
    try:
        pipeline = _load_pipeline()
        numeric_features, categorical_features = get_feature_lists()
        engineered = clean_and_engineer(background_df, require_target=False)
        X = engineered[numeric_features + categorical_features]

        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        X_transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()

        # Cap background sample size for speed -- global importance is stable
        # well before using the entire dataset.
        sample_size = min(500, X_transformed.shape[0])
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(X_transformed.shape[0], size=sample_size, replace=False)
        X_sample = X_transformed[sample_idx]

        explainer_type, model = _get_shap_explainer(pipeline)
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # Binary classifiers may return a list [class0, class1] or a single array
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = X_transformed[rng.choice(X_transformed.shape[0], size=min(100, X_transformed.shape[0]), replace=False)]
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        # shap_values may be a sparse/matrix type depending on encoder output; densify safely
        mean_abs = np.asarray(mean_abs).flatten()

        ranked_idx = np.argsort(mean_abs)[::-1][:top_n]
        return [
            {
                "feature": feature_names[i],
                "business_label": _humanize_feature_name(feature_names[i]),
                "mean_abs_shap_value": round(float(mean_abs[i]), 5),
            }
            for i in ranked_idx
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error("Global SHAP explanation failed: %s", exc)
        raise ExplanationUnavailableError(
            "Global explanation could not be computed for the current model."
        ) from exc


def local_explanation(applicant: dict, top_n: int = 5) -> dict:
    """
    Computes a local (per-applicant) explanation: top factors increasing and
    decreasing predicted risk, based on the actual trained model and the
    actual engineered features for this applicant.

    Returns:
        {
          "top_risk_increasing": [{"business_label": ..., "shap_value": ...}, ...],
          "top_risk_decreasing": [{"business_label": ..., "shap_value": ...}, ...],
        }
    """
    try:
        pipeline = _load_pipeline()
        numeric_features, categorical_features = get_feature_lists()
        raw_df = pd.DataFrame([applicant])
        engineered = clean_and_engineer(raw_df, require_target=False)
        X = engineered[numeric_features + categorical_features]

        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        X_transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()

        explainer_type, model = _get_shap_explainer(pipeline)
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # For a single-row explanation, use a small zero/mean background.
            background = np.zeros((1, X_transformed.shape[1]))
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_transformed)

        row_values = np.asarray(shap_values).flatten()

        contributions = [
            {
                "feature": feature_names[i],
                "business_label": _humanize_feature_name(feature_names[i]),
                "shap_value": round(float(row_values[i]), 5),
            }
            for i in range(len(feature_names))
        ]

        increasing = sorted(
            [c for c in contributions if c["shap_value"] > 0],
            key=lambda c: c["shap_value"], reverse=True,
        )[:top_n]
        decreasing = sorted(
            [c for c in contributions if c["shap_value"] < 0],
            key=lambda c: c["shap_value"],
        )[:top_n]

        return {"top_risk_increasing": increasing, "top_risk_decreasing": decreasing}
    except Exception as exc:  # noqa: BLE001
        logger.error("Local SHAP explanation failed: %s", exc)
        raise ExplanationUnavailableError(
            "An explanation could not be generated for this applicant's prediction."
        ) from exc
