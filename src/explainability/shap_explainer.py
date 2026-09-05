"""
Explainable AI layer: SHAP-based global and local explanations for the
trained credit-risk pipeline, translated into business-readable language.

All explanations are computed from the ACTUAL trained pipeline and the
ACTUAL processed features for the applicant in question -- there is no
fabricated or randomly generated explanation content anywhere in this file.
If SHAP computation fails for any reason (model/version incompatibility,
unexpected input), we fail safely with a clear, loggable error rather than
inventing an explanation.

SIGN CONVENTION (verified against this project's pipeline, do not change
without re-verifying): the model's positive class is TARGET=1 (loan
default). For tree-based models where SHAP returns a two-class list, index
1 (the positive/default class) is selected. For models where SHAP returns a
single array (e.g. XGBoost's raw margin, or LinearExplainer's linear
output), that single array is already monotonically increasing with
P(default). In both cases: POSITIVE SHAP VALUE = pushes the prediction
toward higher default risk. NEGATIVE SHAP VALUE = pushes toward lower risk.
Contribution magnitudes are in the model's internal SHAP/margin units, not
directly in probability percentage points -- callers should present them as
relative influence, not add them arithmetically to a probability.
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

_KNOWN_COLUMNS_CACHE: list[str] | None = None


def _known_columns() -> list[str]:
    """All base + engineered modeling columns, longest-first so a specific
    column like 'NAME_INCOME_TYPE' is matched before any shorter false-prefix."""
    global _KNOWN_COLUMNS_CACHE
    if _KNOWN_COLUMNS_CACHE is None:
        numeric_features, categorical_features = get_feature_lists()
        _KNOWN_COLUMNS_CACHE = sorted(numeric_features + categorical_features, key=len, reverse=True)
    return _KNOWN_COLUMNS_CACHE


def _match_base_column(raw_name: str) -> tuple[str, str | None]:
    """
    Matches a post-ColumnTransformer raw feature name (e.g. 'AMT_INCOME_TOTAL'
    or 'NAME_FAMILY_STATUS_Married') back to its base modeling column and,
    for one-hot encoded categoricals, the category suffix.

    Returns: (base_column, category_suffix_or_None)
    """
    for col in _known_columns():
        if raw_name == col:
            return col, None
        if raw_name.startswith(col + "_"):
            return col, raw_name[len(col) + 1:]
    return raw_name, None


def _humanize_feature_name(transformed_name: str) -> str:
    """
    Converts a post-ColumnTransformer feature name (e.g.
    'numeric__AMT_INCOME_TOTAL' or 'categorical__CODE_GENDER_F') into a
    business-readable label.
    """
    raw_name = transformed_name.split("__", 1)[-1]
    base_col, suffix = _match_base_column(raw_name)
    label = _BUSINESS_LABELS.get(base_col, base_col.replace("_", " ").title())
    if suffix:
        return f"{label} = {suffix}"
    return label


def format_feature_value(raw_column: str, value: object) -> str:
    """
    Formats a raw/engineered feature value for display, using business-aware
    rules (currency for AMT_* columns, years for age/tenure, Yes/No for
    Y/N flags, plain decimals for scores/ratios). Shared by per-applicant
    value display and dataset-median display so both use identical rules.
    """
    if value is None:
        return "N/A"
    try:
        if isinstance(value, float) and value != value:  # NaN
            return "Not available"
    except TypeError:
        pass

    if raw_column in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        return "Yes" if str(value).strip().upper() == "Y" else "No"
    if raw_column == "IS_RETIRED_OR_UNEMPLOYED_FLAG":
        try:
            return "Yes" if int(value) == 1 else "No"
        except (TypeError, ValueError):
            return str(value)
    if raw_column.startswith("AMT_"):
        try:
            return f"\u20b9{float(value):,.0f}"
        except (TypeError, ValueError):
            return str(value)
    if raw_column in ("AGE_YEARS", "EMPLOYED_YEARS"):
        try:
            return f"{float(value):.1f} years"
        except (TypeError, ValueError):
            return str(value)
    if raw_column.startswith("EXT_SOURCE") or raw_column in (
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "REGION_POPULATION_RELATIVE",
    ):
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)
    if raw_column in (
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ):
        try:
            return f"{float(value):.0f}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{float(value):,.2f}"
    return str(value)


def _get_shap_explainer(pipeline):
    """Selects an appropriate SHAP explainer type based on the underlying model."""
    classifier = pipeline.named_steps["classifier"]

    if isinstance(classifier, LogisticRegression):
        # LinearExplainer needs a background distribution; supplied at call time.
        return "linear", classifier
    else:
        # Tree-based models (RandomForest, XGBoost) support the fast, exact TreeExplainer.
        return "tree", classifier


def prediction_confidence(probability: float) -> float:
    """
    Heuristic confidence score in [0, 1]: distance from the 0.5 classification
    threshold, scaled so 0 = maximum uncertainty (probability exactly 0.5) and
    1 = maximum certainty (probability 0 or 1).

    This is NOT a calibrated statistical confidence interval -- it is a
    simple, honestly-labeled distance-from-threshold heuristic. Callers must
    present it as such.
    """
    return round(abs(float(probability) - 0.5) * 2, 4)


def get_engineered_background(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the fully engineered feature matrix (same columns the model
    consumes) for a raw applicant DataFrame -- reusable for both SHAP
    background sampling and dataset-level statistical context (percentiles,
    medians) without duplicating feature-engineering calls.
    """
    numeric_features, categorical_features = get_feature_lists()
    engineered = clean_and_engineer(raw_df, require_target=False)
    return engineered[numeric_features + categorical_features]


def compute_numeric_context(engineered_background: pd.DataFrame, column: str, applicant_value: object) -> dict | None:
    """
    Computes real, dataset-derived statistical context for one numeric
    column: the dataset median and the applicant's percentile rank.

    Percentile (rather than a z-score) is used deliberately -- several of
    this dataset's numeric columns are skewed, and a percentile makes no
    assumption of normality. Returns None if the column isn't numeric/
    present, or the applicant's value is missing, so callers can skip
    displaying context rather than showing a misleading placeholder.
    """
    if column not in engineered_background.columns:
        return None
    series = pd.to_numeric(engineered_background[column], errors="coerce").dropna()
    if series.empty:
        return None
    try:
        value = float(applicant_value)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None

    median = float(series.median())
    percentile = float((series < value).mean() * 100)
    return {
        "column": column,
        "median": median,
        "percentile": round(percentile, 1),
        "n_samples": int(len(series)),
    }


def _compute_global_importance_all(background_df: pd.DataFrame) -> list[dict]:
    """
    Computes global feature importance (mean |SHAP value|) across a
    representative sample of applicants, using the actual trained pipeline.
    Returns ALL features, sorted descending -- callers slice as needed.
    """
    try:
        pipeline = _load_pipeline()
        numeric_features, categorical_features = get_feature_lists()
        engineered = clean_and_engineer(background_df, require_target=False)
        X = engineered[numeric_features + categorical_features]

        preprocessor = pipeline.named_steps["preprocessor"]
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
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = X_transformed[
                rng.choice(X_transformed.shape[0], size=min(100, X_transformed.shape[0]), replace=False)
            ]
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_sample)

        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_abs = np.asarray(mean_abs).flatten()

        ranked_idx = np.argsort(mean_abs)[::-1]
        return [
            {
                "feature": feature_names[i],
                "business_label": _humanize_feature_name(feature_names[i]),
                "mean_abs_shap_value": round(float(mean_abs[i]), 5),
            }
            for i in ranked_idx
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error("Global SHAP importance computation failed: %s", exc)
        raise ExplanationUnavailableError(
            "Global explanation could not be computed for the current model."
        ) from exc


def compute_full_global_importance(background_df: pd.DataFrame) -> list[dict]:
    """Public entry point returning ALL features' global importance, sorted descending."""
    return _compute_global_importance_all(background_df)


def global_explanation(background_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """
    Backward-compatible wrapper: returns only the top_n most important
    features. Signature and return shape are unchanged from prior versions.
    """
    return _compute_global_importance_all(background_df)[:top_n]


def local_explanation(applicant: dict, top_n: int = 5) -> dict:
    """
    Computes a local (per-applicant) explanation: top factors increasing and
    decreasing predicted risk, based on the actual trained model and the
    actual engineered features for this applicant.

    Returns (return shape is backward-compatible; each contribution dict now
    additionally includes 'raw_column', 'applicant_value_raw', and
    'applicant_value_display'):
        {
          "top_risk_increasing": [{"business_label": ..., "shap_value": ..., ...}, ...],
          "top_risk_decreasing": [{"business_label": ..., "shap_value": ..., ...}, ...],
        }
    """
    try:
        pipeline = _load_pipeline()
        numeric_features, categorical_features = get_feature_lists()
        raw_df = pd.DataFrame([applicant])
        engineered = clean_and_engineer(raw_df, require_target=False)
        engineered_row = engineered.iloc[0]
        X = engineered[numeric_features + categorical_features]

        preprocessor = pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(X)
        feature_names = preprocessor.get_feature_names_out()

        explainer_type, model = _get_shap_explainer(pipeline)
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            background = np.zeros((1, X_transformed.shape[1]))
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_transformed)

        row_values = np.asarray(shap_values).flatten()

        contributions = []
        for i in range(len(feature_names)):
            raw_name = feature_names[i].split("__", 1)[-1]
            raw_col, _suffix = _match_base_column(raw_name)
            applicant_raw_value = engineered_row.get(raw_col) if raw_col in engineered_row.index else None
            contributions.append({
                "feature": feature_names[i],
                "business_label": _humanize_feature_name(feature_names[i]),
                "shap_value": round(float(row_values[i]), 5),
                "raw_column": raw_col,
                "applicant_value_raw": applicant_raw_value,
                "applicant_value_display": format_feature_value(raw_col, applicant_raw_value),
            })

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