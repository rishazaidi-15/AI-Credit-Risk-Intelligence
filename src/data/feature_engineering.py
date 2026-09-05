"""
Feature engineering for the Home Credit Default Risk dataset.

Single source of truth for:
  - Which raw columns the model uses (SELECTED_BASE_NUMERIC / SELECTED_BASE_CATEGORICAL)
  - How they're cleaned and engineered into the final modeling features

All transforms here are deterministic, row-wise, and leakage-safe (no
statistics are fit on the data in this module -- fitting happens only inside
the sklearn pipeline in src/ml/pipeline.py, on the training split). This
guarantees training and inference use IDENTICAL feature logic, which is what
keeps the saved pipeline and the UI/chatbot predictions consistent.

Known data quality issue handled here: DAYS_EMPLOYED uses 365243 as a
sentinel/placeholder value for "not currently employed" (see Phase 2's
generic suspicious-value detector, which flags this automatically). We
convert it to NaN and add an explicit flag feature instead of silently
treating it as a real (and wildly implausible) tenure value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# Sentinel value Home Credit uses in DAYS_EMPLOYED for applicants who are not
# currently employed (e.g. retired/unemployed). ~365000 days = ~1000 years.
DAYS_EMPLOYED_SENTINEL = 365243

# --- Base features selected from the raw dataset (documented in README) ---
# Chosen to span demographics, financial, employment, external scoring, and
# credit-bureau signal categories while keeping the model interpretable and
# the SHAP/rule output business-readable (see Phase 0 "Dataset Strategy").
SELECTED_BASE_NUMERIC: list[str] = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT",
    "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    "DAYS_LAST_PHONE_CHANGE",
]

SELECTED_BASE_CATEGORICAL: list[str] = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
]

# --- Engineered numeric features added on top of the base set ---
ENGINEERED_NUMERIC: list[str] = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "EXT_SOURCE_MEAN",
    "IS_RETIRED_OR_UNEMPLOYED_FLAG",
]


def get_feature_lists() -> tuple[list[str], list[str]]:
    """
    Returns (numeric_features, categorical_features) for the FINAL modeling
    feature set (base + engineered). This is the one place every other
    module (training, inference, explainability, rules) should call to know
    which columns the model expects.
    """
    numeric_features = [c for c in SELECTED_BASE_NUMERIC if c != "DAYS_EMPLOYED"] + [
        "DAYS_EMPLOYED"
    ] + ENGINEERED_NUMERIC
    # (DAYS_EMPLOYED kept in the numeric list -- it's cleaned in-place, not dropped)
    numeric_features = list(dict.fromkeys(numeric_features))  # de-dupe, preserve order
    categorical_features = list(SELECTED_BASE_CATEGORICAL)
    return numeric_features, categorical_features


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Ratio that returns NaN (not inf) when the denominator is zero or missing."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def clean_and_engineer(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """
    Applies deterministic cleaning and feature engineering to raw Home Credit
    data. Safe to call on a single-row DataFrame (inference) or the full
    dataset (training) -- no fitted statistics are used here.

    Args:
        df: raw DataFrame containing at least the SELECTED_BASE_* columns.
        require_target: if True, raises if TARGET is missing (use for training).

    Returns:
        DataFrame containing SK_ID_CURR (if present), TARGET (if present),
        and every column returned by get_feature_lists().
    """
    df = df.copy()

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(
            "TARGET column is required for training but was not found in the input data."
        )

    # Ensure every expected base column exists (fill with NaN if a caller passes
    # a partial applicant record, e.g. from a UI form) so downstream logic
    # never KeyErrors on a legitimately optional field. Use np.nan (not
    # pd.NA) so numeric columns stay proper float dtype for sklearn.
    for col in SELECTED_BASE_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in SELECTED_BASE_CATEGORICAL:
        if col not in df.columns:
            df[col] = np.nan

    # --- Known sentinel-value handling: DAYS_EMPLOYED == 365243 ---
    is_sentinel = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    df["IS_RETIRED_OR_UNEMPLOYED_FLAG"] = is_sentinel.astype(int)
    df.loc[is_sentinel, "DAYS_EMPLOYED"] = pd.NA

    # --- Engineered ratios (leakage-safe: purely row-wise arithmetic) ---
    df["CREDIT_INCOME_RATIO"] = _safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = _safe_ratio(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

    numeric_features, categorical_features = get_feature_lists()
    keep_columns = [c for c in (ID_COLUMN, TARGET_COLUMN) if c in df.columns]
    keep_columns += numeric_features + categorical_features

    logger.info(
        "Feature engineering complete: %d rows, %d modeling features "
        "(%d numeric, %d categorical)",
        len(df), len(numeric_features) + len(categorical_features),
        len(numeric_features), len(categorical_features),
    )
    return df[keep_columns]
