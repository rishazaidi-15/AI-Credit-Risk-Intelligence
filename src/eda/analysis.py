"""
Dataset understanding utilities: summary statistics, data quality analysis,
and business-oriented feature categorization.

These functions are pure (DataFrame in, structured result out) so they can be
reused identically by the EDA notebook (Phase 3), the Streamlit EDA page
(Phase 12), and the NL-to-SQL schema description (Phase 10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"

# High-cardinality threshold for categorical columns (business-meaningful default,
# not a hard rule -- documented here rather than buried as a magic number elsewhere).
HIGH_CARDINALITY_THRESHOLD = 50

# A single value dominating this fraction of a numeric column is treated as a
# likely sentinel/placeholder (e.g. Home Credit's DAYS_EMPLOYED=365243 issue),
# flagged generically rather than hardcoded to one dataset's known quirk.
SUSPICIOUS_PLACEHOLDER_FRACTION = 0.5


# -----------------------------------------------------------------------------
# 1. Dataset Summary
# -----------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Produces the high-level dataset summary: shape, dtypes, numerical vs.
    categorical feature lists, target column, memory footprint.
    """
    numerical_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in (ID_COLUMN, TARGET_COLUMN)
    ]
    categorical_features = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
    ]

    dtype_counts = df.dtypes.astype(str).value_counts().to_dict()

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_000_000, 2),
        "dtype_counts": dtype_counts,
        "target_column": TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
        "id_column": ID_COLUMN if ID_COLUMN in df.columns else None,
        "n_numerical_features": len(numerical_features),
        "n_categorical_features": len(categorical_features),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
    }
    logger.info(
        "Dataset summary: %d rows, %d columns (%d numerical, %d categorical)",
        summary["n_rows"], summary["n_columns"],
        summary["n_numerical_features"], summary["n_categorical_features"],
    )
    return summary


# -----------------------------------------------------------------------------
# 2. Data Quality Report
# -----------------------------------------------------------------------------

def _detect_outliers_iqr(series: pd.Series) -> float:
    """Returns the percentage of values outside 1.5x IQR for a numeric series."""
    clean = series.dropna()
    if len(clean) < 10:
        return 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]
    return round(100 * len(outliers) / len(clean), 2)


def _detect_suspicious_placeholders(df: pd.DataFrame, numerical_features: list[str]) -> list[dict]:
    """
    Flags numeric columns where a single value accounts for an unusually large
    share of entries -- a common signature of a sentinel/placeholder value
    (e.g. "365243" used as a stand-in for missing DAYS_EMPLOYED in this dataset).
    """
    findings = []
    for col in numerical_features:
        value_counts = df[col].value_counts(normalize=True, dropna=True)
        if value_counts.empty:
            continue
        top_value, top_fraction = value_counts.index[0], value_counts.iloc[0]
        if top_fraction >= SUSPICIOUS_PLACEHOLDER_FRACTION:
            findings.append({
                "column": col,
                "suspicious_value": top_value,
                "fraction_of_rows": round(top_fraction * 100, 2),
            })
    return findings


def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Produces the data quality analysis: missing values, duplicates, constant
    columns, high-cardinality categoricals, suspicious placeholder values,
    outlier prevalence, and target class distribution.
    """
    numerical_features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in (ID_COLUMN, TARGET_COLUMN)
    ]
    categorical_features = list(df.select_dtypes(include=["object", "category"]).columns)

    # Missing values
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COLUMN].duplicated().sum()) if ID_COLUMN in df.columns else None

    # Constant / near-useless columns
    constant_columns = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # High-cardinality categoricals
    high_cardinality_columns = [
        {"column": c, "unique_values": int(df[c].nunique(dropna=True))}
        for c in categorical_features
        if df[c].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
    ]

    # Suspicious placeholder values
    suspicious_placeholders = _detect_suspicious_placeholders(df, numerical_features)

    # Outlier prevalence (top 10 by outlier %)
    outlier_summary = sorted(
        (
            {"column": c, "outlier_pct": _detect_outliers_iqr(df[c])}
            for c in numerical_features
        ),
        key=lambda r: r["outlier_pct"],
        reverse=True,
    )
    outlier_summary = [r for r in outlier_summary if r["outlier_pct"] > 0][:10]

    # Target distribution
    target_distribution = None
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(normalize=True).round(4) * 100
        target_distribution = {int(k): float(v) for k, v in counts.items()}

    report = {
        "missing_value_pct": missing_pct.to_dict(),
        "n_columns_with_missing": int((missing_pct > 0).sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "suspicious_placeholder_values": suspicious_placeholders,
        "outlier_summary_top10": outlier_summary,
        "target_distribution_pct": target_distribution,
    }

    logger.info(
        "Data quality report: %d columns with missing values, %d duplicate rows, "
        "%d constant columns, %d suspicious placeholder columns",
        report["n_columns_with_missing"], duplicate_rows,
        len(constant_columns), len(suspicious_placeholders),
    )
    return report


# -----------------------------------------------------------------------------
# 3. Business Feature Categorization
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CategoryRule:
    category: str
    match: "callable"  # column_name (str) -> bool


def _prefix(*prefixes: str):
    return lambda col: col.startswith(prefixes)


def _exact(*names: str):
    return lambda col: col in names


def _contains(*substrings: str):
    return lambda col: any(s in col for s in substrings)


# Ordered rules: first match wins. Order matters -- more specific rules first.
_CATEGORY_RULES: list[CategoryRule] = [
    CategoryRule("Identifier", _exact(ID_COLUMN)),
    CategoryRule("Target", _exact(TARGET_COLUMN)),
    CategoryRule("External Scoring Features", _prefix("EXT_SOURCE")),
    CategoryRule(
        "Financial Information",
        _exact("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
               "NAME_INCOME_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE"),
    ),
    CategoryRule(
        "Employment Information",
        _exact("DAYS_EMPLOYED", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "FLAG_EMP_PHONE"),
    ),
    CategoryRule(
        "Demographics",
        _exact("CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
               "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_HOUSING_TYPE",
               "NAME_TYPE_SUITE"),
    ),
    CategoryRule("Credit Bureau Inquiries", _prefix("AMT_REQ_CREDIT_BUREAU")),
    CategoryRule(
        "Social Circle / Repayment Behavior Proxy",
        _contains("CNT_SOCIAL_CIRCLE"),
    ),
    CategoryRule(
        "Housing / Asset Information",
        lambda col: col.startswith((
            "APARTMENTS_", "BASEMENTAREA_", "YEARS_BEGINEXPLUATATION_", "YEARS_BUILD_",
            "COMMONAREA_", "ELEVATORS_", "ENTRANCES_", "FLOORSMAX_", "FLOORSMIN_",
            "LANDAREA_", "LIVINGAPARTMENTS_", "LIVINGAREA_", "NONLIVINGAPARTMENTS_",
            "NONLIVINGAREA_",
        )) or col in (
            "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
            "HOUSETYPE_MODE", "FONDKAPREMONT_MODE",
        ),
    ),
    CategoryRule(
        "Contact & Document Flags",
        lambda col: col.startswith("FLAG_DOCUMENT") or col in (
            "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
        ),
    ),
    CategoryRule(
        "Regional & Application Context",
        lambda col: (
            col in (
                "NAME_CONTRACT_TYPE", "REGION_POPULATION_RELATIVE",
                "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
                "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
                "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE",
            )
            or col.startswith("REG_REGION_") or col.startswith("REG_CITY_")
            or col.startswith("LIVE_REGION_") or col.startswith("LIVE_CITY_")
        ),
    ),
]


def categorize_features(columns: list[str]) -> dict[str, list[str]]:
    """
    Assigns every column to exactly one business category using ordered
    rule-matching. Columns matching no rule fall into "Other".

    Returns:
        {category_name: [column_names]}
    """
    categorized: dict[str, list[str]] = {}
    for col in columns:
        assigned_category = "Other"
        for rule in _CATEGORY_RULES:
            try:
                if rule.match(col):
                    assigned_category = rule.category
                    break
            except TypeError:
                # Defensive: a malformed rule combinator should never crash categorization.
                continue
        categorized.setdefault(assigned_category, []).append(col)

    logger.info(
        "Categorized %d columns into %d business categories: %s",
        len(columns), len(categorized), list(categorized.keys()),
    )
    return categorized


def build_feature_reference(
    columns: list[str],
    description_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Builds a single reference table: column -> business category -> description.
    This is the table the UI, README, and NL-to-SQL schema prompt can all draw from.
    """
    description_map = description_map or {}
    category_by_column = {
        col: category
        for category, cols in categorize_features(columns).items()
        for col in cols
    }

    rows = [
        {
            "column": col,
            "category": category_by_column.get(col, "Other"),
            "description": description_map.get(col, "No official description available."),
        }
        for col in columns
    ]
    return pd.DataFrame(rows)
