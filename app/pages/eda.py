"""EDA & Business Insights page."""

from __future__ import annotations

import streamlit as st

from components.ui_components import dataset_status_banner, error_banner
from src.data.loader import DatasetNotFoundError, DatasetValidationError, check_dataset_availability, load_raw_data
from src.eda.analysis import categorize_features, data_quality_report, dataset_summary
from src.utils.config import settings


@st.cache_data(show_spinner="Loading dataset...")
def _load_data_cached():
    return load_raw_data()


def render() -> None:
    st.header("Data Understanding & Business Insights")
    st.caption("Exploratory analysis of the Home Credit Default Risk dataset.")

    availability = check_dataset_availability()
    if not dataset_status_banner(availability):
        return

    try:
        df = _load_data_cached()
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        error_banner(str(exc))
        return

    summary = dataset_summary(df)
    quality = data_quality_report(df)

    st.subheader("Dataset Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary['n_rows']:,}")
    c2.metric("Columns", summary["n_columns"])
    c3.metric("Numerical features", summary["n_numerical_features"])
    c4.metric("Categorical features", summary["n_categorical_features"])

    st.divider()
    st.subheader("Business Insight 1 — Target (Default) Distribution")
    if quality["target_distribution_pct"]:
        target_dist = quality["target_distribution_pct"]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Non-default rate", f"{target_dist.get(0, 0):.1f}%")
            st.metric("Default rate", f"{target_dist.get(1, 0):.1f}%")
        with col2:
            st.bar_chart({"Percentage": target_dist})
        st.markdown(
            "**Business interpretation:** the dataset is heavily imbalanced -- "
            "most applicants repay their loans. This is why the ML layer evaluates "
            "with ROC-AUC/PR-AUC and class-weighting rather than plain accuracy."
        )

    st.divider()
    st.subheader("Business Insight 2 — External Credit Scores vs. Default")
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
    if ext_cols and "TARGET" in df.columns:
        ext_by_target = df.groupby("TARGET")[ext_cols].mean().round(3)
        st.bar_chart(ext_by_target.T)
        st.markdown(
            "**Business interpretation:** applicants who defaulted (TARGET=1) tend to "
            "have lower average external credit scores than those who repaid -- "
            "consistent with these scores being pre-existing risk indicators."
        )

    st.divider()
    st.subheader("Business Insight 3 — Default Rate by Income Type")
    if "NAME_INCOME_TYPE" in df.columns and "TARGET" in df.columns:
        by_income = (df.groupby("NAME_INCOME_TYPE")["TARGET"].mean() * 100).sort_values(ascending=False)
        st.bar_chart(by_income)
        st.markdown(
            f"**Business interpretation:** default rates vary meaningfully across income "
            f"types, from {by_income.min():.1f}% to {by_income.max():.1f}% -- income type "
            f"alone is not the whole story, but it's a useful segmentation for policy discussions."
        )

    st.divider()
    st.subheader("Business Insight 4 — Credit-to-Income Ratio vs. Default")
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns and "TARGET" in df.columns:
        ratio = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).clip(upper=20)
        ratio_by_target = ratio.groupby(df["TARGET"]).mean().round(2)
        st.bar_chart(ratio_by_target)
        st.markdown(
            "**Business interpretation:** a higher credit-to-income ratio means the "
            "requested loan is large relative to the applicant's income -- worth "
            "comparing between defaulters and non-defaulters as a leverage indicator."
        )

    st.divider()
    st.subheader("Business Insight 5 — Missing Value Patterns")
    if quality["missing_value_pct"]:
        top_missing = dict(list(quality["missing_value_pct"].items())[:10])
        st.bar_chart(top_missing)
        st.markdown(
            "**Business interpretation:** columns with high missingness (e.g. external "
            "scores or employment fields) often reflect applicants with thinner credit "
            "files -- missingness itself can carry risk signal, not just noise."
        )
    else:
        st.write("No missing values detected in this dataset.")

    st.divider()
    st.subheader("Data Quality Observations")
    dq_col1, dq_col2 = st.columns(2)
    with dq_col1:
        st.write(f"**Duplicate rows:** {quality['duplicate_rows']}")
        st.write(f"**Constant columns:** {quality['constant_columns'] or 'None'}")
        st.write(f"**High-cardinality categoricals:** {len(quality['high_cardinality_columns'])}")
    with dq_col2:
        if quality["suspicious_placeholder_values"]:
            st.write("**Suspicious/sentinel values detected:**")
            for item in quality["suspicious_placeholder_values"]:
                st.write(f"- `{item['column']}`: value `{item['suspicious_value']}` in {item['fraction_of_rows']}% of rows")
        else:
            st.write("**Suspicious/sentinel values:** none detected")

    st.divider()
    st.subheader("Business Feature Categories")
    categories = categorize_features(list(df.columns))
    for category, cols in categories.items():
        with st.expander(f"{category} ({len(cols)} columns)"):
            st.write(", ".join(cols))
