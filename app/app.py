"""
AI-Powered Credit Risk Intelligence Platform - Streamlit application entry point.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from pages import chatbot, eda, explainability, prediction, rules
from src.data.loader import DatasetNotFoundError, DatasetValidationError, check_dataset_availability, load_raw_data
from src.eda.analysis import data_quality_report, dataset_summary
from src.ml.predict import is_model_available
from src.utils.config import settings

st.set_page_config(
    page_title="AI-Powered Credit Risk Intelligence Platform",
    page_icon="📊",
    layout="wide",
)


def render_home() -> None:
    st.title("AI-Powered Credit Risk Intelligence Platform")
    st.caption("End-to-end credit risk assessment: data, ML, explainability, business rules, and a Talk-to-Data assistant.")

    config_warnings = settings.validate()
    for warning in config_warnings:
        st.warning(warning)

    availability = check_dataset_availability()
    model_ready = is_model_available()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset status", "Available" if availability["raw_data_available"] else "Not found")
    with col2:
        st.metric("Trained model status", "Ready" if model_ready else "Not trained yet")

    if not availability["raw_data_available"]:
        st.info(
            "Place the Home Credit dataset at the configured paths to unlock the full "
            "dashboard (see README 'Dataset Setup'), or run "
            "`python tests/generate_sample_data.py` for a quick offline smoke test."
        )
        return

    try:
        df = load_raw_data()
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        st.error(str(exc))
        return

    summary = dataset_summary(df)
    quality = data_quality_report(df)

    st.divider()
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicants", f"{summary['n_rows']:,}")
    c2.metric("Features used", "39")
    if quality["target_distribution_pct"]:
        c3.metric("Observed default rate", f"{quality['target_distribution_pct'].get(1, 0):.1f}%")
    c4.metric("Columns with missing data", quality["n_columns_with_missing"])

    if model_ready:
        st.divider()
        st.subheader("Model Status")
        from src.ml.predict import get_feature_metadata
        meta = get_feature_metadata()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active model", meta["best_model_name"].replace("_", " ").title())
        m2.metric("ROC-AUC", meta["test_metrics"]["roc_auc"])
        m3.metric("PR-AUC", meta["test_metrics"]["pr_auc"])
        m4.metric("Recall", meta["test_metrics"]["recall"])

    st.divider()
    st.info(
        "Use the sidebar to explore EDA insights, run a risk prediction, review "
        "explainability, inspect model-derived rules, or ask the AI assistant "
        "questions about the data."
    )


PAGES = {
    " Home / Overview": render_home,
    " EDA & Business Insights": eda.render,
    " Risk Prediction": prediction.render,
    " Explainability": explainability.render,
    " Decision Rules": rules.render,
    " AI Assistant": chatbot.render,
}


def main() -> None:
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("AI-Powered Credit Risk Intelligence Platform")
    st.sidebar.caption(f"LLM Provider: {settings.LLM_PROVIDER}")

    PAGES[selection]()


if __name__ == "__main__":
    main()
