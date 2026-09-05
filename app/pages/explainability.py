"""Explainability page."""

from __future__ import annotations

import streamlit as st

from app.components.ui_components import error_banner, info_banner, risk_badge
from src.data.loader import DatasetNotFoundError, DatasetValidationError, load_raw_data
from src.explainability.shap_explainer import ExplanationUnavailableError, global_explanation, local_explanation
from src.ml.predict import is_model_available


@st.cache_data(show_spinner="Loading dataset for global explanation...")
def _load_data_cached():
    return load_raw_data()


def render() -> None:
    st.header("Prediction Explainability")
    st.caption("Understand which factors drive the model's risk predictions.")

    if not is_model_available():
        info_banner("No trained model found yet. Train one first: `python -m src.ml.train`.")
        return

    st.subheader("Global Explainability")
    st.caption("Which features generally influence the model's predictions the most?")
    try:
        df = _load_data_cached()
        sample = df.sample(min(500, len(df)), random_state=1)
        global_factors = global_explanation(sample)
        chart_data = {f["business_label"]: f["mean_abs_shap_value"] for f in global_factors}
        st.bar_chart(chart_data)
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        error_banner(str(exc))
    except ExplanationUnavailableError as exc:
        error_banner(str(exc))

    st.divider()
    st.subheader("Local Explainability — Last Prediction")

    if "last_applicant" not in st.session_state:
        info_banner("Make a prediction on the Risk Prediction page first to see its explanation here.")
        return

    applicant = st.session_state["last_applicant"]
    prediction = st.session_state["last_prediction"]

    risk_badge(prediction["probability"], prediction["risk_band"])
    st.write("")

    try:
        local = local_explanation(applicant)
    except ExplanationUnavailableError as exc:
        error_banner(str(exc))
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top factors increasing risk**")
        if local["top_risk_increasing"]:
            for factor in local["top_risk_increasing"]:
                st.write(f"- {factor['business_label']}")
        else:
            st.write("No strong risk-increasing factors identified.")

    with col2:
        st.markdown("**Top factors reducing risk**")
        if local["top_risk_decreasing"]:
            for factor in local["top_risk_decreasing"]:
                st.write(f"- {factor['business_label']}")
        else:
            st.write("No strong risk-reducing factors identified.")

    st.caption(
        "Explanations are computed from the actual trained model's SHAP values "
        "for this specific applicant -- not fixed or hardcoded rules."
    )
