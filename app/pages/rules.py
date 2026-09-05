"""Business Rules page."""

from __future__ import annotations

import streamlit as st

from app.components.ui_components import error_banner, info_banner
from src.data.loader import DatasetNotFoundError, DatasetValidationError, load_raw_data
from src.ml.predict import is_model_available
from src.rules.rule_derivation import RULE_DISCLAIMER, RuleDerivationError, derive_rules


@st.cache_data(show_spinner="Deriving business rules...")
def _derive_rules_cached():
    df = load_raw_data()
    return derive_rules(df)


def render() -> None:
    st.header("Model-Derived Decision Rules")
    st.caption("Simplified, auditable rules that approximate the model's decision behavior.")

    if not is_model_available():
        info_banner("No trained model found yet. Train one first: `python -m src.ml.train`.")
        return

    st.warning(RULE_DISCLAIMER)

    try:
        rules = _derive_rules_cached()
    except (DatasetNotFoundError, DatasetValidationError) as exc:
        error_banner(str(exc))
        return
    except RuleDerivationError as exc:
        error_banner(str(exc))
        return

    if not rules:
        st.write("No rules could be derived with sufficient support from the current data.")
        return

    for i, rule in enumerate(rules, start=1):
        band_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}.get(rule["predicted_risk_band"], "gray")
        with st.container(border=True):
            st.markdown(f"**Rule {i}** — :{band_color}[{rule['predicted_risk_band']} RISK]")
            st.write(rule["rule_text"])
            st.caption(f"Supported by {rule['supporting_samples']} applicants, {rule['purity']*100:.0f}% consistent")
