"""
Reusable Streamlit UI components shared across pages. Keeping these here
(rather than duplicating markup in every page) is what keeps the risk-band
color coding and error presentation consistent app-wide.
"""

from __future__ import annotations

import streamlit as st

from src.ml.risk import HIGH_RISK, LOW_RISK, MEDIUM_RISK

_RISK_COLORS = {
    LOW_RISK: "#1B7A43",
    MEDIUM_RISK: "#B8860B",
    HIGH_RISK: "#B3261E",
}
_RISK_BACKGROUNDS = {
    LOW_RISK: "#E7F5EC",
    MEDIUM_RISK: "#FBF1DD",
    HIGH_RISK: "#FBEAEA",
}


def risk_badge(probability: float, risk_band: str) -> None:
    """Renders a colored risk badge with the probability and band label."""
    color = _RISK_COLORS.get(risk_band, "#444444")
    background = _RISK_BACKGROUNDS.get(risk_band, "#F0F0F0")
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            padding: 0.85rem 1.4rem;
            border-radius: 8px;
            background-color: {background};
            border: 1px solid {color};
            color: {color};
            font-weight: 600;
            font-size: 1.1rem;
        ">
            {risk_band} RISK &nbsp;&mdash;&nbsp; {probability * 100:.1f}% default probability
        </div>
        """,
        unsafe_allow_html=True,
    )


def error_banner(message: str) -> None:
    """Displays a clean, user-facing error message. Never pass raw exception text here."""
    st.error(message)


def info_banner(message: str) -> None:
    st.info(message)


def dataset_status_banner(availability: dict) -> bool:
    """
    Shows a banner if the real dataset isn't in place yet. Returns True if
    the dataset is available (so callers can decide whether to proceed).
    """
    if not availability["raw_data_available"]:
        st.warning(
            f"The Home Credit dataset was not found at "
            f"`{availability['raw_data_path']}`. "
            f"Please see the README 'Dataset Setup' section, or run "
            f"`python tests/generate_sample_data.py` for a quick offline smoke test."
        )
        return False
    return True
