"""
Single source of truth for converting a default probability into a business
risk band. Every component that needs a risk band (inference, database
records, the UI, and the chatbot) MUST import get_risk_band from here rather
than re-implementing threshold logic -- this is what prevents contradictory
risk calculations across the app.

Thresholds are configurable via .env (MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD)
and validated at startup by settings.validate().
"""

from __future__ import annotations

from src.utils.config import settings

LOW_RISK = "LOW"
MEDIUM_RISK = "MEDIUM"
HIGH_RISK = "HIGH"

RISK_BAND_ORDER = {LOW_RISK: 0, MEDIUM_RISK: 1, HIGH_RISK: 2}


def get_risk_band(probability: float) -> str:
    """
    Maps a default probability to a risk band using the configured thresholds.

    Documented rationale (see README "Risk Band Logic"): thresholds default
    to 0.30 / 0.60, chosen as reasonable starting points and validated
    against the predicted-probability distribution during training (see
    src/ml/train.py, which logs the distribution for this exact purpose).
    """
    if probability is None or probability != probability:  # NaN check without numpy import
        raise ValueError("probability must be a valid number, got NaN/None")
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    if probability >= settings.HIGH_RISK_THRESHOLD:
        return HIGH_RISK
    if probability >= settings.MEDIUM_RISK_THRESHOLD:
        return MEDIUM_RISK
    return LOW_RISK


def risk_band_to_ordinal(band: str) -> int:
    """Converts a risk band label to an ordinal (0=LOW, 1=MEDIUM, 2=HIGH) for ML use (e.g. rule derivation)."""
    return RISK_BAND_ORDER[band]
