"""
Centralized configuration for the AI-Powered Credit Risk Intelligence Platform.

Every module reads settings from this file instead of hardcoding paths, keys,
or thresholds. Values are sourced from environment variables (populated via a
`.env` file locally, or real environment variables in Docker/production).

Usage:
    from src.utils.config import settings
    print(settings.LLM_PROVIDER)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if present. Safe to call even if the file
# doesn't exist (e.g. in a container where real env vars are injected instead).
load_dotenv()

# Project root = two levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_optional_int(key: str) -> int | None:
    """Reads an int env var that may legitimately be blank (e.g. DATA_SAMPLE_SIZE)."""
    raw = os.getenv(key, "").strip()
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_path(relative_or_absolute: str) -> Path:
    """Resolves a config path relative to the project root unless already absolute."""
    p = Path(relative_or_absolute)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass(frozen=True)
class Settings:
    # --- LLM Provider ---
    LLM_PROVIDER: str = field(default_factory=lambda: _get_str("LLM_PROVIDER", "mock").lower())
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: _get_str("ANTHROPIC_API_KEY", ""))
    ANTHROPIC_MODEL: str = field(default_factory=lambda: _get_str("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    ANTHROPIC_MAX_TOKENS: int = field(default_factory=lambda: _get_int("ANTHROPIC_MAX_TOKENS", 1024))

    # --- Dataset ---
    RAW_DATA_PATH: Path = field(default_factory=lambda: _resolve_path(_get_str("RAW_DATA_PATH", "data/raw/application_train.csv")))
    COLUMN_DESCRIPTION_PATH: Path = field(default_factory=lambda: _resolve_path(_get_str("COLUMN_DESCRIPTION_PATH", "data/raw/HomeCredit_columns_description.csv")))
    PROCESSED_DATA_PATH: Path = field(default_factory=lambda: _resolve_path(_get_str("PROCESSED_DATA_PATH", "data/processed/applicants_processed.csv")))
    DATA_SAMPLE_SIZE: int | None = field(default_factory=lambda: _get_optional_int("DATA_SAMPLE_SIZE"))
    RANDOM_SEED: int = field(default_factory=lambda: _get_int("RANDOM_SEED", 42))

    # --- Model Artifacts ---
    MODEL_DIR: Path = field(default_factory=lambda: _resolve_path(_get_str("MODEL_DIR", "models")))
    MODEL_FILENAME: str = field(default_factory=lambda: _get_str("MODEL_FILENAME", "credit_risk_pipeline.joblib"))
    FEATURE_METADATA_FILENAME: str = field(default_factory=lambda: _get_str("FEATURE_METADATA_FILENAME", "feature_metadata.json"))

    # --- Risk Bands ---
    MEDIUM_RISK_THRESHOLD: float = field(default_factory=lambda: _get_float("MEDIUM_RISK_THRESHOLD", 0.30))
    HIGH_RISK_THRESHOLD: float = field(default_factory=lambda: _get_float("HIGH_RISK_THRESHOLD", 0.60))

    # --- Database ---
    SQLITE_DB_PATH: Path = field(default_factory=lambda: _resolve_path(_get_str("SQLITE_DB_PATH", "data/processed/credit_risk.db")))

    # --- Conversation Memory ---
    CONVERSATION_MEMORY_TURNS: int = field(default_factory=lambda: _get_int("CONVERSATION_MEMORY_TURNS", 3))

    # --- Logging ---
    LOG_LEVEL: str = field(default_factory=lambda: _get_str("LOG_LEVEL", "INFO").upper())

    @property
    def MODEL_PATH(self) -> Path:
        return self.MODEL_DIR / self.MODEL_FILENAME

    @property
    def FEATURE_METADATA_PATH(self) -> Path:
        return self.MODEL_DIR / self.FEATURE_METADATA_FILENAME

    def validate(self) -> list[str]:
        """
        Returns a list of human-readable warnings for misconfiguration.
        Does not raise — callers decide how strict to be (e.g. UI shows a
        banner, training script may hard-fail on missing dataset).
        """
        warnings: list[str] = []

        if self.LLM_PROVIDER not in {"mock", "anthropic"}:
            warnings.append(
                f"LLM_PROVIDER='{self.LLM_PROVIDER}' is not recognized. "
                f"Expected 'mock' or 'anthropic'. Falling back to 'mock' behavior."
            )

        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            warnings.append(
                "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY is empty. "
                "Set the key in your .env file, or switch LLM_PROVIDER back to 'mock'."
            )

        if not (0.0 <= self.MEDIUM_RISK_THRESHOLD < self.HIGH_RISK_THRESHOLD <= 1.0):
            warnings.append(
                "Risk thresholds are invalid: expected "
                "0 <= MEDIUM_RISK_THRESHOLD < HIGH_RISK_THRESHOLD <= 1."
            )

        return warnings


# Singleton instance used across the entire application.
settings = Settings()
