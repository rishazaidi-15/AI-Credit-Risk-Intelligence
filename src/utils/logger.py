"""
Shared logging setup for the AI-Powered Credit Risk Intelligence Platform.

Every module should log through `get_logger(__name__)` rather than using
`print()`. This keeps log formatting consistent and makes it trivial to
distinguish "internal technical detail" (logged) from "clean user-facing
message" (shown in the Streamlit UI) — required for the SQL/LLM error
handling policy: never expose raw exceptions to end users.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded %d rows", len(df))
"""

from __future__ import annotations

import logging
import sys

from src.utils.config import settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    """Configures the root logger exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if this module is imported multiple times
    # (e.g. Streamlit's script re-run behavior).
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-scoped logger with consistent formatting.

    Args:
        name: typically `__name__` of the calling module.
    """
    _configure_root_logger()
    return logging.getLogger(name)
