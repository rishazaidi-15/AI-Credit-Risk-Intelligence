"""
SQLite database layer for the credit risk platform.

Two tables:
  - applicants:   processed applicant records, loaded from the real dataset.
                   This is what the Talk-to-Data chatbot queries.
  - predictions:  a log of prediction results served by the app (probability,
                   risk band, timestamp) -- lets the dashboard show recent
                   activity and the chatbot answer questions about predictions
                   actually made, not fabricated ones.

All functions raise DatabaseError with a clean message on failure; callers
(UI, chatbot) should catch this rather than let a raw sqlite3/SQLAlchemy
exception reach the end user.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns from the processed dataset that are loaded into the `applicants`
# table -- kept to a business-relevant subset so the chatbot's schema stays
# small and query-friendly (per the "token optimization" design goal).
APPLICANTS_TABLE_COLUMNS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT",
    "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "REGION_RATING_CLIENT", "REGION_POPULATION_RELATIVE",
]

APPLICANTS_TABLE = "applicants"
PREDICTIONS_TABLE = "predictions"


class DatabaseError(RuntimeError):
    """Raised on any database failure; message is safe to show to end users."""


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{settings.SQLITE_DB_PATH}")
        logger.info("Connected to SQLite database at %s", settings.SQLITE_DB_PATH)
    return _engine


def init_predictions_table() -> None:
    """Creates the predictions table if it doesn't already exist."""
    try:
        with get_engine().begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sk_id_curr INTEGER,
                    probability REAL NOT NULL,
                    risk_band TEXT NOT NULL,
                    model_name TEXT,
                    created_at_utc TEXT NOT NULL
                )
            """))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialize predictions table: %s", exc)
        raise DatabaseError("Could not initialize the database.") from exc


def load_applicants_table(raw_df: pd.DataFrame) -> int:
    """
    Loads (replaces) the applicants table from a raw dataset DataFrame,
    restricted to APPLICANTS_TABLE_COLUMNS that actually exist in the data.

    Returns the number of rows loaded.
    """
    try:
        available_columns = [c for c in APPLICANTS_TABLE_COLUMNS if c in raw_df.columns]
        subset = raw_df[available_columns].copy()
        subset.to_sql(APPLICANTS_TABLE, get_engine(), if_exists="replace", index=False)
        logger.info("Loaded %d rows into '%s' table (%d columns)", len(subset), APPLICANTS_TABLE, len(available_columns))
        return len(subset)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load applicants table: %s", exc)
        raise DatabaseError("Could not load applicant data into the database.") from exc


def is_applicants_table_populated() -> bool:
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {APPLICANTS_TABLE}"))
            return result.scalar() > 0
    except Exception:
        return False


def save_prediction(sk_id_curr: int | None, probability: float, risk_band: str, model_name: str) -> None:
    """Logs one prediction result. Never raises to the caller on failure -- logs and continues,
    since a failed audit-log write should not block showing the user their prediction."""
    try:
        init_predictions_table()
        with get_engine().begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO predictions (sk_id_curr, probability, risk_band, model_name, created_at_utc)
                    VALUES (:sk_id_curr, :probability, :risk_band, :model_name, :created_at_utc)
                """),
                {
                    "sk_id_curr": sk_id_curr,
                    "probability": probability,
                    "risk_band": risk_band,
                    "model_name": model_name,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save prediction record (non-fatal): %s", exc)


def run_read_only_query(sql: str, max_rows: int = 200) -> pd.DataFrame:
    """
    Executes a SQL query and returns the result as a DataFrame, capped at
    max_rows (part of the token-optimization strategy -- large result sets
    are never sent back to the LLM in full).

    This function assumes `sql` has ALREADY been validated as safe/read-only
    by src.talk_to_data.sql_validator -- it does not re-validate.
    """
    try:
        with get_engine().connect() as conn:
            result_df = pd.read_sql_query(text(sql), conn)
        if len(result_df) > max_rows:
            logger.info("Query returned %d rows; truncating to %d for downstream use.", len(result_df), max_rows)
            result_df = result_df.head(max_rows)
        return result_df
    except Exception as exc:  # noqa: BLE001
        logger.error("Query execution failed for SQL [%s]: %s", sql, exc)
        raise DatabaseError(
            "The query could not be executed against the database. "
            "It may reference a column or table that doesn't exist."
        ) from exc


def get_schema_description() -> str:
    """
    Returns a compact, human-readable schema description for the LLM prompt
    -- only the applicants table's columns, since that's what the chatbot
    answers questions about. Kept short deliberately (token optimization).
    """
    columns_line = ", ".join(APPLICANTS_TABLE_COLUMNS)
    return (
        f"Table: {APPLICANTS_TABLE}\n"
        f"Columns: {columns_line}\n"
        f"Notes: TARGET is 1 if the applicant defaulted, 0 otherwise. "
        f"AMT_* columns are monetary amounts. DAYS_BIRTH and DAYS_EMPLOYED "
        f"are negative day counts relative to the application date."
    )
