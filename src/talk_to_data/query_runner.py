"""
Thin wrapper around the database layer for the Talk-to-Data flow, keeping
nl_to_sql.py focused on orchestration rather than DB/error-handling details.
"""

from __future__ import annotations

import pandas as pd

from src.database.database import DatabaseError, run_read_only_query
from src.utils.logger import get_logger

logger = get_logger(__name__)


def execute_validated_sql(sql: str) -> pd.DataFrame:
    """
    Executes a SQL string that has ALREADY passed sql_validator.validate_sql.
    Raises DatabaseError (clean, user-safe message) on failure.
    """
    return run_read_only_query(sql)


def summarize_result_for_llm(result_df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Converts a query result DataFrame into a compact text summary suitable
    for injecting into the answer-generation prompt -- part of the token
    optimization strategy (never send large result sets to the LLM).
    """
    if result_df.empty:
        return "[]"
    if len(result_df) > max_rows:
        truncated = result_df.head(max_rows)
        return f"{truncated.to_dict(orient='records')} (truncated to first {max_rows} of {len(result_df)} rows)"
    return str(result_df.to_dict(orient="records"))
