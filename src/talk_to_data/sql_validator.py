"""
SQL validation for the Talk-to-Data chatbot. This is the hallucination /
safety guardrail: no LLM-generated SQL reaches the database without passing
these checks first.

Enforces:
  - Exactly one statement, and it must be a SELECT (read-only).
  - No disallowed keywords (DROP, DELETE, UPDATE, INSERT, ALTER, CREATE,
    ATTACH, PRAGMA, etc.) anywhere in the query.
  - Every referenced table exists in the known schema.
  - Every referenced column exists in the known schema for its table
    (best-effort -- complex queries with expressions are allowed through
    column-existence checks with a conservative fallback).
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from src.database.database import APPLICANTS_TABLE, APPLICANTS_TABLE_COLUMNS, PREDICTIONS_TABLE
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_TABLES = {
    APPLICANTS_TABLE: set(c.upper() for c in APPLICANTS_TABLE_COLUMNS),
    PREDICTIONS_TABLE: {"ID", "SK_ID_CURR", "PROBABILITY", "RISK_BAND", "MODEL_NAME", "CREATED_AT_UTC"},
}

_DISALLOWED_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "ATTACH",
    "DETACH", "PRAGMA", "REPLACE", "TRUNCATE", "GRANT", "REVOKE", "VACUUM",
}


class SQLValidationError(ValueError):
    """Raised with a clean, specific reason when a query fails validation."""


def _keyword_blacklist_check(sql: str) -> None:
    upper_sql = sql.upper()
    for keyword in _DISALLOWED_KEYWORDS:
        # Word-boundary-ish check to avoid false positives on substrings.
        if f" {keyword} " in f" {upper_sql} " or upper_sql.strip().startswith(keyword):
            raise SQLValidationError(
                f"Query contains a disallowed operation ('{keyword}'). "
                f"Only read-only SELECT queries are permitted."
            )


def validate_sql(sql: str) -> str:
    """
    Validates a SQL string against the allow-listed schema and read-only
    constraints. Returns the (possibly reformatted) validated SQL on success.

    Raises:
        SQLValidationError: with a specific, user-safe reason on failure.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("No SQL was generated for this question.")

    _keyword_blacklist_check(sql)

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except Exception as exc:  # noqa: BLE001
        logger.error("SQL parse error for [%s]: %s", sql, exc)
        raise SQLValidationError("The generated SQL could not be parsed.") from exc

    if len(statements) != 1 or statements[0] is None:
        raise SQLValidationError("Only a single SQL statement is allowed per query.")

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        raise SQLValidationError("Only SELECT queries are permitted.")

    # Validate referenced tables
    referenced_tables = {t.name.upper() for t in statement.find_all(exp.Table)}
    allowed_table_names = {t.upper() for t in _ALLOWED_TABLES}
    unknown_tables = referenced_tables - allowed_table_names
    if unknown_tables:
        raise SQLValidationError(
            f"Query references unknown table(s): {', '.join(unknown_tables)}. "
            f"Available tables: {', '.join(_ALLOWED_TABLES.keys())}."
        )
    if not referenced_tables:
        raise SQLValidationError("Query does not reference any known table.")

    # Validate referenced columns (best-effort: skip '*' and computed expressions)
    allowed_columns: set[str] = set()
    for table_name in referenced_tables:
        for t, cols in _ALLOWED_TABLES.items():
            if t.upper() == table_name:
                allowed_columns |= cols

    # Column aliases defined in the SELECT list (e.g. "AVG(x) AS default_rate")
    # are legitimately reused elsewhere in the statement (ORDER BY, HAVING) --
    # they are not schema columns and must not be rejected as unknown.
    defined_aliases = {
        select_expr.alias.upper()
        for select_expr in statement.expressions
        if isinstance(select_expr, exp.Alias) and select_expr.alias
    }
    allowed_columns |= defined_aliases

    for column in statement.find_all(exp.Column):
        col_name = column.name.upper()
        if col_name and col_name not in allowed_columns:
            raise SQLValidationError(
                f"Query references an unknown column: '{column.name}'. "
                f"This column does not exist in the available schema."
            )

    validated_sql = statement.sql(dialect="sqlite")
    logger.info("SQL validated successfully: %s", validated_sql)
    return validated_sql
