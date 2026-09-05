"""
Tests for the Talk-to-Data SQL validator.

Run with: pytest tests/test_sql_validator.py -v
"""

from __future__ import annotations

import pytest

from src.talk_to_data.sql_validator import SQLValidationError, validate_sql


# -----------------------------------------------------------------------------
# Valid queries must pass
# -----------------------------------------------------------------------------

def test_valid_aggregation_query_passes():
    validate_sql("SELECT AVG(AMT_INCOME_TOTAL) FROM applicants")


def test_valid_filter_query_passes():
    validate_sql("SELECT COUNT(*) FROM applicants WHERE CNT_CHILDREN > 0")


def test_valid_group_by_with_alias_and_order_by_passes():
    validate_sql(
        "SELECT NAME_INCOME_TYPE, AVG(TARGET) AS default_rate FROM applicants "
        "GROUP BY NAME_INCOME_TYPE ORDER BY default_rate DESC LIMIT 5"
    )


def test_valid_case_expression_query_passes():
    validate_sql(
        "SELECT CASE WHEN DAYS_EMPLOYED = 365243 THEN 'Not employed' ELSE 'Employed' END "
        "AS employment_status, AVG(TARGET) AS default_rate FROM applicants "
        "GROUP BY employment_status"
    )


# -----------------------------------------------------------------------------
# Unsafe operations must be rejected
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "DROP TABLE applicants",
    "DELETE FROM applicants WHERE SK_ID_CURR = 1",
    "UPDATE applicants SET TARGET = 0",
    "INSERT INTO applicants (SK_ID_CURR) VALUES (1)",
    "ALTER TABLE applicants ADD COLUMN hacked TEXT",
    "CREATE TABLE evil (id INTEGER)",
    "SELECT * FROM applicants; DROP TABLE applicants;",
    "ATTACH DATABASE 'other.db' AS other",
    "PRAGMA table_info(applicants)",
])
def test_unsafe_operations_are_rejected(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


# -----------------------------------------------------------------------------
# Schema violations must be rejected
# -----------------------------------------------------------------------------

def test_unknown_table_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT * FROM secret_table")


def test_unknown_column_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT made_up_column FROM applicants")


def test_empty_sql_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("")


def test_non_select_statement_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_sql("EXPLAIN SELECT * FROM applicants")
