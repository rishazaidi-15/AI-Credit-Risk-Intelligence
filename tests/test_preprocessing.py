"""
Tests for data loading and feature engineering consistency.

Run with: pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    DAYS_EMPLOYED_SENTINEL,
    clean_and_engineer,
    get_feature_lists,
)
from src.data.loader import DatasetNotFoundError, DatasetValidationError, load_raw_data
from tests.generate_sample_data import generate_sample_dataframe


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return generate_sample_dataframe(n_rows=300, random_seed=1)


def test_load_raw_data_missing_file_raises_clean_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetNotFoundError):
        load_raw_data(path=missing_path)


def test_load_raw_data_rejects_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad_csv, index=False)
    with pytest.raises(DatasetValidationError):
        load_raw_data(path=bad_csv)


def test_load_raw_data_configurable_sampling(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = generate_sample_dataframe(n_rows=200, random_seed=2)
    df.to_csv(csv_path, index=False)

    full = load_raw_data(path=csv_path, sample_size=None)
    sampled = load_raw_data(path=csv_path, sample_size=50, random_state=42)

    assert len(full) == 200
    assert len(sampled) == 50


def test_clean_and_engineer_produces_expected_columns(sample_df):
    engineered = clean_and_engineer(sample_df, require_target=True)
    numeric_features, categorical_features = get_feature_lists()

    for col in numeric_features + categorical_features:
        assert col in engineered.columns, f"Expected engineered column missing: {col}"
    assert "SK_ID_CURR" in engineered.columns
    assert "TARGET" in engineered.columns


def test_clean_and_engineer_requires_target_when_asked(sample_df):
    df_without_target = sample_df.drop(columns=["TARGET"])
    with pytest.raises(ValueError):
        clean_and_engineer(df_without_target, require_target=True)


def test_days_employed_sentinel_is_cleaned(sample_df):
    engineered = clean_and_engineer(sample_df, require_target=False)
    # No engineered DAYS_EMPLOYED value should equal the raw sentinel anymore.
    assert not (engineered["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any()
    # The flag should be set for rows that originally had the sentinel.
    original_sentinel_mask = sample_df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    assert engineered.loc[original_sentinel_mask, "IS_RETIRED_OR_UNEMPLOYED_FLAG"].eq(1).all()


def test_clean_and_engineer_handles_partial_applicant_record():
    """A UI form might submit only a few fields -- this must not KeyError or produce NaT/object dtype issues."""
    sparse = pd.DataFrame([{"AMT_INCOME_TOTAL": 100000}])
    engineered = clean_and_engineer(sparse, require_target=False)
    numeric_features, categorical_features = get_feature_lists()

    for col in numeric_features:
        assert pd.api.types.is_float_dtype(engineered[col]) or pd.api.types.is_integer_dtype(engineered[col]), (
            f"Numeric feature {col} has unexpected dtype {engineered[col].dtype}"
        )


def test_clean_and_engineer_is_deterministic_for_training_and_inference(sample_df):
    """The same row must engineer to identical features whether processed as
    part of a full batch (training-style) or alone (inference-style) --
    this is what keeps train/predict behavior consistent."""
    batch_result = clean_and_engineer(sample_df, require_target=True)
    single_row = sample_df.iloc[[0]].drop(columns=["TARGET"])
    single_result = clean_and_engineer(single_row, require_target=False)

    numeric_features, categorical_features = get_feature_lists()
    for col in numeric_features + categorical_features:
        batch_val = batch_result.iloc[0][col]
        single_val = single_result.iloc[0][col]
        if pd.isna(batch_val) and pd.isna(single_val):
            continue
        if isinstance(batch_val, float):
            assert batch_val == pytest.approx(single_val), f"Mismatch in {col}"
        else:
            assert batch_val == single_val, f"Mismatch in {col}"


def test_safe_ratio_handles_zero_denominator():
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [0, 100000],
        "AMT_CREDIT": [50000, 200000],
        "AMT_ANNUITY": [1000, 2000],
        "AMT_GOODS_PRICE": [45000, 190000],
        "DAYS_BIRTH": [-10000, -12000],
        "DAYS_EMPLOYED": [-500, -1000],
        "EXT_SOURCE_1": [0.5, 0.6],
        "EXT_SOURCE_2": [0.5, 0.6],
        "EXT_SOURCE_3": [0.5, 0.6],
    })
    engineered = clean_and_engineer(df, require_target=False)
    assert pd.isna(engineered.loc[0, "CREDIT_INCOME_RATIO"])
    assert not pd.isna(engineered.loc[1, "CREDIT_INCOME_RATIO"])
