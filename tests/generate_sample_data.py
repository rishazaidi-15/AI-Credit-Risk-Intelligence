"""
DEVELOPMENT / TESTING TOOL ONLY.

Generates a synthetic CSV that mimics the schema of the real
application_train.csv from the Home Credit Default Risk dataset, covering
every column the modeling pipeline (src/data/feature_engineering.py) expects.
This lets you smoke-test the ENTIRE pipeline -- loading, EDA, training,
inference, explainability, rule derivation, the database layer, and the UI
-- without waiting on the ~700MB Kaggle download.

This is NOT the project dataset and must never be used for actual model
training, evaluation, or any deliverable. The assignment requires the real
Home Credit Default Risk dataset (see README "Dataset Setup"). TARGET here
is synthetically generated with a plausible (but fake) relationship to a
few features purely so smoke-tested models produce non-degenerate metrics
(e.g. ROC-AUC noticeably above 0.5) -- it carries no real-world meaning.

Usage:
    python tests/generate_sample_data.py
    # writes tests/fixtures/sample_application_train.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_application_train.csv"


def generate_sample_dataframe(n_rows: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """Builds a synthetic DataFrame covering the full modeling feature schema."""
    rng = np.random.default_rng(random_seed)

    ext_source_1 = rng.uniform(0, 1, size=n_rows)
    ext_source_2 = rng.uniform(0, 1, size=n_rows)
    ext_source_3 = np.where(rng.random(n_rows) < 0.2, np.nan, rng.uniform(0, 1, size=n_rows))
    amt_income_total = rng.normal(180000, 60000, size=n_rows).clip(min=25000)
    amt_credit = rng.normal(600000, 200000, size=n_rows).clip(min=45000)
    days_birth = -rng.integers(7300, 25000, size=n_rows)
    flag_own_car = rng.choice(["Y", "N"], size=n_rows)

    days_employed_raw = -rng.integers(30, 15000, size=n_rows)
    is_sentinel = rng.random(n_rows) < 0.55
    days_employed = np.where(is_sentinel, 365243, days_employed_raw)

    # Synthetic (fake) relationship so the smoke-tested model has learnable
    # signal: lower external scores, higher credit-to-income ratio, and
    # younger age nudge default probability up. Coefficients are arbitrary.
    ext_mean = np.nanmean(np.vstack([ext_source_1, ext_source_2, ext_source_3]), axis=0)
    credit_income_ratio = amt_credit / amt_income_total
    age_years = -days_birth / 365.25

    logit = (
        -2.0
        - 3.0 * ext_mean
        + 0.15 * (credit_income_ratio - credit_income_ratio.mean())
        - 0.02 * (age_years - age_years.mean())
    )
    default_prob = 1 / (1 + np.exp(-logit))
    target = (rng.random(n_rows) < default_prob).astype(int)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100001, 100001 + n_rows),
        "TARGET": target,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], size=n_rows),
        "CODE_GENDER": rng.choice(["M", "F"], size=n_rows),
        "FLAG_OWN_CAR": flag_own_car,
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n_rows),
        "CNT_CHILDREN": rng.integers(0, 5, size=n_rows),
        "AMT_INCOME_TOTAL": amt_income_total,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": rng.normal(27000, 8000, size=n_rows).clip(min=1000),
        "AMT_GOODS_PRICE": rng.normal(540000, 180000, size=n_rows).clip(min=10000),
        "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", "Spouse, partner"], size=n_rows),
        "NAME_INCOME_TYPE": rng.choice(
            ["Working", "Commercial associate", "Pensioner", "State servant"], size=n_rows
        ),
        "NAME_EDUCATION_TYPE": rng.choice(
            ["Secondary", "Higher education", "Incomplete higher"], size=n_rows
        ),
        "NAME_FAMILY_STATUS": rng.choice(
            ["Married", "Single / not married", "Civil marriage"], size=n_rows
        ),
        "NAME_HOUSING_TYPE": rng.choice(
            ["House / apartment", "Rented apartment", "With parents"], size=n_rows
        ),
        "DAYS_BIRTH": days_birth,
        # Simulates the real dataset's well-known DAYS_EMPLOYED sentinel-value
        # quirk (365243 used as a placeholder), so the placeholder detector
        # in analysis.py -- and the cleaning logic in feature_engineering.py --
        # both have something realistic to catch.
        "DAYS_EMPLOYED": days_employed,
        "DAYS_REGISTRATION": -rng.integers(0, 20000, size=n_rows).astype(float),
        "DAYS_ID_PUBLISH": -rng.integers(0, 7000, size=n_rows).astype(float),
        "OWN_CAR_AGE": np.where(flag_own_car == "Y", rng.integers(0, 20, size=n_rows).astype(float), np.nan),
        "OCCUPATION_TYPE": rng.choice(["Laborers", "Sales staff", "Core staff", None], size=n_rows),
        "ORGANIZATION_TYPE": rng.choice(
            ["Business Entity Type 3", "Self-employed", "Government"], size=n_rows
        ),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, size=n_rows).astype(float),
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "REGION_RATING_CLIENT": rng.integers(1, 4, size=n_rows),
        "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.08, size=n_rows),
        "WEEKDAY_APPR_PROCESS_START": rng.choice(
            ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"], size=n_rows
        ),
        "FLAG_DOCUMENT_3": rng.choice([0, 1], size=n_rows),
        "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 10, size=n_rows).astype(float),
        "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 10, size=n_rows).astype(float),
        "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, size=n_rows).astype(float),
        "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4000, size=n_rows).astype(float),
    })

    return df


def main() -> None:
    df = generate_sample_dataframe()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Synthetic smoke-test dataset written to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Reminder: this is for pipeline smoke-testing only, not the real dataset.")


if __name__ == "__main__":
    main()
