"""
Preprocessing + model pipeline construction.

One ColumnTransformer definition is shared across all candidate models
(Logistic Regression, Random Forest, XGBoost). Standard-scaling numeric
features doesn't materially hurt tree-based models and keeps a single,
consistent preprocessing path -- avoiding the leakage and drift risk of
maintaining separate preprocessing per model. This is a deliberate
simplicity tradeoff, documented here and in the README.

The fitted pipeline (preprocessor + classifier) is what gets saved to disk
and reused for every prediction, so training and inference are always
consistent by construction (one artifact, not reimplemented logic).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Builds the shared preprocessing step:
      - Numeric: median imputation (robust to outliers/skew) + standard scaling.
      - Categorical: most-frequent imputation + one-hot encoding, with unseen
        categories at inference time safely ignored (handle_unknown='ignore')
        rather than raising -- required for real-world inference robustness.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    classifier,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Wraps a classifier with the shared preprocessing step into one fittable pipeline."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
