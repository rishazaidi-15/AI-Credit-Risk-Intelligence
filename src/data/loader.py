"""
Dataset loading and initial validation for the Home Credit Default Risk data.

This module is the single entry point for reading:
  1. application_train.csv
     -> the primary modeling dataset (has TARGET)
  2. HomeCredit_columns_description.csv
     -> official column descriptions, used for business-readable feature
        categorization and chatbot/UI explanations.

Design decisions:
  - Sampling is OPTIONAL and controlled by settings.DATA_SAMPLE_SIZE.
  - When sampling is enabled, rows are selected before the full DataFrame is
    constructed, avoiding unnecessary memory usage.
  - Leave DATA_SAMPLE_SIZE blank to load the full dataset.
  - Errors are raised with clear, specific messages and logged internally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Internal sentinel used to distinguish:
# load_raw_data()        -> use DATA_SAMPLE_SIZE from config
# load_raw_data(None)    -> explicitly load the full dataset
_USE_CONFIG = object()


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a required dataset file is missing from disk."""


class DatasetValidationError(ValueError):
    """Raised when a loaded dataset fails a basic structural check."""


def _require_file(path: Path, human_name: str) -> None:
    """Ensure a required file exists."""
    if not path.exists():
        message = (
            f"{human_name} not found at expected path: {path}. "
            f"Please download it from the Home Credit Default Risk Kaggle "
            f"competition and place it at this location (see README "
            f"'Dataset Setup' section)."
        )
        logger.error(message)
        raise DatasetNotFoundError(message)


def _count_data_rows(path: Path) -> int:
    """
    Count CSV data rows without loading the dataset into a DataFrame.

    The first row is assumed to be the CSV header.
    """
    with path.open("r", encoding="utf-8", errors="replace") as file:
        total_lines = sum(1 for _ in file)

    return max(total_lines - 1, 0)


def _read_sampled_csv(
    path: Path,
    sample_size: int,
    random_state: int,
    total_rows: int,
) -> pd.DataFrame:
    """
    Read a reproducible random sample of rows without first constructing
    a full DataFrame containing the entire dataset.

    Pandas row numbers include the header as row 0, so data rows are numbered
    from 1 through total_rows.
    """
    rng = np.random.default_rng(random_state)

    selected_rows = set(
        rng.choice(
            np.arange(1, total_rows + 1),
            size=sample_size,
            replace=False,
        ).tolist()
    )

    return pd.read_csv(
        path,
        skiprows=lambda row_number: (
            row_number != 0 and row_number not in selected_rows
        ),
    )


def load_raw_data(
    path: str | Path | None = None,
    sample_size: int | None | object = _USE_CONFIG,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Load application_train.csv with optional memory-efficient row sampling.

    Args:
        path:
            Optional override for the CSV path. Defaults to
            settings.RAW_DATA_PATH.

        sample_size:
            Number of rows to randomly sample.

            - If omitted, uses settings.DATA_SAMPLE_SIZE.
            - If None is explicitly passed, loads the full dataset.
            - If an integer is passed, uses that sample size.

        random_state:
            Random seed for reproducible sampling. Defaults to
            settings.RANDOM_SEED.

    Returns:
        A validated pandas DataFrame.

    Raises:
        DatasetNotFoundError:
            If the dataset file does not exist.

        DatasetValidationError:
            If the dataset cannot be read or fails structural validation.
    """
    resolved_path = (
        Path(path)
        if path is not None
        else settings.RAW_DATA_PATH
    )

    _require_file(resolved_path, "application_train.csv")

    if sample_size is _USE_CONFIG:
        effective_sample_size = settings.DATA_SAMPLE_SIZE
    else:
        effective_sample_size = sample_size

    effective_seed = (
        random_state
        if random_state is not None
        else settings.RANDOM_SEED
    )

    logger.info("Loading raw dataset from %s", resolved_path)

    try:
        # -------------------------------------------------------------
        # Full dataset mode
        # -------------------------------------------------------------
        if effective_sample_size is None:
            df = pd.read_csv(resolved_path)

            _validate_application_train(df, resolved_path)

            logger.info(
                "Using full dataset: %d rows, %d columns",
                df.shape[0],
                df.shape[1],
            )

            return df

        # Validate the configured sample size.
        if not isinstance(effective_sample_size, int):
            raise DatasetValidationError(
                "DATA_SAMPLE_SIZE must be a positive integer or blank/None. "
                f"Received: {effective_sample_size!r}"
            )

        if effective_sample_size <= 0:
            raise DatasetValidationError(
                "DATA_SAMPLE_SIZE must be greater than 0. "
                f"Received: {effective_sample_size}"
            )

        # Count rows without loading the complete CSV into memory.
        total_rows = _count_data_rows(resolved_path)

        if total_rows == 0:
            raise DatasetValidationError(
                f"Loaded dataset from {resolved_path} is empty."
            )

        # If the requested sample includes the entire dataset,
        # simply load everything.
        if effective_sample_size >= total_rows:
            logger.info(
                "Requested sample size (%d) is greater than or equal to "
                "dataset size (%d). Loading the full dataset.",
                effective_sample_size,
                total_rows,
            )

            df = pd.read_csv(resolved_path)

            _validate_application_train(df, resolved_path)

            logger.info(
                "Using full dataset: %d rows, %d columns",
                df.shape[0],
                df.shape[1],
            )

            return df

        # -------------------------------------------------------------
        # Memory-efficient sampling mode
        # -------------------------------------------------------------
        logger.info(
            "Sampling %d of %d rows during CSV loading "
            "(random_state=%d) as configured via DATA_SAMPLE_SIZE",
            effective_sample_size,
            total_rows,
            effective_seed,
        )

        df = _read_sampled_csv(
            path=resolved_path,
            sample_size=effective_sample_size,
            random_state=effective_seed,
            total_rows=total_rows,
        )

        _validate_application_train(df, resolved_path)

        logger.info(
            "Loaded sampled dataset: %d rows, %d columns",
            df.shape[0],
            df.shape[1],
        )

        return df

    except (DatasetNotFoundError, DatasetValidationError):
        raise

    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read CSV at %s: %s", resolved_path, exc)

        raise DatasetValidationError(
            f"Could not read the dataset file at {resolved_path}. "
            f"It may be corrupted or not a valid CSV."
        ) from exc


def _validate_application_train(
    df: pd.DataFrame,
    source_path: Path,
) -> None:
    """
    Perform basic structural checks so downstream code can trust the dataset.
    """
    if df.empty:
        message = f"Loaded dataset from {source_path} is empty."
        logger.error(message)
        raise DatasetValidationError(message)

    missing_required = [
        column
        for column in (ID_COLUMN, TARGET_COLUMN)
        if column not in df.columns
    ]

    if missing_required:
        message = (
            f"Dataset at {source_path} is missing required column(s): "
            f"{missing_required}. Confirm this is the correct "
            f"application_train.csv from the Home Credit Default Risk "
            f"competition."
        )
        logger.error(message)
        raise DatasetValidationError(message)

    unexpected_targets = (
        set(df[TARGET_COLUMN].dropna().unique()) - {0, 1}
    )

    if unexpected_targets:
        logger.warning(
            "TARGET column contains unexpected values beyond {0, 1}: %s",
            unexpected_targets,
        )


def load_column_description(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the official HomeCredit_columns_description.csv reference file.

    The official Kaggle file may use a non-UTF-8 encoding, so several
    encodings are attempted.

    Returns an empty DataFrame if the description file is unavailable because
    it is helpful for business readability but is not required for model
    training or inference.
    """
    resolved_path = (
        Path(path)
        if path is not None
        else settings.COLUMN_DESCRIPTION_PATH
    )

    if not resolved_path.exists():
        logger.warning(
            "Column description file not found at %s. Feature categorization "
            "will proceed using column-name-based rules only, without "
            "official descriptions.",
            resolved_path,
        )

        return pd.DataFrame(
            columns=["Table", "Row", "Description", "Special"]
        )

    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            desc_df = pd.read_csv(
                resolved_path,
                encoding=encoding,
            )

            logger.info(
                "Loaded column description file from %s "
                "(encoding=%s, %d rows)",
                resolved_path,
                encoding,
                len(desc_df),
            )

            return desc_df

        except UnicodeDecodeError:
            continue

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to read column description file: %s",
                exc,
            )

            return pd.DataFrame(
                columns=["Table", "Row", "Description", "Special"]
            )

    logger.error(
        "Could not decode column description file at %s "
        "with any known encoding.",
        resolved_path,
    )

    return pd.DataFrame(
        columns=["Table", "Row", "Description", "Special"]
    )


def get_column_description_map(
    description_df: pd.DataFrame,
    table_filter: str | None = "application_{train|test}.csv",
) -> dict[str, str]:
    """
    Build a {column_name: description} lookup from the description file.

    Args:
        description_df:
            Output of load_column_description().

        table_filter:
            If provided and the 'Table' column exists, restrict to rows
            matching this value. Home Credit's file uses the literal string
            'application_{train|test}.csv' for application columns.
            Pass None to disable filtering.
    """
    if description_df.empty or "Row" not in description_df.columns:
        return {}

    df = description_df

    if table_filter is not None and "Table" in df.columns:
        filtered = df[df["Table"] == table_filter]

        if not filtered.empty:
            df = filtered

    if "Description" not in df.columns:
        return {}

    return dict(
        zip(
            df["Row"].astype(str),
            df["Description"].astype(str),
        )
    )


def check_dataset_availability() -> dict[str, bool | str]:
    """
    Non-raising status check for use in the UI.

    Returns:
        Dictionary containing dataset availability and expected paths.
    """
    return {
        "raw_data_available": settings.RAW_DATA_PATH.exists(),
        "column_description_available": (
            settings.COLUMN_DESCRIPTION_PATH.exists()
        ),
        "raw_data_path": str(settings.RAW_DATA_PATH),
        "column_description_path": str(
            settings.COLUMN_DESCRIPTION_PATH
        ),
    }