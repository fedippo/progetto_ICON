"""Preprocessing pipeline for the Steam Games dataset."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import pandas as pd

try:
    from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler
except ModuleNotFoundError:
    KBinsDiscretizer = None
    MinMaxScaler = None

from config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    DISCRETIZATION_BINS,
    DISCRETIZED_DATA_PATH,
    MIN_REVIEW_COUNT,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
    PROJECT_COLUMNS,
    RANDOM_STATE,
    RAW_COLUMNS,
    RAW_DATA_PATH,
    SAMPLE_SIZE,
)


def build_corrected_header(path: str = RAW_DATA_PATH) -> list[str]:
    """Split the merged 'DiscountDLC count' header into two real columns."""
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        original_header = next(csv.reader(csv_file))

    corrected_header: list[str] = []
    for column in original_header:
        if column == "DiscountDLC count":
            corrected_header.extend(["Discount", "DLC count"])
        else:
            corrected_header.append(column)

    return corrected_header


def load_dataset(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset with the corrected header."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}.")

    corrected_header = build_corrected_header(path)
    return pd.read_csv(
        data_path,
        header=0,
        names=corrected_header,
        usecols=RAW_COLUMNS,
        low_memory=False,
    )


def parse_language_count(value: object) -> int:
    """Count supported languages stored as a stringified list."""
    if not isinstance(value, str) or value.strip() in {"", "[]"}:
        return 0

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return len([item for item in value.split(",") if item.strip()])

    if isinstance(parsed, list):
        return len(parsed)
    return 0


def extract_primary_genre(value: object) -> str | None:
    """Use the first Steam genre as the main genre."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.split(",")[0].strip()


def has_multiplayer(value: object) -> int:
    """Derive a binary multiplayer feature from Steam categories."""
    if not isinstance(value, str):
        return 0
    categories = value.lower()
    multiplayer_markers = [
        "multi-player",
        "multiplayer",
        "co-op",
        "pvp",
        "online co-op",
        "lan co-op",
    ]
    return int(any(marker in categories for marker in multiplayer_markers))


def derive_project_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the feature set used by the project."""
    derived = df.copy()

    numeric_columns = [
        "Price",
        "Positive",
        "Negative",
        "Average playtime forever",
        "Median playtime forever",
    ]
    for column in numeric_columns:
        derived[column] = pd.to_numeric(derived[column], errors="coerce").fillna(0)

    review_count = derived["Positive"] + derived["Negative"]
    derived["Review_Count"] = review_count
    derived["Review_Score_Pct"] = (
        derived["Positive"].div(review_count).fillna(0).mul(100)
    )
    derived["Primary_Genre"] = derived["Genres"].apply(extract_primary_genre)
    derived["Languages_Count"] = derived["Supported languages"].apply(
        parse_language_count
    )
    derived["Multiplayer"] = derived["Categories"].apply(has_multiplayer)
    derived["Playtime_Hours"] = (
        derived["Average playtime forever"]
        .where(derived["Average playtime forever"] > 0, derived["Median playtime forever"])
        .div(60)
    )

    return derived[PROJECT_COLUMNS].copy()


def clean_and_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Filter weak rows and keep a reproducible sample for the 25-hour scope."""
    cleaned = df.drop_duplicates(subset=["AppID"]).copy()
    cleaned = cleaned.dropna(subset=["Name", "Primary_Genre"])
    cleaned = cleaned[cleaned["Review_Count"] >= MIN_REVIEW_COUNT]
    cleaned = cleaned[cleaned["Price"].between(0, 100)]
    cleaned = cleaned[cleaned["Playtime_Hours"] >= 0]
    cleaned = cleaned.reset_index(drop=True)

    if len(cleaned) > SAMPLE_SIZE:
        cleaned = cleaned.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    return cleaned.reset_index(drop=True)


def normalize_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric features for distance-based models."""
    normalized = df.copy()

    if MinMaxScaler is not None:
        scaler = MinMaxScaler()
        normalized[NUMERIC_FEATURES] = scaler.fit_transform(normalized[NUMERIC_FEATURES])
        return normalized

    for column in NUMERIC_FEATURES:
        minimum = normalized[column].min()
        maximum = normalized[column].max()
        if maximum == minimum:
            normalized[column] = 0
        else:
            normalized[column] = (normalized[column] - minimum) / (maximum - minimum)

    return normalized


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize numeric values and encode categories for Bayesian Networks."""
    discretized = df.copy()

    for column, bins in DISCRETIZATION_BINS.items():
        if column not in discretized.columns:
            continue

        unique_values = discretized[column].nunique(dropna=True)
        if unique_values < 2:
            discretized[column] = 0
            continue

        n_bins = min(bins, unique_values)

        if KBinsDiscretizer is not None:
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            discretized[column] = discretizer.fit_transform(
                discretized[[column]]
            ).astype(int)
        else:
            discretized[column] = pd.qcut(
                discretized[column],
                q=n_bins,
                labels=False,
                duplicates="drop",
            ).fillna(0).astype(int)

    for column in CATEGORICAL_FEATURES:
        if column in discretized.columns:
            discretized[column] = discretized[column].astype("category").cat.codes

    return discretized


def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline and save all intermediate datasets."""
    raw_df = load_dataset()
    project_df = derive_project_features(raw_df)
    clean_df = clean_and_sample(project_df)
    normalized_df = normalize_numeric_features(clean_df)
    discretized_df = discretize_features(clean_df)

    Path(CLEAN_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    normalized_df.to_csv(PROCESSED_DATA_PATH, index=False)
    discretized_df.to_csv(DISCRETIZED_DATA_PATH, index=False)

    return clean_df, normalized_df, discretized_df


if __name__ == "__main__":
    clean, normalized, discretized = run_preprocessing()
    print(f"Clean dataset shape: {clean.shape}")
    print(f"Normalized dataset shape: {normalized.shape}")
    print(f"Discretized dataset shape: {discretized.shape}")
