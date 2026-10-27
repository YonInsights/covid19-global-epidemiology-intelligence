"""Data ingestion, validation, and sanitization for COVID-19 epidemiological data."""

from pathlib import Path
from typing import Union
import pandas as pd


def load_raw_covid_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load raw COVID-19 dataset from CSV.

    Args:
        filepath: Path to raw CSV file.

    Returns:
        DataFrame containing raw records.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Epidemiological dataset not found at: {path}")

    df = pd.read_csv(path)
    required_cols = {"Date", "Region", "Confirmed", "Deaths", "Recovered"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df


def clean_covid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw COVID-19 dataset.

    - Parses and formats Date column.
    - Imputes missing State / Province names as 'National / Aggregate'.
    - Strips whitespace from geographic names.
    - Ensures non-negative integer counts for Confirmed, Deaths, and Recovered.

    Args:
        df: Raw DataFrame.

    Returns:
        Standardized, cleaned DataFrame.
    """
    cleaned = df.copy()

    # Parse dates
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")

    # Clean geographic text columns
    cleaned["Region"] = cleaned["Region"].astype(str).str.strip()
    if "State" in cleaned.columns:
        cleaned["State"] = (
            cleaned["State"]
            .fillna("National / Aggregate")
            .astype(str)
            .str.strip()
            .replace({"": "National / Aggregate", "nan": "National / Aggregate"})
        )

    # Clean counts: numeric, non-null, non-negative
    metric_cols = ["Confirmed", "Deaths", "Recovered"]
    for col in metric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    # Sort deterministically
    cleaned = cleaned.sort_values(by=["Confirmed", "Deaths"], ascending=False).reset_index(drop=True)
    return cleaned


def save_processed_data(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Persist processed dataset to disk.

    Args:
        df: Cleaned or enriched DataFrame.
        output_path: Target CSV destination path.

    Returns:
        Resolved Path.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest
