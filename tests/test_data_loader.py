"""Unit tests for COVID-19 data loading and schema validation."""

import pandas as pd
import pytest
from src.data_loader import clean_covid_data, load_raw_covid_data


def test_load_raw_covid_data():
    """Verify raw CSV loads with mandatory epidemiological columns."""
    df = load_raw_covid_data("data/raw/covid19_global_raw.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    for expected_col in ["Date", "Region", "Confirmed", "Deaths", "Recovered"]:
        assert expected_col in df.columns


def test_clean_covid_data():
    """Verify missing state imputation, date parsing, and non-negative counts."""
    raw_sample = pd.DataFrame(
        {
            "Date": ["4/29/2020", "4/29/2020"],
            "State": [None, "Hubei"],
            "Region": [" Afghanistan ", "Mainland China"],
            "Confirmed": [1939, 68128],
            "Deaths": [60, 4512],
            "Recovered": [252, 63616],
        }
    )
    cleaned = clean_covid_data(raw_sample)

    afg = cleaned[cleaned["Region"] == "Afghanistan"].iloc[0]
    hubei = cleaned[cleaned["Region"] == "Mainland China"].iloc[0]

    assert afg["State"] == "National / Aggregate"
    assert hubei["State"] == "Hubei"
    assert afg["Region"] == "Afghanistan"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
    assert (cleaned["Confirmed"] >= 0).all()
