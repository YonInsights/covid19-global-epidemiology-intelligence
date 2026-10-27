"""Unit tests for epidemiological rate calculations and health risk tiering."""

import numpy as np
import pandas as pd
import pytest
from src.epidemiology import (
    calculate_active_cases,
    calculate_case_fatality_rate,
    calculate_closed_case_mortality,
    calculate_recovery_rate,
    calculate_severity_risk_index,
    enrich_epidemiological_metrics,
)


@pytest.fixture
def sample_epi_df() -> pd.DataFrame:
    """Fixture with representative pandemic counts."""
    return pd.DataFrame(
        {
            "Region": ["Nation A", "Nation B", "Nation C"],
            "Confirmed": [1000, 500, 0],
            "Deaths": [50, 20, 0],
            "Recovered": [450, 100, 0],
        }
    )


def test_calculate_active_cases(sample_epi_df):
    """Verify Active = Confirmed - (Deaths + Recovered)."""
    active = calculate_active_cases(sample_epi_df)
    assert active.iloc[0] == 500  # 1000 - (50 + 450)
    assert active.iloc[1] == 380  # 500 - (20 + 100)
    assert active.iloc[2] == 0


def test_calculate_case_fatality_rate(sample_epi_df):
    """Verify CFR (%) = (Deaths / Confirmed) * 100."""
    cfr = calculate_case_fatality_rate(sample_epi_df)
    assert cfr.iloc[0] == 5.0  # (50 / 1000) * 100
    assert cfr.iloc[1] == 4.0  # (20 / 500) * 100
    assert cfr.iloc[2] == 0.0  # Division by zero handled


def test_calculate_recovery_rate(sample_epi_df):
    """Verify Recovery Rate (%) = (Recovered / Confirmed) * 100."""
    rec = calculate_recovery_rate(sample_epi_df)
    assert rec.iloc[0] == 45.0  # (450 / 1000) * 100
    assert rec.iloc[1] == 20.0  # (100 / 500) * 100
    assert rec.iloc[2] == 0.0


def test_calculate_closed_case_mortality(sample_epi_df):
    """Verify Closed Mortality (%) = (Deaths / (Deaths + Recovered)) * 100."""
    closed_mort = calculate_closed_case_mortality(sample_epi_df)
    assert np.isclose(closed_mort.iloc[0], 10.0)  # (50 / 500) * 100
    assert np.isclose(closed_mort.iloc[1], 16.67, atol=0.01)  # (20 / 120) * 100


def test_enrich_epidemiological_metrics(sample_epi_df):
    """Verify full dataset enrichment with valid risk tiers."""
    enriched = enrich_epidemiological_metrics(sample_epi_df)
    expected_cols = [
        "Active_Cases",
        "CFR_Pct",
        "Recovery_Rate_Pct",
        "Closed_Cases",
        "Closed_Mortality_Pct",
        "Severity_Risk_Index",
        "Health_Risk_Tier",
    ]
    for col in expected_cols:
        assert col in enriched.columns

    valid_tiers = {"Critical Outbreak", "High Transmission", "Moderate Surveillance", "Low / Contained"}
    assert set(enriched["Health_Risk_Tier"].unique()).issubset(valid_tiers)
