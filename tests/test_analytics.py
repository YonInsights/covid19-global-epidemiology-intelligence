"""Unit tests for regional rollups, ranking, and macro surveillance KPIs."""

import pandas as pd
import pytest
from src.analytics import aggregate_by_region, get_global_kpis, get_top_outbreaks


@pytest.fixture
def raw_subnational_df() -> pd.DataFrame:
    """Fixture with subnational records for the same region."""
    return pd.DataFrame(
        {
            "Region": ["US", "US", "Italy"],
            "State": ["New York", "California", "Lombardy"],
            "Confirmed": [300000, 50000, 200000],
            "Deaths": [24000, 2000, 27000],
            "Recovered": [50000, 10000, 75000],
        }
    )


def test_aggregate_by_region(raw_subnational_df):
    """Verify subnational records roll up accurately into single regional rows."""
    regional = aggregate_by_region(raw_subnational_df)
    assert len(regional) == 2  # US and Italy

    us_row = regional[regional["Region"] == "US"].iloc[0]
    assert us_row["Confirmed"] == 350000
    assert us_row["Deaths"] == 26000
    assert us_row["Recovered"] == 60000
    assert us_row["Active_Cases"] == 264000  # 350000 - 86000
    assert "CFR_Pct" in regional.columns


def test_get_global_kpis(raw_subnational_df):
    """Verify worldwide sums and rate calculations."""
    kpis = get_global_kpis(raw_subnational_df)
    assert kpis["total_confirmed"] == 550000
    assert kpis["total_deaths"] == 53000
    assert kpis["total_recovered"] == 135000
    assert kpis["total_active"] == 362000
    assert kpis["total_countries_affected"] == 2
    assert kpis["global_cfr_pct"] > 0


def test_get_top_outbreaks(raw_subnational_df):
    """Verify sorting and ranking of top outbreaks."""
    regional = aggregate_by_region(raw_subnational_df)
    top1 = get_top_outbreaks(regional, metric="Confirmed", n=1)
    assert len(top1) == 1
    assert top1.loc[0, "Region"] == "US"
