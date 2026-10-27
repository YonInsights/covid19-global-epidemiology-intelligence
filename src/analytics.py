"""Regional aggregation, macro surveillance KPIs, and outbreak ranking utilities."""

from typing import Dict, Union
import pandas as pd
from src.epidemiology import enrich_epidemiological_metrics


def aggregate_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate subnational records to unique country/region levels.

    Rolls up cumulative counts and recomputes exact macro epidemiological rates.

    Args:
        df: Cleaned COVID-19 DataFrame.

    Returns:
        DataFrame aggregated at the Region level.
    """
    aggregated = (
        df.groupby("Region", as_index=False)[["Confirmed", "Deaths", "Recovered"]]
        .sum()
        .sort_values(by="Confirmed", ascending=False)
        .reset_index(drop=True)
    )
    return enrich_epidemiological_metrics(aggregated)


def get_global_kpis(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Compute headline worldwide epidemiological surveillance indicators.

    Args:
        df: COVID-19 DataFrame.

    Returns:
        Dictionary containing global sums and population-level rates.
    """
    total_confirmed = int(df["Confirmed"].sum())
    total_deaths = int(df["Deaths"].sum())
    total_recovered = int(df["Recovered"].sum())
    total_active = max(0, total_confirmed - (total_deaths + total_recovered))
    global_cfr = round((total_deaths / total_confirmed) * 100.0, 2) if total_confirmed > 0 else 0.0
    global_recovery_rate = (
        round((total_recovered / total_confirmed) * 100.0, 2) if total_confirmed > 0 else 0.0
    )

    return {
        "total_confirmed": total_confirmed,
        "total_deaths": total_deaths,
        "total_recovered": total_recovered,
        "total_active": total_active,
        "global_cfr_pct": global_cfr,
        "global_recovery_rate_pct": global_recovery_rate,
        "total_countries_affected": int(df["Region"].nunique()),
    }


def get_top_outbreaks(
    df_regional: pd.DataFrame, metric: str = "Confirmed", n: int = 10
) -> pd.DataFrame:
    """Retrieve top N regions sorted by specified epidemiological signal.

    Args:
        df_regional: Aggregated regional DataFrame.
        metric: Column name to sort on (e.g. 'Confirmed', 'Deaths', 'Active_Cases', 'Severity_Risk_Index').
        n: Number of records to return.

    Returns:
        Top N DataFrame.
    """
    if metric not in df_regional.columns:
        raise ValueError(f"Sort metric '{metric}' not found in DataFrame.")

    return df_regional.sort_values(by=metric, ascending=False).head(n).reset_index(drop=True)
