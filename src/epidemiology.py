"""Epidemiological metric calculations for pandemic monitoring."""

import numpy as np
import pandas as pd


def calculate_active_cases(df: pd.DataFrame) -> pd.Series:
    """Calculate currently active infection counts.

    Formula: Active = Confirmed - (Deaths + Recovered)
    """
    active = df["Confirmed"] - (df["Deaths"] + df["Recovered"])
    return active.clip(lower=0)


def calculate_case_fatality_rate(df: pd.DataFrame) -> pd.Series:
    """Calculate Case Fatality Rate (CFR %) per confirmed cohort.

    Formula: CFR (%) = (Deaths / Confirmed) * 100
    """
    cfr = np.where(df["Confirmed"] > 0, (df["Deaths"] / df["Confirmed"]) * 100.0, 0.0)
    return pd.Series(cfr, index=df.index, dtype=float).round(2)


def calculate_recovery_rate(df: pd.DataFrame) -> pd.Series:
    """Calculate Recovery Rate (%) relative to confirmed cohort.

    Formula: Recovery Rate (%) = (Recovered / Confirmed) * 100
    """
    rec = np.where(df["Confirmed"] > 0, (df["Recovered"] / df["Confirmed"]) * 100.0, 0.0)
    return pd.Series(rec, index=df.index, dtype=float).round(2)


def calculate_closed_case_mortality(df: pd.DataFrame) -> pd.Series:
    """Calculate mortality among resolved/closed infection outcomes.

    Formula: Closed Mortality (%) = (Deaths / (Deaths + Recovered)) * 100
    """
    closed = df["Deaths"] + df["Recovered"]
    mortality = np.where(closed > 0, (df["Deaths"] / closed) * 100.0, 0.0)
    return pd.Series(mortality, index=df.index, dtype=float).round(2)


def calculate_severity_risk_index(df: pd.DataFrame) -> pd.Series:
    """Compute normalized composite Epidemiological Severity Risk Index.

    Combines log-scale active infection pressure with case fatality rate:
    Formula: Severity = log1p(Active) * (1 + (CFR / 100))
    """
    active = calculate_active_cases(df)
    cfr = calculate_case_fatality_rate(df)
    score = np.log1p(active) * (1.0 + (cfr / 100.0))
    return pd.Series(score, index=df.index, dtype=float).round(2)


def enrich_epidemiological_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich dataset with complete suite of public health metrics.

    Args:
        df: Cleaned COVID-19 DataFrame.

    Returns:
        DataFrame populated with Active_Cases, CFR_Pct, Recovery_Rate_Pct,
        Closed_Mortality_Pct, Severity_Risk_Index, and Health_Risk_Tier.
    """
    enriched = df.copy()

    enriched["Active_Cases"] = calculate_active_cases(enriched)
    enriched["CFR_Pct"] = calculate_case_fatality_rate(enriched)
    enriched["Recovery_Rate_Pct"] = calculate_recovery_rate(enriched)
    enriched["Closed_Cases"] = enriched["Deaths"] + enriched["Recovered"]
    enriched["Closed_Mortality_Pct"] = calculate_closed_case_mortality(enriched)
    enriched["Severity_Risk_Index"] = calculate_severity_risk_index(enriched)

    # Classify initial risk tiers based on Severity Index
    def assign_tier(score: float) -> str:
        if score >= 12.0:
            return "Critical Outbreak"
        elif score >= 8.0:
            return "High Transmission"
        elif score >= 4.0:
            return "Moderate Surveillance"
        else:
            return "Low / Contained"

    enriched["Health_Risk_Tier"] = enriched["Severity_Risk_Index"].apply(assign_tier)
    return enriched
