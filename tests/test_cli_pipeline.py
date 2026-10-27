"""Integration test for the end-to-end command-line surveillance pipeline."""

from pathlib import Path
import pytest
from src.main import run_surveillance_pipeline


def test_run_surveillance_pipeline(tmp_path):
    """Verify complete pipeline execution, KPI generation, and artifact export."""
    temp_processed = tmp_path / "processed"
    temp_model = tmp_path / "models" / "test_cluster.joblib"

    kpis = run_surveillance_pipeline(
        raw_data_path="data/raw/covid19_global_raw.csv",
        processed_dir=str(temp_processed),
        model_output_path=str(temp_model),
        n_clusters=4,
    )

    assert isinstance(kpis, dict)
    assert kpis["total_confirmed"] > 3000000
    assert kpis["total_countries_affected"] > 180

    assert (temp_processed / "covid19_cleaned.csv").exists()
    assert (temp_processed / "covid19_regional_summary.csv").exists()
    assert (temp_processed / "covid19_clustered_risk_profiles.csv").exists()
    assert temp_model.exists()
