"""Unit tests for unsupervised epidemiological risk clustering."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.clustering import build_clustering_pipeline, fit_and_assign_clusters


@pytest.fixture
def mock_regional_df() -> pd.DataFrame:
    """Fixture with varied outbreak profiles."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Region": [f"Country {i}" for i in range(20)],
            "Confirmed": np.random.randint(100, 500000, size=20),
            "Active_Cases": np.random.randint(50, 300000, size=20),
            "CFR_Pct": np.random.uniform(0.5, 15.0, size=20),
            "Recovery_Rate_Pct": np.random.uniform(10.0, 90.0, size=20),
            "Severity_Risk_Index": np.random.uniform(2.0, 15.0, size=20),
        }
    )


def test_build_clustering_pipeline():
    """Verify unsupervised learning pipeline construction."""
    pipe = build_clustering_pipeline(n_clusters=3)
    assert isinstance(pipe, Pipeline)
    assert "scaler" in pipe.named_steps
    assert "kmeans" in pipe.named_steps


def test_fit_and_assign_clusters(mock_regional_df):
    """Verify cluster assignment, label ordering, and pipeline repeatability."""
    clustered, model = fit_and_assign_clusters(mock_regional_df, n_clusters=4, random_state=42)

    assert "Cluster_ID" in clustered.columns
    assert "Risk_Profile_Cluster" in clustered.columns
    assert len(clustered) == len(mock_regional_df)
    assert clustered["Cluster_ID"].nunique() == 4

    # Verify cluster labels adhere to defined tiers
    expected_tiers = {
        "Tier 1: Critical Outbreak Epicenters",
        "Tier 2: High Transmission Surge",
        "Tier 3: Moderate Active Surveillance",
        "Tier 4: Contained / Low Impact",
    }
    assert set(clustered["Risk_Profile_Cluster"].unique()).issubset(expected_tiers)
