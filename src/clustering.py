"""Unsupervised machine learning clustering for regional pandemic risk profiling."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLUSTERING_FEATURES = [
    "Confirmed",
    "Active_Cases",
    "CFR_Pct",
    "Recovery_Rate_Pct",
    "Severity_Risk_Index",
]


def prepare_cluster_features(
    df: pd.DataFrame, features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transform skewed count signals with log1p prior to standardization.

    Args:
        df: Enriched regional DataFrame.
        features: List of feature names to use.

    Returns:
        Tuple of (original_df_subset, transformed_feature_df).
    """
    feats = features or CLUSTERING_FEATURES
    X = df[feats].copy()

    # Log-transform raw counts to mitigate extreme power-law dispersion
    X_trans = X.copy()
    if "Confirmed" in X_trans.columns:
        X_trans["Confirmed"] = np.log1p(X_trans["Confirmed"])
    if "Active_Cases" in X_trans.columns:
        X_trans["Active_Cases"] = np.log1p(X_trans["Active_Cases"])

    return X, X_trans


def build_clustering_pipeline(n_clusters: int = 4, random_state: int = 42) -> Pipeline:
    """Build StandardScaler + KMeans unsupervised learning pipeline.

    Args:
        n_clusters: Number of risk clusters (default 4).
        random_state: Random state for deterministic repeatability.

    Returns:
        Configured Pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=10,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_and_assign_clusters(
    df_regional: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> Tuple[pd.DataFrame, Pipeline]:
    """Fit clustering pipeline and assign interpretable pandemic risk profiles.

    Args:
        df_regional: Regional aggregated DataFrame.
        n_clusters: Target cluster count.
        random_state: Random seed.

    Returns:
        Tuple of (clustered_dataframe, fitted_pipeline).
    """
    clustered = df_regional.copy()
    _, X_trans = prepare_cluster_features(clustered)

    pipeline = build_clustering_pipeline(n_clusters=n_clusters, random_state=random_state)
    raw_clusters = pipeline.fit_predict(X_trans)
    clustered["Cluster_ID"] = raw_clusters

    # Map clusters by average severity index to order them logically
    cluster_severity = (
        clustered.groupby("Cluster_ID")["Severity_Risk_Index"].mean().sort_values(ascending=False)
    )

    cluster_labels = [
        "Tier 1: Critical Outbreak Epicenters",
        "Tier 2: High Transmission Surge",
        "Tier 3: Moderate Active Surveillance",
        "Tier 4: Contained / Low Impact",
    ]

    tier_map = {}
    for rank, cluster_id in enumerate(cluster_severity.index):
        tier_map[cluster_id] = cluster_labels[rank] if rank < len(cluster_labels) else f"Tier {rank+1}"

    clustered["Risk_Profile_Cluster"] = clustered["Cluster_ID"].map(tier_map)
    return clustered, pipeline


def save_cluster_model(pipeline: Pipeline, filepath: Union[str, Path]) -> Path:
    """Serialize fitted cluster model to disk.

    Args:
        pipeline: Fitted pipeline object.
        filepath: Destination path.

    Returns:
        Path of written artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_cluster_model(filepath: Union[str, Path]) -> Pipeline:
    """Load serialized cluster pipeline.

    Args:
        filepath: Filepath to joblib file.

    Returns:
        Loaded Pipeline.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)
