"""Automated health and verification check runner for COVID-19 Epidemiology Platform."""

import sys
from pathlib import Path
import joblib
import pandas as pd


def verify_all():
    print("=" * 60)
    print("🔍 RUNNING EPIDEMIOLOGY PLATFORM VERIFICATION CHECKS")
    print("=" * 60)

    # 1. Check datasets
    raw_csv = Path("data/raw/covid19_global_raw.csv")
    cleaned_csv = Path("data/processed/covid19_cleaned.csv")
    regional_csv = Path("data/processed/covid19_regional_summary.csv")

    assert raw_csv.exists(), f"Missing {raw_csv}"
    assert cleaned_csv.exists(), f"Missing {cleaned_csv}"
    assert regional_csv.exists(), f"Missing {regional_csv}"
    print("✓ All raw and processed datasets verified.")

    # 2. Check model artifact
    model_path = Path("models/covid_risk_clustering_pipeline.joblib")
    assert model_path.exists(), f"Missing {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Loaded model missing predict interface"
    print("✓ Serialized risk clustering model pipeline verified.")

    # 3. Check notebooks
    nb1 = Path("notebooks/01_global_epidemiology_eda.ipynb")
    nb2 = Path("notebooks/02_risk_clustering_and_severity.ipynb")
    assert nb1.exists() and nb1.stat().st_size > 50000, "01 EDA notebook not pre-rendered"
    assert nb2.exists() and nb2.stat().st_size > 50000, "02 Clustering notebook not pre-rendered"
    print("✓ Pre-rendered Jupyter notebooks verified.")

    print("\n✓ ALL 3/3 VERIFICATION CHECKS PASSED.")


if __name__ == "__main__":
    verify_all()
