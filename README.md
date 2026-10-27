# 🌐 COVID-19 Global Epidemiology Intelligence & Health Risk Analytics

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-14%20passed%20%7C%20100%25-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-black.svg)](https://pep8.org/)

An enterprise-grade **Global Pandemic Surveillance & Health Risk Analytics Platform**. This project ingests worldwide COVID-19 epidemiological data, standardizes subnational observations, computes core public health indicators (Active Infection Pressure, Case Fatality Rate, Recovery Velocity, Closed Case Mortality), and performs **unsupervised machine learning risk clustering** across 187 sovereign nations.

---

## 📌 Executive Summary & Public Health Problem

During acute global infectious disease outbreaks, headline cumulative case counts provide an incomplete picture of pandemic severity. Public health authorities, hospital administrators, international aid organizations, and enterprise supply chain risk planners require granular, rate-adjusted intelligence:

### Core Surveillance Questions Addressed:
1. **Active Transmission vs. Closed Outcomes:** How many patients remain actively infected versus resolved cases (clinical recoveries or deaths)?
2. **Case Fatality Rate (CFR) Variance:** Which nations suffer acute clinical mortality strain, and where do testing thresholds artificially suppress or inflate perceived mortality?
3. **Recovery Velocity:** Where are medical discharge protocols and containment interventions successfully outpacing transmission?
4. **Multidimensional Risk Segmentation:** How can sovereign nations be clustered into operational tiers using unsupervised machine learning to optimize medical resource allocation?

---

## 🔬 Epidemiological Methodology & Metrics

To capture the true operational dynamics of pandemic spread, standard public health equations were implemented:

* **Active Infection Burden:**
  ```text
  Active Cases = Confirmed - (Deaths + Recovered)
  ```
* **Case Fatality Rate (CFR %):**
  ```text
  CFR (%) = (Deaths / Confirmed) × 100
  ```
* **Clinical Recovery Velocity (%):**
  ```text
  Recovery Rate (%) = (Recovered / Confirmed) × 100
  ```
* **Closed Outcome Mortality (%):**
  ```text
  Closed Case Mortality (%) = (Deaths / (Deaths + Recovered)) × 100
  ```
* **Composite Severity Risk Index:**
  Combines log-scale active transmission volume with the regional fatality rate:
  ```text
  Severity Risk Index = ln(1 + Active Cases) × (1 + (CFR / 100))
  ```

---

## 📊 Global Pandemic Surveillance Headline KPIs (Snapshot)

* **Worldwide Confirmed Cases:** `3,193,886`
* **Worldwide Documented Deaths:** `227,638`
* **Worldwide Clinical Recoveries:** `972,719`
* **Active Infectious Load:** `1,993,529` (62.4% of total cumulative cases were actively infectious)
* **Global Case Fatality Rate (CFR):** `7.13%`
* **Global Recovery Velocity:** `30.46%`
* **Sovereign Nations / Territories Monitored:** `187`

---

## 🚨 Top 10 Severest Outbreak Epicenters

The table below presents the primary outbreak centers ranked by confirmed case volume at the critical snapshot:

| Rank | Country / Region | Confirmed Cases | Deaths | Recovered | Active Infections | CFR (%) | Recovery Rate (%) | Risk Cluster Profile |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **US** | 1,039,909 | 60,967 | 120,720 | 858,222 | 5.86% | 11.61% | `Tier 1: Critical Outbreak Epicenters` |
| 2 | **Spain** | 236,899 | 24,275 | 132,929 | 79,695 | 10.25% | 56.11% | `Tier 1: Critical Outbreak Epicenters` |
| 3 | **Italy** | 203,591 | 27,682 | 71,252 | 104,657 | 13.60% | 35.00% | `Tier 1: Critical Outbreak Epicenters` |
| 4 | **France** | 166,543 | 24,121 | 49,118 | 93,304 | 14.48% | 29.49% | `Tier 1: Critical Outbreak Epicenters` |
| 5 | **UK** | 166,441 | 26,166 | 857 | 139,418 | 15.72% | 0.52% | `Tier 1: Critical Outbreak Epicenters` |
| 6 | **Germany** | 161,539 | 6,467 | 120,400 | 34,672 | 4.00% | 74.53% | `Tier 1: Critical Outbreak Epicenters` |
| 7 | **Turkey** | 117,589 | 3,081 | 44,022 | 70,486 | 2.62% | 37.44% | `Tier 1: Critical Outbreak Epicenters` |
| 8 | **Russia** | 99,399 | 969 | 10,286 | 88,144 | 0.97% | 10.35% | `Tier 1: Critical Outbreak Epicenters` |
| 9 | **Iran** | 93,657 | 5,957 | 73,791 | 13,909 | 6.36% | 78.79% | `Tier 1: Critical Outbreak Epicenters` |
| 10 | **Mainland China** | 82,862 | 4,633 | 77,610 | 619 | 5.59% | 93.66% | `Tier 1: Critical Outbreak Epicenters` |

---

## 🧠 Unsupervised Machine Learning Health Risk Clustering

Using **StandardScaler** and **K-Means Clustering** ($k=4$), all 187 sovereign territories were segmented into multidimensional operational tiers:

```text
Log(Confirmed) + Log(Active) + CFR (%) + Recovery (%) + Severity Index ──> [StandardScaler] ──> [K-Means (k=4)]
```

| Cluster Tier | Sovereign Nations | Cumulative Infections | Avg. CFR (%) | Avg. Recovery Rate (%) | Operational Health Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: Critical Outbreak Epicenters** | **47** | 2,964,939 | 5.54% | 34.62% | Acute hospital stress, massive active cases, high ICU ventilator requirements. |
| **Tier 2: High Transmission Surge** | **74** | 218,242 | 2.87% | 40.15% | Accelerating community transmission, urgent contact tracing and bed expansion needed. |
| **Tier 3: Moderate Active Surveillance** | **49** | 10,009 | 1.18% | 51.80% | Manageable transmission rate, localized containment measures effective. |
| **Tier 4: Contained / Low Impact** | **17** | 696 | 11.63% | 78.40% | High recovery or isolated mortality anomalies, stabilized epidemic curves. |

---

## 🗂️ Project Structure

```text
covid19-global-epidemiology-intelligence/
├── data/
│   ├── raw/
│   │   └── covid19_global_raw.csv              # Raw global surveillance records
│   └── processed/
│       ├── covid19_cleaned.csv                 # Cleaned subnational and national observations
│       ├── covid19_regional_summary.csv        # Country-level aggregated surveillance dataset
│       └── covid19_clustered_risk_profiles.csv # Dataset enriched with K-Means risk tiers
├── notebooks/
│   ├── 01_global_epidemiology_eda.ipynb        # Pre-rendered EDA, CFR plots & distributions
│   ├── 02_risk_clustering_and_severity.ipynb   # Pre-rendered K-Means clustering & scatter plots
│   └── legacy_project_6.ipynb                  # Archived original exploration notebook
├── src/
│   ├── __init__.py                             # Package metadata and versioning
│   ├── data_loader.py                          # Ingestion, validation, and missing state imputation
│   ├── epidemiology.py                         # Active cases, CFR, recovery velocity, and severity index
│   ├── analytics.py                            # Country rollups, global KPIs, and ranking utilities
│   ├── clustering.py                           # K-Means clustering pipeline and profile mapping
│   └── main.py                                 # Main CLI execution pipeline
├── models/
│   └── covid_risk_clustering_pipeline.joblib   # Exported unsupervised risk cluster pipeline
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py                     # Schema validation and sanitization tests
│   ├── test_epidemiology.py                    # Rate calculation and formula tests
│   ├── test_analytics.py                       # Regional aggregation and KPI tests
│   ├── test_clustering.py                     # ML pipeline and cluster integrity tests
│   ├── test_cli_pipeline.py                    # End-to-end integration test
│   └── test_report.py                          # Automated summary reporting test
├── scripts/
│   ├── build_notebooks.py                      # Programmatic notebook generation script
│   └── generate_report.py                      # Markdown table and surveillance report generator
├── venv/                                       # Isolated Python virtual environment
├── .gitignore                                  # Git configuration ignoring venv, cache, and checkpoints
├── LICENSE                                     # MIT License
├── pyproject.toml                              # Package metadata & pytest configuration
├── requirements.txt                            # Pinned dependencies
└── README.md                                   # Executive portfolio documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YonInsights/covid19-global-epidemiology-intelligence.git
cd covid19-global-epidemiology-intelligence
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite (100% Pass)
```bash
pytest -v
```
Output:
```text
tests/test_analytics.py::test_aggregate_by_region PASSED                 [  7%]
tests/test_analytics.py::test_get_global_kpis PASSED                     [ 14%]
tests/test_analytics.py::test_get_top_outbreaks PASSED                   [ 21%]
tests/test_cli_pipeline.py::test_run_surveillance_pipeline PASSED        [ 28%]
tests/test_clustering.py::test_build_clustering_pipeline PASSED          [ 35%]
tests/test_clustering.py::test_fit_and_assign_clusters PASSED            [ 42%]
tests/test_data_loader.py::test_load_raw_covid_data PASSED               [ 50%]
tests/test_data_loader.py::test_clean_covid_data PASSED                  [ 57%]
tests/test_epidemiology.py::test_calculate_active_cases PASSED           [ 64%]
tests/test_epidemiology.py::test_calculate_case_fatality_rate PASSED     [ 71%]
tests/test_epidemiology.py::test_calculate_recovery_rate PASSED          [ 78%]
tests/test_epidemiology.py::test_calculate_closed_case_mortality PASSED  [ 85%]
tests/test_epidemiology.py::test_enrich_epidemiological_metrics PASSED   [ 92%]
tests/test_report.py::test_generate_summary PASSED                       [100%]
============================= 14 passed in 2.05s ==============================
```

### 5. Execute End-to-End Surveillance Pipeline
```bash
python -m src.main
```

---

## 💻 Python API Usage (Surveillance Example)

Analyzing new epidemiological data using the modular `src` library:

```python
import pandas as pd
from src.epidemiology import enrich_epidemiological_metrics
from src.clustering import load_cluster_model, prepare_cluster_features

# 1. Ingest epidemiological record
new_data = pd.DataFrame([{
    "Region": "Test Territory",
    "Confirmed": 150000,
    "Deaths": 9500,
    "Recovered": 65000
}])

# 2. Compute public health rates
enriched = enrich_epidemiological_metrics(new_data)
print(f"Active Infections: {enriched['Active_Cases'].iloc[0]:,}")
print(f"Case Fatality Rate: {enriched['CFR_Pct'].iloc[0]}%")
print(f"Recovery Velocity: {enriched['Recovery_Rate_Pct'].iloc[0]}%")
print(f"Severity Risk Score: {enriched['Severity_Risk_Index'].iloc[0]}")

# 3. Predict operational risk cluster
pipeline = load_cluster_model("models/covid_risk_clustering_pipeline.joblib")
_, X_trans = prepare_cluster_features(enriched)
cluster_id = pipeline.predict(X_trans)[0]
print(f"Assigned Risk Cluster: {cluster_id}")
```

---

## 📈 Exploratory & Modeling Notebooks

| Notebook | Focus | View Online |
| :--- | :--- | :--- |
| **[`01_global_epidemiology_eda.ipynb`](notebooks/01_global_epidemiology_eda.ipynb)** | Data sanitization, headline global KPIs, CFR distributions, and active case concentration. | Pre-rendered with all charts |
| **[`02_risk_clustering_and_severity.ipynb`](notebooks/02_risk_clustering_and_severity.ipynb)** | Feature standardization, K-Means clustering, risk profiling, and resource allocation recommendations. | Pre-rendered with all charts |

---

## 👤 Author & Portfolio

* **Portfolio:** [Yonatan Insights](https://github.com/YonInsights)
* **GitHub Repository:** [covid19-global-epidemiology-intelligence](https://github.com/YonInsights/covid19-global-epidemiology-intelligence)
* **Contact:** Open an issue or connect on LinkedIn.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
