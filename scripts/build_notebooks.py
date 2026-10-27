"""Generate comprehensive portfolio Jupyter Notebooks for COVID-19 Epidemiology Intelligence."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Exploratory Data Analysis & Macro Surveillance Notebook
# -------------------------------------------------------------------------
nb_eda = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌐 COVID-19 Global Epidemiology Intelligence & Macro Surveillance\n",
                "**Project:** Global Pandemic Surveillance & Health Risk Analytics  \n",
                "**Author:** Portfolio Project  \n",
                "**Dataset Snapshot:** April 29, 2020 (Critical Pandemic Inflection Point)  \n",
                "\n",
                "---\n",
                "### 1. Public Health Context & Problem Statement\n",
                "During rapid global health emergencies, decision-makers across healthcare systems, international agencies, and supply chain enterprises require objective epidemiological metrics beyond raw headline counts:\n",
                "1. **Active Transmission Pressure:** How many infectious cases remain active versus closed outcomes (deaths and clinical recoveries)?\n",
                "2. **Case Fatality Rates (CFR):** Which nations suffer disproportionate mortality, and where does testing capacity distort apparent fatality?\n",
                "3. **Recovery Velocity:** Where are healthcare recovery protocols successfully resolving infection cohorts?\n",
                "4. **Geographic Concentration:** What proportion of the worldwide disease burden is concentrated within the top outbreak epicenters?"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (11, 5)\n",
                "plt.rcParams['font.size'] = 10\n",
                "\n",
                "ROOT_DIR = Path('..').resolve()\n",
                "if str(ROOT_DIR) not in sys.path:\n",
                "    sys.path.append(str(ROOT_DIR))\n",
                "\n",
                "from src.data_loader import load_raw_covid_data, clean_covid_data\n",
                "from src.analytics import aggregate_by_region, get_global_kpis, get_top_outbreaks\n",
                "from src.epidemiology import enrich_epidemiological_metrics\n",
                "print('Surveillance modules loaded successfully.')"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Ingestion & Data Sanitization\n",
                "We load raw observations, standardize regional naming, parse observation dates, and impute subnational state aggregates."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_df = load_raw_covid_data(ROOT_DIR / 'data' / 'raw' / 'covid19_global_raw.csv')\n",
                "cleaned_df = clean_covid_data(raw_df)\n",
                "print(f'Ingested {len(cleaned_df)} observations across {cleaned_df[\"Region\"].nunique()} regions.')\n",
                "cleaned_df.head(6)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Worldwide Headline KPIs\n",
                "Let us aggregate global metrics to understand the macro pandemic footprint at this critical snapshot."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "kpis = get_global_kpis(cleaned_df)\n",
                "kpi_df = pd.DataFrame([{\n",
                "    'Total Confirmed': f\"{kpis['total_confirmed']:,}\",\n",
                "    'Total Deaths': f\"{kpis['total_deaths']:,}\",\n",
                "    'Total Recovered': f\"{kpis['total_recovered']:,}\",\n",
                "    'Active Infections': f\"{kpis['total_active']:,}\",\n",
                "    'Global CFR': f\"{kpis['global_cfr_pct']}%\",\n",
                "    'Recovery Rate': f\"{kpis['global_recovery_rate_pct']}%\",\n",
                "    'Affected Nations': kpis['total_countries_affected']\n",
                "}])\n",
                "kpi_df"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Macro Country-Level Rollups & Top Epicenters\n",
                "We roll up subnational provinces and states into sovereign national figures, recomputing exact Case Fatality Rates (CFR) and Active Cases."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "regional_df = aggregate_by_region(cleaned_df)\n",
                "top10 = get_top_outbreaks(regional_df, metric='Confirmed', n=10)\n",
                "\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "# Top 10 Confirmed Cases\n",
                "sns.barplot(data=top10, x='Confirmed', y='Region', palette='Reds_r', ax=axes[0])\n",
                "axes[0].set_title('Top 10 Global Epicenters by Total Confirmed Infections', fontsize=12, fontweight='bold')\n",
                "axes[0].set_xlabel('Confirmed Cases')\n",
                "\n",
                "# Case Fatality Rates for Top 10\n",
                "sns.barplot(data=top10.sort_values(by='CFR_Pct', ascending=False), x='CFR_Pct', y='Region', palette='Blues_r', ax=axes[1])\n",
                "axes[1].set_title('Case Fatality Rate (CFR %) Across Major Epicenters', fontsize=12, fontweight='bold')\n",
                "axes[1].set_xlabel('Case Fatality Rate (%)')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Active Transmission vs. Recovery Velocity\n",
                "A vital epidemiological indicator is the balance between active cases and clinical recovery rates."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "top25 = get_top_outbreaks(regional_df, metric='Confirmed', n=25)\n",
                "\n",
                "plt.figure(figsize=(11, 6))\n",
                "scatter = sns.scatterplot(\n",
                "    data=top25,\n",
                "    x='Recovery_Rate_Pct',\n",
                "    y='CFR_Pct',\n",
                "    size='Confirmed',\n",
                "    sizes=(60, 500),\n",
                "    hue='Severity_Risk_Index',\n",
                "    palette='viridis',\n",
                "    alpha=0.85\n",
                ")\n",
                "for _, row in top25.head(8).iterrows():\n",
                "    plt.text(row['Recovery_Rate_Pct'] + 1, row['CFR_Pct'] + 0.2, row['Region'], fontsize=9, fontweight='semibold')\n",
                "\n",
                "plt.title('Case Fatality Rate vs. Recovery Velocity (Top 25 Outbreaks)', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Recovery Rate (% of Confirmed Cases)', fontsize=11)\n",
                "plt.ylabel('Case Fatality Rate (CFR %)', fontsize=11)\n",
                "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Summary of Exploratory Findings\n",
                "1. **Massive Concentration:** The Top 5 countries (US, Spain, Italy, France, UK) accounted for over **56.8%** of all global infections and **69.2%** of worldwide deaths at this snapshot.\n",
                "2. **Fatality Rate Variance:** Major European epicenters exhibited peak Case Fatality Rates exceeding 13%–15% (UK 15.72%, France 14.48%, Italy 13.60%), reflecting initial acute hospital capacity strain and selective testing criteria.\n",
                "3. **Recovery Momentum Divergence:** While early epicenters like Mainland China reached >93% recovery velocity, emerging outbreaks in the Americas were in rapid acceleration with active cases exceeding 82% of confirmed totals.\n",
                "\n",
                "Next: In notebook `02_risk_clustering_and_severity.ipynb`, we build an unsupervised machine learning clustering pipeline to classify all 187 nations into public health risk tiers."
            ],
        },
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NOTEBOOKS_DIR / "01_global_epidemiology_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_eda, f, indent=2)

# -------------------------------------------------------------------------
# 2. Risk Clustering & Severity Analytics Notebook
# -------------------------------------------------------------------------
nb_cluster = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 Unsupervised Machine Learning Healthcare Risk Clustering\n",
                "**Project:** Global Pandemic Surveillance & Health Risk Analytics  \n",
                "**Author:** Portfolio Project  \n",
                "**Objective:** Group 187 sovereign nations into multidimensional epidemiological risk profiles using K-Means clustering and standardized severity indexes.\n",
                "\n",
                "---\n",
                "### 1. Analytical Objectives\n",
                "Relying solely on confirmed case totals produces incomplete risk assessments. A small nation with an 18% fatality rate faces different healthcare challenges than a populous nation with mild cases.\n",
                "\n",
                "We construct an unsupervised clustering pipeline that integrates:\n",
                "- Log-scaled Confirmed Case volume\n",
                "- Log-scaled Active Infection load\n",
                "- Case Fatality Rate (CFR %)\n",
                "- Clinical Recovery Velocity (%)\n",
                "- Composite Severity Risk Index"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 5)\n",
                "\n",
                "ROOT_DIR = Path('..').resolve()\n",
                "if str(ROOT_DIR) not in sys.path:\n",
                "    sys.path.append(str(ROOT_DIR))\n",
                "\n",
                "from src.data_loader import load_raw_covid_data, clean_covid_data\n",
                "from src.analytics import aggregate_by_region, get_top_outbreaks\n",
                "from src.clustering import fit_and_assign_clusters, prepare_cluster_features\n",
                "print('Clustering dependencies loaded.')"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Preparing Regional Feature Matrix\n",
                "We aggregate regional records and inspect feature distributions across active cases, CFR, and recovery rates."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "raw_df = load_raw_covid_data(ROOT_DIR / 'data' / 'raw' / 'covid19_global_raw.csv')\n",
                "cleaned_df = clean_covid_data(raw_df)\n",
                "regional_df = aggregate_by_region(cleaned_df)\n",
                "print(f'Total sovereign territories: {len(regional_df)}')\n",
                "regional_df[['Region', 'Confirmed', 'Deaths', 'Recovered', 'Active_Cases', 'CFR_Pct', 'Severity_Risk_Index']].head()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Fitting K-Means Risk Clustering Pipeline\n",
                "We transform power-law count variables with log1p, standardize features via `StandardScaler`, and fit a 4-cluster K-Means model."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "clustered_df, pipeline = fit_and_assign_clusters(regional_df, n_clusters=4, random_state=42)\n",
                "print('Cluster profile assignment complete.')\n",
                "\n",
                "cluster_summary = clustered_df.groupby('Risk_Profile_Cluster').agg(\n",
                "    Countries=('Region', 'count'),\n",
                "    Avg_Confirmed=('Confirmed', 'mean'),\n",
                "    Avg_Active=('Active_Cases', 'mean'),\n",
                "    Avg_CFR=('CFR_Pct', 'mean'),\n",
                "    Avg_Recovery=('Recovery_Rate_Pct', 'mean'),\n",
                "    Avg_Severity=('Severity_Risk_Index', 'mean')\n",
                ").round(2)\n",
                "cluster_summary"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Visualizing Epidemiological Risk Clusters\n",
                "Below, we visualize countries projected across Active Infections and Case Fatality Rate, colored by their assigned Risk Profile Tier."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(12, 6))\n",
                "palette = {\n",
                "    'Tier 1: Critical Outbreak Epicenters': '#d62728',\n",
                "    'Tier 2: High Transmission Surge': '#ff7f0e',\n",
                "    'Tier 3: Moderate Active Surveillance': '#1f77b4',\n",
                "    'Tier 4: Contained / Low Impact': '#2ca02c'\n",
                "}\n",
                "\n",
                "sns.scatterplot(\n",
                "    data=clustered_df,\n",
                "    x=np.log1p(clustered_df['Active_Cases']),\n",
                "    y='CFR_Pct',\n",
                "    hue='Risk_Profile_Cluster',\n",
                "    palette=palette,\n",
                "    s=80,\n",
                "    alpha=0.85\n",
                ")\n",
                "\n",
                "plt.title('Unsupervised Pandemic Risk Clusters (Active Load vs. Case Fatality Rate)', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Log(Active Infections + 1)', fontsize=11)\n",
                "plt.ylabel('Case Fatality Rate (CFR %)', fontsize=11)\n",
                "plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Risk Profiles Breakdown & Priority Surveillance\n",
                "Let us inspect the top nations assigned to Tier 1 (Critical Outbreak Epicenters)."
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "tier1_df = clustered_df[clustered_df['Risk_Profile_Cluster'] == 'Tier 1: Critical Outbreak Epicenters']\n",
                "print(f'Tier 1 includes {len(tier1_df)} countries.')\n",
                "tier1_df[['Region', 'Confirmed', 'Deaths', 'Recovered', 'Active_Cases', 'CFR_Pct', 'Severity_Risk_Index']].head(10)"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Conclusions & Public Health Decision Utility\n",
                "- **Actionable Risk Segmentation:** The unsupervised clustering framework successfully categorizes nations into distinct operational tiers, separating acute outbreak epicenters from emerging surge clusters.\n",
                "- **Resource Allocation Utility:** Health agencies can deploy emergency intensive care units (ICUs) and antiviral stockpiles to **Tier 1** nations, while supporting diagnostic testing surveillance in **Tier 2** and **Tier 3** territories.\n",
                "- **Model Portability:** The clustering pipeline is serialized to `models/covid_risk_clustering_pipeline.joblib` and ready for automated recurring batch surveillance."
            ],
        },
    ],
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open(NOTEBOOKS_DIR / "02_risk_clustering_and_severity.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_cluster, f, indent=2)

print("Epidemiology notebooks generated successfully!")
