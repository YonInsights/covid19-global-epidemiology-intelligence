"""Command-line execution pipeline for COVID-19 Global Epidemiology Intelligence."""

import argparse
from pathlib import Path
from src.analytics import aggregate_by_region, get_global_kpis, get_top_outbreaks
from src.clustering import fit_and_assign_clusters, save_cluster_model
from src.data_loader import clean_covid_data, load_raw_covid_data, save_processed_data


def run_surveillance_pipeline(
    raw_data_path: str = "data/raw/covid19_global_raw.csv",
    processed_dir: str = "data/processed",
    model_output_path: str = "models/covid_risk_clustering_pipeline.joblib",
    n_clusters: int = 4,
) -> dict:
    """Execute end-to-end data ingestion, rate calculation, clustering, and persistence."""
    print("=" * 70)
    print("🌐 COVID-19 GLOBAL EPIDEMIOLOGY INTELLIGENCE - SURVEILLANCE PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/5] Ingesting raw epidemiological records from '{raw_data_path}'...")
    raw_df = load_raw_covid_data(raw_data_path)
    print(f"      Ingested {len(raw_df)} subnational & national observations.")

    # 2. Clean Data
    print("\n[2/5] Cleaning records and standardizing geographic entities...")
    cleaned_df = clean_covid_data(raw_df)
    save_processed_data(cleaned_df, Path(processed_dir) / "covid19_cleaned.csv")
    print(f"      Persisted cleaned record set to '{processed_dir}/covid19_cleaned.csv'.")

    # 3. Macro Regional Rollups & Rate Calculation
    print("\n[3/5] Computing macro country rollups, active cases, and fatality rates...")
    regional_df = aggregate_by_region(cleaned_df)
    save_processed_data(regional_df, Path(processed_dir) / "covid19_regional_summary.csv")
    print(f"      Aggregated to {len(regional_df)} sovereign nations/regions.")

    # 4. Unsupervised Health Risk Clustering
    print(f"\n[4/5] Fitting unsupervised K-Means risk clustering (k={n_clusters})...")
    clustered_df, cluster_pipe = fit_and_assign_clusters(regional_df, n_clusters=n_clusters)
    save_cluster_model(cluster_pipe, model_output_path)
    save_processed_data(clustered_df, Path(processed_dir) / "covid19_clustered_risk_profiles.csv")
    print(f"      Exported cluster pipeline to '{model_output_path}'.")

    # 5. Global Surveillance Briefing
    print("\n[5/5] Generating Global Pandemic Surveillance Briefing...")
    kpis = get_global_kpis(cleaned_df)
    print("\n" + "=" * 70)
    print("📊 WORLDWIDE EPIDEMIOLOGICAL HEADLINE KPIS (SNAPSHOT)")
    print("=" * 70)
    print(f"   • Total Confirmed Cases:   {kpis['total_confirmed']:>12,d}")
    print(f"   • Total Global Deaths:     {kpis['total_deaths']:>12,d}")
    print(f"   • Total Documented Recov:  {kpis['total_recovered']:>12,d}")
    print(f"   • Total Active Infections: {kpis['total_active']:>12,d}")
    print(f"   • Global Case Fatality (CFR): {kpis['global_cfr_pct']:>9.2f}%")
    print(f"   • Global Recovery Velocity:   {kpis['global_recovery_rate_pct']:>9.2f}%")
    print(f"   • Sovereign Countries Affected: {kpis['total_countries_affected']:>7,d}")

    print("\n" + "-" * 70)
    print("🚨 TOP 5 SEVEREST OUTBREAK EPICENTERS (BY CONFIRMED BURDEN):")
    print("-" * 70)
    top_epicenters = get_top_outbreaks(clustered_df, metric="Confirmed", n=5)
    for _, row in top_epicenters.iterrows():
        print(
            f"   • {row['Region']:<18} | "
            f"Confirmed: {row['Confirmed']:>9,d} | "
            f"Deaths: {row['Deaths']:>7,d} | "
            f"CFR: {row['CFR_Pct']:>5.2f}% | "
            f"Active: {row['Active_Cases']:>9,d} | "
            f"[{row['Risk_Profile_Cluster']}]"
        )

    print("\n" + "-" * 70)
    print("📈 RISK CLUSTER BREAKDOWN (DISTRIBUTION OF NATIONS):")
    print("-" * 70)
    for tier, group in clustered_df.groupby("Risk_Profile_Cluster"):
        total_inf = group["Confirmed"].sum()
        avg_cfr = group["CFR_Pct"].mean()
        print(
            f"   • {tier:<38} : {len(group):>3d} countries ({total_inf:>10,d} total cases | Avg CFR: {avg_cfr:>5.2f}%)"
        )

    print("\n✓ Surveillance pipeline execution completed successfully.\n")
    return kpis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COVID-19 Global Epidemiology Intelligence")
    parser.add_argument("--raw-data", default="data/raw/covid19_global_raw.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-out", default="models/covid_risk_clustering_pipeline.joblib")
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()

    run_surveillance_pipeline(
        raw_data_path=args.raw_data,
        processed_dir=args.processed_dir,
        model_output_path=args.model_out,
        n_clusters=args.clusters,
    )
