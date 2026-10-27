"""Generate markdown summary tables and surveillance findings for documentation."""

from src.analytics import aggregate_by_region, get_global_kpis, get_top_outbreaks
from src.clustering import fit_and_assign_clusters
from src.data_loader import clean_covid_data, load_raw_covid_data


def generate_summary():
    raw_df = load_raw_covid_data("data/raw/covid19_global_raw.csv")
    cleaned_df = clean_covid_data(raw_df)
    regional_df = aggregate_by_region(cleaned_df)
    clustered_df, _ = fit_and_assign_clusters(regional_df, n_clusters=4)
    kpis = get_global_kpis(cleaned_df)

    print("### Global Surveillance KPIs")
    print(f"- Total Confirmed: {kpis['total_confirmed']:,}")
    print(f"- Total Deaths: {kpis['total_deaths']:,}")
    print(f"- Total Recovered: {kpis['total_recovered']:,}")
    print(f"- Total Active: {kpis['total_active']:,}")
    print(f"- Global CFR: {kpis['global_cfr_pct']}%")
    print(f"- Global Recovery Rate: {kpis['global_recovery_rate_pct']}%")
    print(f"- Total Sovereign Countries: {kpis['total_countries_affected']}")

    print("\n### Top 10 Severest Outbreak Epicenters")
    print(
        "| Rank | Country / Region | Confirmed Cases | Deaths | Recovered | Active Infections | CFR (%) | Recovery Rate (%) | Risk Cluster |"
    )
    print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    top10 = get_top_outbreaks(clustered_df, metric="Confirmed", n=10)
    for idx, row in top10.iterrows():
        print(
            f"| {idx+1} | **{row['Region']}** | {row['Confirmed']:,} | {row['Deaths']:,} | {row['Recovered']:,} | {row['Active_Cases']:,} | {row['CFR_Pct']:.2f}% | {row['Recovery_Rate_Pct']:.2f}% | `{row['Risk_Profile_Cluster']}` |"
        )


if __name__ == "__main__":
    generate_summary()
