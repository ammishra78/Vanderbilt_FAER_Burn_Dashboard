#!/usr/bin/env python3
"""
Interactive dashboard for phagocytosis fluorescence curves + ROS curves.

Launch:
    py -m streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy import stats

from build_final_summary import normalize_infection_status, tbsa_bin
from extract_phagocytosis import treatment_sort_key
from merge_phagocytosis_clinical import load_clinical_patient_table

BASE = Path(__file__).resolve().parent
EXTRACTED = BASE / "extracted_data"

VISIT_ORDER = {"Day 1": 1, "Day 3": 3, "Day 7": 7, "Day 8": 8, "Discharge": 9, "Unspecified": 99}
TREATMENT_COLORS = {
    "Vehicle": "#1f77b4",
    "MPLA": "#d62728",
    "GCSF": "#2ca02c",
    "MPLA+GCSF": "#9467bd",
}


def sort_visit_days(values: list[str]) -> list[str]:
    return sorted(values, key=lambda v: (VISIT_ORDER.get(str(v), 50), str(v)))


def age_group(value: object) -> str:
    age = pd.to_numeric(value, errors="coerce")
    if pd.isna(age):
        return "Unknown"
    if age < 30:
        return "<30"
    if age <= 50:
        return "30-50"
    return ">50"


@st.cache_data(show_spinner="Loading datasets...")
def load_data(base_dir: str) -> dict:
    base = Path(base_dir)
    phago_ts = pd.read_csv(base / "extracted_data/phagocytosis_timeseries.csv")
    phago_backfilled = base / "extracted_data/phagocytosis_endpoint_final_with_backfill.csv"
    if phago_backfilled.exists():
        phago_endpoint = pd.read_csv(phago_backfilled).rename(
            columns={"phagocytosis_adjusted_mfi_final": "adjusted_mfi_final"}
        )
    else:
        phago_endpoint = pd.read_csv(base / "extracted_data/phagocytosis_endpoint_final.csv")
        phago_endpoint["phagocytosis_source"] = "raw_kinetics"
    ros_endpoint = pd.read_csv(base / "extracted_data/ros_endpoint.csv")
    clinical = load_clinical_patient_table(base / "Burn Study - Infection Data_Pt_information.xlsx")

    clinical["age"] = pd.to_numeric(clinical.get("age"), errors="coerce")
    clinical["pct_tbsa"] = pd.to_numeric(clinical.get("pct_tbsa"), errors="coerce")
    clinical["infection_group"] = clinical.get("infection_y_n").map(normalize_infection_status)
    clinical["infection_group"] = clinical["infection_group"].replace({"Unknown": "Unsure/Unknown", "Unsure": "Unsure/Unknown"})
    clinical["tbsa_group"] = clinical["pct_tbsa"].map(tbsa_bin)
    clinical["age_group"] = clinical["age"].map(age_group)

    merge_cols = ["patient_id", "age", "age_group", "pct_tbsa", "tbsa_group", "infection_y_n", "infection_group", "sex"]
    phago_endpoint = phago_endpoint.merge(clinical[merge_cols], on="patient_id", how="left")
    ros_endpoint = ros_endpoint.merge(clinical[merge_cols], on="patient_id", how="left")
    return {
        "phago_ts": phago_ts,
        "phago_endpoint": phago_endpoint,
        "ros_endpoint": ros_endpoint,
        "clinical": clinical,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if "visit_day" in out.columns and filters["visit_days"]:
        out = out[out["visit_day"].isin(filters["visit_days"])]
    if "treatment" in out.columns and filters["treatments"]:
        out = out[out["treatment"].isin(filters["treatments"])]
    if "infection_group" in out.columns and filters["infection_groups"]:
        out = out[out["infection_group"].isin(filters["infection_groups"])]
    if "tbsa_group" in out.columns and filters["tbsa_groups"]:
        out = out[out["tbsa_group"].isin(filters["tbsa_groups"])]
    if "age_group" in out.columns and filters["age_groups"]:
        out = out[out["age_group"].isin(filters["age_groups"])]
    return out


def draw_phago_curve(ts: pd.DataFrame, patient_id: int, visit_day: str) -> plt.Figure | None:
    subset = ts[(ts["patient_id"] == patient_id) & (ts["visit_day"].astype(str) == str(visit_day))]
    if subset.empty:
        return None
    treatments = sorted(subset["treatment"].unique(), key=treatment_sort_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for treatment in treatments:
        part = subset[subset["treatment"] == treatment].sort_values("time_minutes")
        ax.plot(
            part["time_minutes"],
            part["adjusted_mfi"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=treatment,
            color=TREATMENT_COLORS.get(treatment),
        )
    ax.set_title(f"Patient {patient_id} - Phagocytosis fluorescence ({visit_day})")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Adjusted MFI (blank subtracted)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Treatment")
    fig.tight_layout()
    return fig


def draw_patient_ros_curve(ros: pd.DataFrame, patient_id: int) -> plt.Figure | None:
    subset = ros[ros["patient_id"] == patient_id]
    if subset.empty:
        return None
    visits = sort_visit_days([str(v) for v in subset["visit_day"].dropna().unique()])
    idx = {visit: i for i, visit in enumerate(visits)}
    treatments = sorted(subset["treatment"].unique(), key=treatment_sort_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for treatment in treatments:
        part = subset[subset["treatment"] == treatment].copy()
        part["idx"] = part["visit_day"].astype(str).map(idx)
        part = part.sort_values("idx")
        ax.plot(
            part["idx"],
            part["adjusted_ros_mfi"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=treatment,
            color=TREATMENT_COLORS.get(treatment),
        )
    ax.set_xticks(range(len(visits)))
    ax.set_xticklabels(visits)
    ax.set_title(f"Patient {patient_id} - ROS endpoint curve")
    ax.set_xlabel("Visit day")
    ax.set_ylabel("Adjusted ROS MFI (Sample ROS - NS)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Treatment")
    fig.tight_layout()
    return fig


def draw_grouped_bar(df: pd.DataFrame, value_col: str, strat_col: str, title: str, ylabel: str) -> plt.Figure | None:
    if df.empty:
        return None
    work = df.copy()
    work["treatment"] = work["treatment"].astype(str).str.strip()
    work[strat_col] = work[strat_col].astype(str).str.strip()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    treatments = sorted(work["treatment"].dropna().unique(), key=treatment_sort_key)
    strata = sorted(work[strat_col].dropna().unique(), key=str)
    if not treatments or not strata:
        return None

    width = 0.8 / max(len(strata), 1)
    x_base = list(range(len(treatments)))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, stratum in enumerate(strata):
        means, sems = [], []
        for treatment in treatments:
            vals = work[(work["treatment"] == treatment) & (work[strat_col] == stratum)][value_col]
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0)
        offset = (i - (len(strata) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_base],
            means,
            width=width * 0.95,
            yerr=sems,
            capsize=3,
            label=f"{stratum}",
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(treatments)
    ax.set_title(title)
    ax.set_xlabel("Treatment")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title=strat_col.replace("_", " ").title())
    fig.tight_layout()
    return fig


def draw_stratum_xaxis_bar(
    df: pd.DataFrame,
    value_col: str,
    stratum_col: str,
    title: str,
    ylabel: str,
) -> plt.Figure | None:
    """Plot chosen stratum on X-axis with treatment bars."""
    if df.empty or stratum_col not in df.columns:
        return None
    work = df.copy()
    work["treatment"] = work["treatment"].astype(str).str.strip()
    work[stratum_col] = work[stratum_col].astype(str).str.strip()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    order_map = {
        "tbsa_group": ["<10%", "10-20%", ">20%", "Unknown"],
        "infection_group": ["Not infected", "Infected", "Unsure/Unknown"],
        "age_group": ["<30", "30-50", ">50", "Unknown"],
    }
    x_order = order_map.get(stratum_col)
    unique_vals = set(work[stratum_col].dropna().astype(str))
    if x_order is None:
        x_groups = sorted(unique_vals, key=str)
    else:
        x_groups = [x for x in x_order if x in unique_vals]
    treatments = sorted(work["treatment"].dropna().unique(), key=treatment_sort_key)
    if not x_groups or not treatments:
        return None

    width = 0.8 / max(len(treatments), 1)
    x_base = list(range(len(x_groups)))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, treatment in enumerate(treatments):
        means, sems = [], []
        for grp in x_groups:
            vals = work[(work[stratum_col] == grp) & (work["treatment"] == treatment)][value_col]
            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0)
        offset = (i - (len(treatments) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_base],
            means,
            width=width * 0.95,
            yerr=sems,
            capsize=3,
            label=treatment,
            color=TREATMENT_COLORS.get(treatment),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(x_groups)
    ax.set_title(title)
    ax.set_xlabel(stratum_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Treatment")
    fig.tight_layout()
    return fig


def paired_stats_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    comparisons = [("Vehicle", "MPLA"), ("Vehicle", "GCSF"), ("MPLA", "GCSF")]
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(
        index=["patient_id", "visit_day"],
        columns="treatment",
        values=value_col,
        aggfunc="first",
    )
    rows: list[dict] = []
    for a, b in comparisons:
        if a not in wide.columns or b not in wide.columns:
            continue
        paired = wide[[a, b]].dropna()
        if len(paired) < 2:
            continue
        diff = paired[b] - paired[a]
        t_stat, t_p = stats.ttest_rel(paired[b], paired[a])
        try:
            w_stat, w_p = stats.wilcoxon(paired[b], paired[a])
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        rows.append(
            {
                "comparison": f"{b} vs {a}",
                "n_pairs": int(len(paired)),
                "mean_diff": round(float(diff.mean()), 3),
                "paired_t_pvalue": float(t_p),
                "wilcoxon_pvalue": float(w_p) if w_p == w_p else None,
            }
        )
    return pd.DataFrame(rows)


def patient_value_map(df: pd.DataFrame, value_col: str) -> dict[int, str]:
    """Create JSON strings: {visit: {treatment: value}} per patient."""
    out: dict[int, str] = {}
    if df.empty:
        return out
    for patient_id, g in df.groupby("patient_id"):
        visit_map: dict[str, dict[str, float]] = {}
        for visit, vg in g.groupby("visit_day"):
            key = str(visit)
            visit_map[key] = {}
            for _, row in vg.iterrows():
                value = row.get(value_col)
                if pd.notna(value):
                    visit_map[key][str(row.get("treatment"))] = round(float(value), 3)
        out[int(patient_id)] = json.dumps(visit_map, sort_keys=True)
    return out


def build_patient_explorer_table(
    clinical: pd.DataFrame,
    phago_endpoint: pd.DataFrame,
    ros_endpoint: pd.DataFrame,
) -> pd.DataFrame:
    ph_map = patient_value_map(phago_endpoint, "adjusted_mfi_final")
    ros_map = patient_value_map(ros_endpoint, "adjusted_ros_mfi")
    days_collected_map: dict[int, str] = {}
    all_ids = set()
    all_ids.update(set(phago_endpoint["patient_id"].dropna().astype(int).tolist()))
    all_ids.update(set(ros_endpoint["patient_id"].dropna().astype(int).tolist()))
    for patient_id in all_ids:
        ph_days = set(
            phago_endpoint[phago_endpoint["patient_id"] == patient_id]["visit_day"]
            .dropna()
            .astype(str)
            .tolist()
        )
        ros_days = set(
            ros_endpoint[ros_endpoint["patient_id"] == patient_id]["visit_day"]
            .dropna()
            .astype(str)
            .tolist()
        )
        merged_days = sort_visit_days(sorted(ph_days.union(ros_days)))
        days_collected_map[patient_id] = ", ".join(merged_days)
    base = clinical.copy()
    base["patient_id"] = pd.to_numeric(base["patient_id"], errors="coerce")
    base = base[base["patient_id"].notna()].copy()
    base["patient_id"] = base["patient_id"].astype(int)
    base["final_phagocytosis_mfi"] = base["patient_id"].map(ph_map)
    base["final_ros_mfi"] = base["patient_id"].map(ros_map)
    base["days_collected"] = base["patient_id"].map(days_collected_map).fillna("")
    return base[
        [
            "patient_id",
            "days_collected",
            "age",
            "sex",
            "infection_y_n",
            "infection_group",
            "pct_tbsa",
            "tbsa_group",
            "final_phagocytosis_mfi",
            "final_ros_mfi",
        ]
    ].sort_values("patient_id")


def main() -> None:
    st.set_page_config(page_title="Burn Fluorescence Dashboard", layout="wide")
    st.title("Burn Patient Fluorescence Dashboard")

    data = load_data(str(BASE))
    phago_ts = data["phago_ts"]
    phago_endpoint = data["phago_endpoint"]
    ros_endpoint = data["ros_endpoint"]
    clinical = data["clinical"]

    all_visits = sort_visit_days([str(v) for v in phago_endpoint["visit_day"].dropna().unique()])
    all_treatments = sorted(phago_endpoint["treatment"].dropna().unique(), key=treatment_sort_key)
    all_infection = sorted(phago_endpoint["infection_group"].dropna().unique(), key=str)
    all_tbsa = sorted(phago_endpoint["tbsa_group"].dropna().unique(), key=str)
    all_age_groups = sorted(phago_endpoint["age_group"].dropna().unique(), key=str)

    with st.sidebar:
        st.header("Cohort Filters")
        visit_days = st.multiselect("Visit days", all_visits, default=all_visits)
        treatments = st.multiselect("Treatments", all_treatments, default=all_treatments)
        infection_groups = st.multiselect("Infection", all_infection, default=all_infection)
        tbsa_groups = st.multiselect("TBSA bins", all_tbsa, default=all_tbsa)
        age_groups = st.multiselect("Age groups", all_age_groups, default=all_age_groups)
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    filters = {
        "visit_days": visit_days,
        "treatments": treatments,
        "infection_groups": infection_groups,
        "tbsa_groups": tbsa_groups,
        "age_groups": age_groups,
    }
    phago_f = apply_filters(phago_endpoint, filters)
    ros_f = apply_filters(ros_endpoint, filters)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Patient Explorer", "Stratified Phagocytosis", "Stratified ROS", "Tables"]
    )

    with tab1:
        st.subheader("Patient-level curves")
        patients = sorted(
            set(phago_ts["patient_id"].dropna().astype(int).unique()).union(
                set(ros_endpoint["patient_id"].dropna().astype(int).unique())
            )
        )
        selected_patient = st.selectbox("Patient", patients, format_func=lambda x: f"Patient {x}")
        patient_visits_kinetics = sort_visit_days(
            [str(v) for v in phago_ts[phago_ts["patient_id"] == selected_patient]["visit_day"].dropna().unique()]
        )
        patient_visits_endpoint = sort_visit_days(
            [str(v) for v in phago_endpoint[phago_endpoint["patient_id"] == selected_patient]["visit_day"].dropna().unique()]
        )
        patient_visits_ros = sort_visit_days(
            [str(v) for v in ros_endpoint[ros_endpoint["patient_id"] == selected_patient]["visit_day"].dropna().unique()]
        )
        patient_visits = sort_visit_days(
            sorted(set(patient_visits_kinetics).union(set(patient_visits_endpoint)).union(set(patient_visits_ros)))
        )
        visit_badges: dict[str, str] = {}
        for visit in patient_visits:
            has_phago = visit in patient_visits_kinetics or visit in patient_visits_endpoint
            has_ros = visit in patient_visits_ros
            if has_ros and not has_phago:
                visit_badges[visit] = "ROS only"
            elif has_ros and has_phago:
                visit_badges[visit] = "Phago + ROS"
            else:
                visit_badges[visit] = "Fluorescence only"
        selected_visit = (
            st.selectbox(
                "Phagocytosis visit",
                patient_visits,
                format_func=lambda v: f"{v} ({visit_badges.get(v, 'Phago + ROS')})",
            )
            if patient_visits
            else None
        )
        c1, c2 = st.columns(2)
        with c1:
            if selected_visit is not None:
                fig = draw_phago_curve(phago_ts, selected_patient, selected_visit)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("No fluorescence kinetics curve for this patient/visit.")
                    endpoint_row = phago_endpoint[
                        (phago_endpoint["patient_id"] == selected_patient)
                        & (phago_endpoint["visit_day"].astype(str) == str(selected_visit))
                    ]
                    if not endpoint_row.empty:
                        st.dataframe(
                            endpoint_row[
                                ["patient_id", "visit_day", "treatment", "adjusted_mfi_final", "phagocytosis_source"]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                        if (
                            endpoint_row["phagocytosis_source"]
                            .astype(str)
                            .str.contains("backfilled_combined_am", case=False)
                            .any()
                        ):
                            st.warning("This visit is backfilled from Combined_AM; fluorescence kinetics values are unavailable.")
        with c2:
            fig = draw_patient_ros_curve(ros_endpoint, selected_patient)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No ROS curve available for this patient.")

        st.subheader("Patient explorer table")
        st.caption(
            "Final values are shown as JSON maps by visit and treatment: "
            "{visit_day: {treatment: adjusted_MFI}}."
        )
        explorer_table = build_patient_explorer_table(clinical, phago_endpoint, ros_endpoint)
        st.dataframe(explorer_table, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Phagocytosis endpoint stratification")
        strat_col = st.selectbox("Stratify phagocytosis by", ["infection_group", "tbsa_group", "age_group"])
        c_left, c_right = st.columns(2)
        with c_left:
            fig = draw_grouped_bar(
                phago_f,
                "adjusted_mfi_final",
                strat_col,
                f"Phagocytosis by treatment stratified by {strat_col}",
                "Adjusted MFI (final timepoint)",
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        with c_right:
            fig_tbsa = draw_stratum_xaxis_bar(
                phago_f,
                "adjusted_mfi_final",
                strat_col,
                f"Phagocytosis with {strat_col} on X-axis",
                "Adjusted MFI (final timepoint)",
            )
            if fig_tbsa:
                st.pyplot(fig_tbsa)
                plt.close(fig_tbsa)
            else:
                st.info("Alternate X-axis view unavailable for current filters.")
        st.dataframe(
            phago_f[
                [
                    "patient_id",
                    "visit_day",
                    "treatment",
                    "adjusted_mfi_final",
                    "phagocytosis_source",
                    "age",
                    "pct_tbsa",
                    "tbsa_group",
                    "infection_group",
                ]
            ].sort_values(["patient_id", "visit_day", "treatment"], key=lambda s: s.map(str)),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Paired p-values (current filters)**")
        ph_stats = paired_stats_table(phago_f, "adjusted_mfi_final")
        if ph_stats.empty:
            st.info("Not enough paired observations for p-values with current filters.")
        else:
            st.dataframe(ph_stats, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download phagocytosis p-values (CSV)",
                data=ph_stats.to_csv(index=False),
                file_name="phagocytosis_pvalues_filtered.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("ROS endpoint stratification (Sample ROS - NS)")
        strat_col = st.selectbox("Stratify ROS by", ["infection_group", "tbsa_group", "age_group"], key="ros_strat")
        c_left, c_right = st.columns(2)
        with c_left:
            fig = draw_grouped_bar(
                ros_f,
                "adjusted_ros_mfi",
                strat_col,
                f"ROS by treatment stratified by {strat_col}",
                "Adjusted ROS MFI (Sample ROS - NS)",
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        with c_right:
            fig_tbsa = draw_stratum_xaxis_bar(
                ros_f,
                "adjusted_ros_mfi",
                strat_col,
                f"ROS with {strat_col} on X-axis",
                "Adjusted ROS MFI (Sample ROS - NS)",
            )
            if fig_tbsa:
                st.pyplot(fig_tbsa)
                plt.close(fig_tbsa)
            else:
                st.info("Alternate X-axis view unavailable for current filters.")
        st.dataframe(
            ros_f[
                [
                    "patient_id",
                    "visit_day",
                    "treatment",
                    "adjusted_ros_mfi",
                    "sample_ros_mfi",
                    "ns_ros_mfi",
                    "age",
                    "pct_tbsa",
                    "tbsa_group",
                    "infection_group",
                ]
            ].sort_values(["patient_id", "visit_day", "treatment"], key=lambda s: s.map(str)),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Paired p-values (current filters)**")
        ros_stats = paired_stats_table(ros_f, "adjusted_ros_mfi")
        if ros_stats.empty:
            st.info("Not enough paired observations for p-values with current filters.")
        else:
            st.dataframe(ros_stats, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download ROS p-values (CSV)",
                data=ros_stats.to_csv(index=False),
                file_name="ros_pvalues_filtered.csv",
                mime="text/csv",
            )

    with tab4:
        st.subheader("Clinical table")
        st.dataframe(
            clinical[
                [
                    "patient_id",
                    "sex",
                    "age",
                    "pct_tbsa",
                    "tbsa_group",
                    "infection_y_n",
                    "infection_group",
                    "admission_date",
                    "discharge_date",
                ]
            ].sort_values("patient_id", key=lambda s: s.map(str)),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Regenerated final summary")
        path = EXTRACTED / "final_patient_summary.csv"
        if path.exists():
            st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
        else:
            st.info("Run `py build_final_summary.py` to generate final summary table.")


if __name__ == "__main__":
    main()

