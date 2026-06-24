#!/usr/bin/env python3
"""Build final merged summary tables for phagocytosis + ROS + clinical metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from merge_phagocytosis_clinical import load_clinical_patient_table


BASE = Path(__file__).resolve().parent
EXTRACTED = BASE / "extracted_data"
COMBINED_AM = BASE / "Burn Patient Data Combined_AM.xlsx"


def normalize_infection_status(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().upper()
    if text in {"Y", "YES"}:
        return "Infected"
    if text in {"N", "NO"} or text.startswith("N"):
        return "Not infected"
    return "Unsure"


def tbsa_bin(value: object) -> str:
    pct = pd.to_numeric(value, errors="coerce")
    if pd.isna(pct):
        return "Unknown"
    if pct < 10:
        return "<10%"
    if pct <= 20:
        return "10-20%"
    return ">20%"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phago = pd.read_csv(EXTRACTED / "phagocytosis_endpoint_final.csv")
    ros = pd.read_csv(EXTRACTED / "ros_endpoint.csv")
    clinical = load_clinical_patient_table(BASE / "Burn Study - Infection Data_Pt_information.xlsx")
    clinical["pct_tbsa"] = pd.to_numeric(clinical.get("pct_tbsa"), errors="coerce")
    clinical["infection_group"] = clinical.get("infection_y_n").map(normalize_infection_status)
    clinical["tbsa_group"] = clinical["pct_tbsa"].map(tbsa_bin)
    return phago, ros, clinical


def load_combined_am_phagocytosis(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["patient_id", "visit_day", "treatment", "phagocytosis_adjusted_mfi_final"])
    df = pd.read_excel(path, sheet_name="Sheet1", header=1)
    df["Patient ID"] = df["Patient ID"].ffill()
    df["Day"] = df.groupby("Patient ID")["Day"].ffill()
    df = df.dropna(subset=["Treatment"]).copy()
    treat_map = {"Veh": "Vehicle", "MPLA": "MPLA", "GCSF": "GCSF"}
    df["treatment"] = df["Treatment"].astype(str).str.strip().map(lambda x: treat_map.get(x, x))
    df["patient_id"] = pd.to_numeric(df["Patient ID"], errors="coerce")
    df["visit_day"] = df["Day"].astype(str).str.strip()
    df["phagocytosis_adjusted_mfi_final"] = pd.to_numeric(df["Phagocytosis (MFI)"], errors="coerce")
    out = df[["patient_id", "visit_day", "treatment", "phagocytosis_adjusted_mfi_final"]].dropna(
        subset=["patient_id", "phagocytosis_adjusted_mfi_final"]
    )
    out["patient_id"] = out["patient_id"].astype(int)
    return out


def backfill_phagocytosis_from_combined_am(ph: pd.DataFrame) -> pd.DataFrame:
    combined = load_combined_am_phagocytosis(COMBINED_AM)
    ph = ph.copy()
    ph["phagocytosis_source"] = "raw_kinetics"
    if combined.empty:
        return ph

    keys = ["patient_id", "visit_day", "treatment"]
    existing = set(tuple(x) for x in ph[keys].astype(str).itertuples(index=False, name=None))
    rows_to_add: list[dict] = []
    for _, row in combined.iterrows():
        key = (str(int(row["patient_id"])), str(row["visit_day"]), str(row["treatment"]))
        if key in existing:
            continue
        rows_to_add.append(
            {
                "patient_id": int(row["patient_id"]),
                "visit_day": str(row["visit_day"]),
                "treatment": str(row["treatment"]),
                "phagocytosis_adjusted_mfi_final": float(row["phagocytosis_adjusted_mfi_final"]),
                "phagocytosis_source": "backfilled_combined_am_raw_kinetics_unavailable",
            }
        )
    if rows_to_add:
        ph = pd.concat([ph, pd.DataFrame(rows_to_add)], ignore_index=True)
    return ph


def build_visit_treatment_summary(
    phago: pd.DataFrame, ros: pd.DataFrame, clinical: pd.DataFrame
) -> pd.DataFrame:
    ph = phago.rename(
        columns={
            "adjusted_mfi_final": "phagocytosis_adjusted_mfi_final",
            "raw_mfi_final": "phagocytosis_raw_mfi_final",
            "blank_mfi_final": "phagocytosis_blank_mfi_final",
            "final_time_minutes": "phagocytosis_final_time_minutes",
            "n_replicates": "phagocytosis_n_replicates",
        }
    )
    rs = ros.rename(
        columns={
            "adjusted_ros_mfi": "ros_adjusted_mfi",
            "sample_ros_mfi": "ros_sample_mfi",
            "ns_ros_mfi": "ros_ns_mfi",
        }
    )
    ph = backfill_phagocytosis_from_combined_am(ph)
    merge_keys = ["patient_id", "visit_day", "treatment"]
    merged = ph.merge(rs[merge_keys + ["ros_adjusted_mfi", "ros_sample_mfi", "ros_ns_mfi"]], on=merge_keys, how="outer")
    merged = merged.merge(clinical, on="patient_id", how="left")
    merged["data_status"] = "provisional_missing_patients_17_18_confirmed_missing_15_16_reassigned"
    return merged.sort_values(["patient_id", "visit_day", "treatment"], key=lambda s: s.map(str))


def build_patient_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for patient_id, g in merged.groupby("patient_id"):
        visits = sorted(g["visit_day"].dropna().unique(), key=str)
        ph_map: dict[str, dict[str, float]] = {}
        ros_map: dict[str, dict[str, float]] = {}
        for visit, vg in g.groupby("visit_day"):
            visit_key = str(visit)
            ph_map[visit_key] = {}
            ros_map[visit_key] = {}
            for _, row in vg.iterrows():
                treatment = str(row.get("treatment"))
                ph = row.get("phagocytosis_adjusted_mfi_final")
                rs = row.get("ros_adjusted_mfi")
                if pd.notna(ph):
                    ph_map[visit_key][treatment] = round(float(ph), 3)
                if pd.notna(rs):
                    ros_map[visit_key][treatment] = round(float(rs), 3)

        first = g.iloc[0].to_dict()
        rows.append(
            {
                "patient_id": patient_id,
                "sex": first.get("sex"),
                "age": first.get("age"),
                "pct_tbsa": first.get("pct_tbsa"),
                "tbsa_group": first.get("tbsa_group"),
                "infection_y_n": first.get("infection_y_n"),
                "infection_group": first.get("infection_group"),
                "collected_days": ", ".join(visits),
                "n_collected_days": len(visits),
                "phagocytosis_adjusted_mfi_final": json.dumps(ph_map, sort_keys=True),
                "ros_adjusted_mfi": json.dumps(ros_map, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id")


def main() -> int:
    phago, ros, clinical = load_inputs()
    visit_treatment = build_visit_treatment_summary(phago, ros, clinical)
    patient_summary = build_patient_summary(visit_treatment)

    vt_path = EXTRACTED / "final_patient_visit_treatment_summary.csv"
    ps_path = EXTRACTED / "final_patient_summary.csv"
    ph_backfilled_path = EXTRACTED / "phagocytosis_endpoint_final_with_backfill.csv"
    visit_treatment.to_csv(vt_path, index=False)
    patient_summary.to_csv(ps_path, index=False)
    ph_cols = [
        "patient_id",
        "visit_day",
        "treatment",
        "phagocytosis_adjusted_mfi_final",
        "phagocytosis_source",
    ]
    visit_treatment[ph_cols].dropna(subset=["phagocytosis_adjusted_mfi_final"]).drop_duplicates(
        ["patient_id", "visit_day", "treatment"], keep="first"
    ).to_csv(ph_backfilled_path, index=False)

    print(f"Wrote {len(visit_treatment)} rows to {vt_path}")
    print(f"Wrote {len(patient_summary)} rows to {ps_path}")
    print(f"Wrote backfilled phagocytosis endpoint rows to {ph_backfilled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

