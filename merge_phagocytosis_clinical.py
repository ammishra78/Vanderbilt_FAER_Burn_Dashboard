#!/usr/bin/env python3
"""
Merge phagocytosis time-series data with burn study patient information.

Joins extracted_data/phagocytosis_timeseries.csv to
Burn Study - Infection Data_Pt_information.xlsx (Enrollments sheet) on patient_id.

The clinical sheet may contain multiple culture rows per patient; this script
keeps one demographics row per patient (first enrollment row after forward-filling
patient IDs on continuation lines).

Install:
    pip install pandas openpyxl
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_TIMESERIES = Path("extracted_data/phagocytosis_timeseries.csv")
DEFAULT_CLINICAL = Path("Burn Study - Infection Data_Pt_information.xlsx")
DEFAULT_OUTPUT = Path("extracted_data/phagocytosis_timeseries_clinical.csv")


def to_snake_case(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("%", "pct_").replace("/", "_")
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_patient_id(value: object) -> int | str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value).is_integer():
            return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def load_clinical_patient_table(path: Path, sheet_name: str = "Enrollments") -> pd.DataFrame:
    clinical = pd.read_excel(path, sheet_name=sheet_name)
    clinical = clinical.rename(columns={"Patient ID": "patient_id"})
    clinical["patient_id"] = clinical["patient_id"].ffill().map(normalize_patient_id)

    # Drop rows that never received a patient id.
    clinical = clinical[clinical["patient_id"].notna()].copy()

    # One enrollment/demographics row per patient.
    clinical = (
        clinical.groupby("patient_id", as_index=False)
        .first()
        .sort_values("patient_id", key=lambda s: s.map(lambda x: (isinstance(x, str), x)))
        .reset_index(drop=True)
    )

    rename_map = {col: to_snake_case(col) for col in clinical.columns if col != "patient_id"}
    clinical = clinical.rename(columns=rename_map)

    drop_cols = [col for col in clinical.columns if col.startswith("unnamed")]
    clinical = clinical.drop(columns=drop_cols, errors="ignore")

    return clinical


def load_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    timeseries = pd.read_csv(path)
    timeseries["patient_id"] = timeseries["patient_id"].map(normalize_patient_id)
    return timeseries


def merge_phagocytosis_clinical(
    timeseries: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    how: str = "left",
) -> pd.DataFrame:
    merged = timeseries.merge(clinical, on="patient_id", how=how, suffixes=("", "_clinical_dup"))
    dup_cols = [col for col in merged.columns if col.endswith("_clinical_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    # Phagocytosis columns first, then clinical metadata.
    phago_cols = [col for col in timeseries.columns if col in merged.columns]
    clinical_cols = [col for col in clinical.columns if col != "patient_id" and col in merged.columns]
    return merged[phago_cols + clinical_cols]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge phagocytosis time series with burn study patient information."
    )
    parser.add_argument("--timeseries", type=Path, default=DEFAULT_TIMESERIES)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--clinical-sheet", default="Enrollments")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--how",
        choices=("left", "inner", "right", "outer"),
        default="left",
        help="Merge type (default: left — keep all phagocytosis rows).",
    )
    parser.add_argument(
        "--clinical-only",
        type=Path,
        help="Optional path to also save the one-row-per-patient clinical table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        timeseries = load_timeseries(args.timeseries)
        clinical = load_clinical_patient_table(args.clinical, sheet_name=args.clinical_sheet)
        merged = merge_phagocytosis_clinical(timeseries, clinical, how=args.how)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    if args.clinical_only:
        args.clinical_only.parent.mkdir(parents=True, exist_ok=True)
        clinical.to_csv(args.clinical_only, index=False)

    ts_patients = set(timeseries["patient_id"].dropna())
    clinical_patients = set(clinical["patient_id"].dropna())
    matched = ts_patients & clinical_patients
    missing_clinical = sorted(ts_patients - clinical_patients, key=str)
    no_phagocytosis = sorted(clinical_patients - ts_patients, key=str)

    print(f"Wrote {len(merged)} rows to {args.output}")
    print(f"Phagocytosis patients: {len(ts_patients)}")
    print(f"Clinical patients: {len(clinical_patients)}")
    print(f"Matched on patient_id: {len(matched)}")
    if missing_clinical:
        print(f"No clinical row for phagocytosis patient(s): {missing_clinical}")
    if no_phagocytosis:
        print(f"Clinical patient(s) without phagocytosis time series: {no_phagocytosis}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
