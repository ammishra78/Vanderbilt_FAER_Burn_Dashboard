#!/usr/bin/env python3
"""
Extract phagocytosis kinetic data from Synergy H1 plate-reader exports.

Each non-blank treatment arm is reported separately (Vehicle, MPLA, GCSF,
MPLA+GCSF, etc.). Visit day is parsed from the sheet name, e.g.
"Phagocytosis Day 1" -> "Day 1", "Phagocytosis Discharge" -> "Discharge".

Processing steps (per timepoint):
  1. Average replicate wells for each treatment and for Blank
  2. Subtract blank fluorescence from each treatment
  3. Floor negative values at zero

Source Excel/CSV files are read only. Results are written to a new CSV.

Install dependencies:
    pip install -r requirements.txt

Examples:
    py extract_phagocytosis.py --input "Burn Patient #13 Phagocytosis and ROS Production.xlsx" --patient-id 13
    py extract_phagocytosis.py --batch-dir . --output extracted_data/phagocytosis_timeseries.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import time
from pathlib import Path

import pandas as pd

WELL_PATTERN = re.compile(r"^[A-H](?:10|11|12|[1-9])$", re.IGNORECASE)

# Display order for plots; any other detected treatment is appended alphabetically.
TREATMENT_ORDER = ("Vehicle", "MPLA", "GCSF", "MPLA+GCSF")


def normalize_treatment(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    text = label.strip().lower()
    if not text:
        return None
    if "blank" in text:
        return "Blank"
    if "mpla+gcsf" in text or "mpla + gcsf" in text:
        return "MPLA+GCSF"
    if "mpla" in text:
        return "MPLA"
    if "gcsf" in text:
        return "GCSF"
    if text in {"veh", "vehicle"} or text.startswith("veh"):
        return "Vehicle"
    return None


def is_treatment_label(label: object) -> bool:
    group = normalize_treatment(label)
    return group is not None and group != "Blank"


def treatment_sort_key(treatment: str) -> tuple[int, str]:
    try:
        return (TREATMENT_ORDER.index(treatment), treatment)
    except ValueError:
        return (len(TREATMENT_ORDER), treatment)


def format_time(value: object) -> str | None:
    if isinstance(value, time):
        return f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
    if isinstance(value, str):
        text = value.strip()
        if re.match(r"\d+:\d{2}:\d{2}", text):
            return text
    return None


def time_to_minutes(time_str: str) -> float:
    hours, minutes, seconds = (int(part) for part in time_str.split(":"))
    return hours * 60 + minutes + seconds / 60


def parse_visit_day(sheet_name: str) -> str | None:
    match = re.search(
        r"phagocytosis\s+(.+)$",
        sheet_name.strip(),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def parse_patient_id_from_text(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"patient\s*#?\s*(\d+)", value, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def is_phagocytosis_sheet(sheet_name: str) -> bool:
    return "phagocytosis" in sheet_name.lower()


def list_phagocytosis_sheets(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return ["__csv__"]
    workbook = pd.ExcelFile(path)
    return [name for name in workbook.sheet_names if is_phagocytosis_sheet(name)]


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    if sheet_name == "__csv__":
        return pd.read_csv(path, header=None)
    return pd.read_excel(path, sheet_name=sheet_name, header=None)


def find_organized_block(df: pd.DataFrame) -> dict | None:
    start_row = None
    for i in range(df.shape[0]):
        for j in range(min(3, df.shape[1])):
            value = df.iat[i, j]
            if isinstance(value, str) and "treatment groups organized" in value.lower():
                start_row = i
                break
        if start_row is not None:
            break
    if start_row is None:
        return None

    label_row = start_row + 1
    well_row = start_row + 2

    time_col = None
    for col in range(df.shape[1]):
        if str(df.iat[well_row, col]).strip().lower() == "time":
            time_col = col
            break
    if time_col is None:
        return None

    group_columns: dict[str, list[int]] = {}
    for col in range(time_col + 1, df.shape[1]):
        group = normalize_treatment(df.iat[label_row, col])
        well = df.iat[well_row, col]
        if group and isinstance(well, str) and WELL_PATTERN.match(well.strip()):
            group_columns.setdefault(group, []).append(col)

    if not group_columns or "Blank" not in group_columns:
        return None

    data_start = start_row + 3
    data_end = df.shape[0]
    for i in range(data_start, df.shape[0]):
        for j in range(min(3, df.shape[1])):
            value = df.iat[i, j]
            if isinstance(value, str) and value.lower().startswith("averages"):
                data_end = i
                break
        if data_end != df.shape[0]:
            break

    column_patient_ids = infer_patient_ids_by_column(
        df,
        start_col=time_col + 1,
        end_col=df.shape[1],
        hint_row=label_row - 1,
    )

    return {
        "time_col": time_col,
        "data_start": data_start,
        "data_end": data_end,
        "group_columns": group_columns,
        "column_patient_ids": column_patient_ids,
        "source": "organized_block",
    }


def find_kinetic_block(df: pd.DataFrame) -> dict | None:
    header_row = None
    time_col = None

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if str(df.iat[row, col]).strip().lower() != "time":
                continue
            wells: list[tuple[int, str]] = []
            for well_col in range(col + 1, min(col + 97, df.shape[1])):
                value = df.iat[row, well_col]
                if isinstance(value, str) and WELL_PATTERN.match(value.strip()):
                    wells.append((well_col, value.strip().upper()))
            if len(wells) >= 3:
                header_row = row
                time_col = col
                break
        if header_row is not None:
            break

    if header_row is None or time_col is None:
        return None

    label_row = None
    for candidate in range(header_row - 3, header_row):
        if candidate < 0:
            continue
        labels = [df.iat[candidate, col] for col, _ in _well_columns_from_header(df, header_row, time_col)]
        if any(is_treatment_label(label) for label in labels):
            label_row = candidate
            break

    if label_row is None:
        return None

    group_columns: dict[str, list[int]] = {}
    for col, _well in _well_columns_from_header(df, header_row, time_col):
        group = normalize_treatment(df.iat[label_row, col])
        if group:
            group_columns.setdefault(group, []).append(col)

    if not group_columns or "Blank" not in group_columns:
        return None

    data_end = df.shape[0]
    for row in range(header_row + 1, df.shape[0]):
        for col in range(min(3, df.shape[1])):
            value = df.iat[row, col]
            if isinstance(value, str) and "results" in value.lower():
                data_end = row
                break
        if data_end != df.shape[0]:
            break

    column_patient_ids = infer_patient_ids_by_column(
        df,
        start_col=time_col + 1,
        end_col=df.shape[1],
        hint_row=label_row - 1,
    )

    return {
        "time_col": time_col,
        "data_start": header_row + 1,
        "data_end": data_end,
        "group_columns": group_columns,
        "column_patient_ids": column_patient_ids,
        "source": "kinetic_block",
    }


def _well_columns_from_header(
    df: pd.DataFrame, header_row: int, time_col: int
) -> list[tuple[int, str]]:
    wells: list[tuple[int, str]] = []
    for col in range(time_col + 1, df.shape[1]):
        value = df.iat[header_row, col]
        if isinstance(value, str) and WELL_PATTERN.match(value.strip()):
            wells.append((col, value.strip().upper()))
        elif wells:
            break
    return wells


def infer_patient_ids_by_column(
    df: pd.DataFrame,
    *,
    start_col: int,
    end_col: int,
    hint_row: int,
) -> dict[int, int]:
    """Infer patient IDs from merged label rows above organized blocks."""
    if hint_row < 0 or hint_row >= df.shape[0]:
        return {}
    mapping: dict[int, int] = {}
    current_patient: int | None = None
    for col in range(start_col, end_col):
        parsed = parse_patient_id_from_text(df.iat[hint_row, col])
        if parsed is not None:
            current_patient = parsed
        if current_patient is not None:
            mapping[col] = current_patient
    return mapping


def average_replicates(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def extract_timepoint_rows(df: pd.DataFrame, block: dict) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for row in range(block["data_start"], block["data_end"]):
        time_str = format_time(df.iat[row, block["time_col"]])
        if time_str:
            rows.append((row, time_str))
    return rows


def extract_phagocytosis_from_sheet(
    df: pd.DataFrame,
    *,
    patient_id: str | int | None,
    source_file: str,
    sheet_name: str,
) -> pd.DataFrame:
    block = find_organized_block(df) or find_kinetic_block(df)
    if block is None:
        raise ValueError(
            f"Could not locate phagocytosis data in sheet '{sheet_name}'. "
            "Expected a kinetic table or a 'Treatment groups organized' block."
        )

    visit_day = parse_visit_day(sheet_name) if sheet_name != "__csv__" else None
    records: list[dict] = []

    column_patient_ids: dict[int, int] = block.get("column_patient_ids", {})

    for row_idx, time_str in extract_timepoint_rows(df, block):
        values_by_group_patient: dict[tuple[str, int | str | None], list[float]] = {}
        blank_values: list[float] = []

        for group, columns in block["group_columns"].items():
            for col in columns:
                value = df.iat[row_idx, col]
                if not (pd.notna(value) and isinstance(value, (int, float))):
                    continue
                value_f = float(value)
                if group == "Blank":
                    blank_values.append(value_f)
                    continue
                row_patient_id: int | str | None = column_patient_ids.get(col, patient_id)
                key = (group, row_patient_id)
                values_by_group_patient.setdefault(key, []).append(value_f)

        blank = average_replicates(blank_values)
        if blank is None:
            continue

        for (group, row_patient_id), values in sorted(
            values_by_group_patient.items(),
            key=lambda item: (treatment_sort_key(item[0][0]), str(item[0][1])),
        ):
            raw_value = average_replicates(values)
            if raw_value is None or row_patient_id is None:
                continue
            adjusted = max(0.0, raw_value - blank)
            records.append(
                {
                    "patient_id": row_patient_id,
                    "source_file": source_file,
                    "sheet_name": sheet_name if sheet_name != "__csv__" else None,
                    "visit_day": visit_day,
                    "time": time_str,
                    "time_minutes": round(time_to_minutes(time_str), 2),
                    "treatment": group,
                    "raw_mfi": round(raw_value, 4),
                    "blank_mfi": round(blank, 4),
                    "adjusted_mfi": round(adjusted, 4),
                    "n_replicates": len(values),
                    "parser": block["source"],
                }
            )

    if not records:
        raise ValueError(f"No kinetic timepoints extracted from sheet '{sheet_name}'.")

    return pd.DataFrame(records)


def extract_phagocytosis_file(
    path: Path,
    *,
    patient_id: str | int | None = None,
    sheets: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    selected_sheets = sheets or list_phagocytosis_sheets(path)
    if not selected_sheets:
        raise ValueError(f"No phagocytosis sheets found in {path.name}")

    frames: list[pd.DataFrame] = []
    for sheet_name in selected_sheets:
        df = read_sheet(path, sheet_name)
        frames.append(
            extract_phagocytosis_from_sheet(
                df,
                patient_id=patient_id,
                source_file=path.name,
                sheet_name=sheet_name,
            )
        )

    result = pd.concat(frames, ignore_index=True)
    sort_cols = ["visit_day", "time_minutes", "treatment"]
    return result.sort_values(sort_cols, kind="stable").reset_index(drop=True)


TIMESERIES_COLUMNS = [
    "patient_id",
    "visit_day",
    "treatment",
    "time_minutes",
    "time",
    "adjusted_mfi",
    "raw_mfi",
    "blank_mfi",
    "n_replicates",
    "sheet_name",
    "source_file",
    "parser",
]


def finalize_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Return a consistently ordered time-series dataframe."""
    columns = [col for col in TIMESERIES_COLUMNS if col in df.columns]
    extra = [col for col in df.columns if col not in columns]
    return df[columns + extra].sort_values(
        ["patient_id", "visit_day", "time_minutes", "treatment"],
        kind="stable",
    ).reset_index(drop=True)


def endpoint_at_final_time(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient / visit / treatment at the last kinetic timepoint."""
    if df.empty:
        return df
    endpoint = (
        df.sort_values("time_minutes")
        .groupby(["patient_id", "visit_day", "treatment"], as_index=False)
        .tail(1)
    )
    return endpoint.rename(
        columns={
            "time": "final_time",
            "time_minutes": "final_time_minutes",
            "raw_mfi": "raw_mfi_final",
            "blank_mfi": "blank_mfi_final",
            "adjusted_mfi": "adjusted_mfi_final",
        }
    )


def parse_patient_id_from_folder(folder_name: str) -> int | None:
    match = re.search(r"#(\d+)", folder_name)
    return int(match.group(1)) if match else None


def discover_patient_workbooks(batch_dir: Path) -> list[tuple[int, Path]]:
    workbooks: list[tuple[int, Path]] = []
    for folder in sorted(batch_dir.glob("Burn Patient #*")):
        if not folder.is_dir():
            continue
        patient_id = parse_patient_id_from_folder(folder.name)
        if patient_id is None:
            continue
        for workbook in sorted(folder.glob("*.xlsx")):
            if workbook.name.startswith("~$"):
                continue
            workbooks.append((patient_id, workbook))
    return workbooks


def extract_all_patients(batch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract phagocytosis time series for every patient workbook in batch_dir."""
    frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    for patient_id, workbook in discover_patient_workbooks(batch_dir):
        for sheet_name in list_phagocytosis_sheets(workbook):
            try:
                raw_sheet = read_sheet(workbook, sheet_name)
                extracted = extract_phagocytosis_from_sheet(
                    raw_sheet,
                    patient_id=patient_id,
                    source_file=workbook.name,
                    sheet_name=sheet_name,
                )
                frames.append(extracted)
            except (ValueError, FileNotFoundError) as exc:
                errors.append(
                    {
                        "patient_id": patient_id,
                        "source_file": workbook.name,
                        "sheet_name": sheet_name,
                        "error": str(exc),
                    }
                )

    timeseries = finalize_timeseries(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    error_log = pd.DataFrame(errors)
    return timeseries, error_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract phagocytosis values from a plate-reader Excel/CSV export."
    )
    parser.add_argument(
        "--input",
        help="Path to a single Excel (.xlsx) or CSV plate-reader export.",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Extract all patient workbooks under this directory (Burn Patient #* folders).",
    )
    parser.add_argument(
        "--output",
        help="Output CSV path. Defaults to <input_stem>_phagocytosis.csv or extracted_data/phagocytosis_timeseries.csv for batch mode.",
    )
    parser.add_argument(
        "--endpoint-output",
        type=Path,
        help="Optional CSV with final-timepoint summary (batch mode). Defaults next to --output.",
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        help="Optional CSV listing sheets that could not be parsed (batch mode).",
    )
    parser.add_argument(
        "--patient-id",
        help="Optional patient identifier to include in the output.",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        dest="sheets",
        help="Process only the named sheet (repeatable). Defaults to all phagocytosis sheets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.input) == bool(args.batch_dir):
        print("Error: specify exactly one of --input or --batch-dir.", file=sys.stderr)
        return 1

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        output_path = (
            Path(args.output)
            if args.output
            else batch_dir / "extracted_data" / "phagocytosis_timeseries.csv"
        )
        endpoint_path = (
            Path(args.endpoint_output)
            if args.endpoint_output
            else output_path.with_name("phagocytosis_endpoint_final.csv")
        )
        errors_path = (
            Path(args.errors_output)
            if args.errors_output
            else output_path.with_name("phagocytosis_extraction_errors.csv")
        )

        try:
            result, errors = extract_all_patients(batch_dir)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if result.empty:
            print("Error: no phagocytosis data extracted.", file=sys.stderr)
            if not errors.empty:
                errors.to_csv(errors_path, index=False)
                print(f"Wrote error log to {errors_path}", file=sys.stderr)
            return 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        endpoint_at_final_time(result).to_csv(endpoint_path, index=False)
        if not errors.empty:
            errors.to_csv(errors_path, index=False)

        patients = sorted(result["patient_id"].dropna().unique().tolist())
        visit_days = sorted(result["visit_day"].dropna().unique().tolist())
        treatments = sorted(result["treatment"].unique().tolist(), key=treatment_sort_key)
        print(f"Wrote {len(result)} rows to {output_path}")
        print(f"Wrote {len(endpoint_at_final_time(result))} endpoint rows to {endpoint_path}")
        print(f"Patients: {', '.join(str(p) for p in patients)}")
        print(f"Visit days: {', '.join(visit_days)}")
        print(f"Treatments: {', '.join(treatments)}")
        if not errors.empty:
            print(f"Skipped {len(errors)} sheet(s); see {errors_path}")
        return 0

    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_phagocytosis.csv")
    )

    try:
        result = finalize_timeseries(
            extract_phagocytosis_file(
                input_path,
                patient_id=args.patient_id,
                sheets=args.sheets,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    result.to_csv(output_path, index=False)

    visit_days = sorted(result["visit_day"].dropna().unique().tolist())
    treatments = sorted(result["treatment"].unique().tolist(), key=treatment_sort_key)
    print(f"Wrote {len(result)} rows to {output_path}")
    print(f"Visit days: {', '.join(visit_days) if visit_days else 'n/a'}")
    print(f"Treatments: {', '.join(treatments)}")
    print(f"Timepoints per sheet: {result.groupby(['sheet_name', 'treatment']).size().max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
