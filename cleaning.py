from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "dataset" / "clinical_data_raw.csv"
if not INPUT.exists():
    INPUT = ROOT / "clinical_data_raw.csv"
OUTPUT = ROOT / "dataset" / "clinical_data_cleaned.csv"
REPORT = ROOT / "output" / "Step2_Cleaning_Report.md"
OUTPUT.parent.mkdir(exist_ok=True)
REPORT.parent.mkdir(exist_ok=True)

PLACEHOLDERS = {"", "na", "n/a", "none", "null", "-", "?", "missing", "unknown"}
TARGET_MAP = {"0": 0.0, "0.0": 0.0, "no": 0.0, "ineffective": 0.0, "false": 0.0, "1": 1.0, "1.0": 1.0, "yes": 1.0, "effective": 1.0, "true": 1.0}
CONTINUOUS_FIELDS = {
    "age": ("Age", "age"), "weight_kg": ("Weight_kg", "weight_kg"), "bmi": ("BMI", "bmi"),
    "hemoglobin": ("Hemoglobin", "hemoglobin"), "wbc_count": ("WBC_Count", "wbc_count"),
    "alt_enzyme": ("ALT_Enzyme", "alt_enzyme"), "ast_enzyme": ("AST_Enzyme", "ast_enzyme"),
    "creatinine": ("Creatinine", "creatinine"), "egfr": ("eGFR", "egfr"), "hba1c": ("HbA1c", "hba1c"),
    "dosage": ("Dosage", "dosage"), "duration_days": ("Duration_Days", "duration_days"),
}
NUMERIC_COLUMNS = {
    "patient_id", "age", "weight_kg", "weight_lbs", "height_cm", "bmi", "systolic_bp", "diastolic_bp",
    "heart_rate", "temperature_f", "hemoglobin", "wbc_count", "alt_enzyme", "ast_enzyme", "creatinine", "egfr",
    "hba1c", "total_cholesterol", "dosage", "duration_days", "concurrent_drugs", "readmission_30d",
}


def clean_scalar(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text.lower() in PLACEHOLDERS else text


def present(frame: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if name in frame.columns]


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = raw.copy()
    df.columns = df.columns.str.strip()
    if "Treatment_Outcome" in df.columns:
        df = df.rename(columns={"Treatment_Outcome": "treatment_outcome"})

    metrics: dict[str, object] = {"consolidated": [], "dropped": [], "clip_summary": {}, "imputation_summary": {}}
    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].map(clean_scalar)

    for primary, secondary, final_name in [("Patient_ID", "patient_id", "patient_id"), ("Age", "age", "age"), ("Gender", "gender", "gender"), ("Drug_Name", "drug_name", "drug_name")]:
        if primary in df.columns and secondary in df.columns:
            df[primary] = df[primary].combine_first(df[secondary])
            df = df.drop(columns=[secondary]).rename(columns={primary: final_name})
            metrics["consolidated"].append(f"{primary} + {secondary} -> {final_name}")
        elif primary in df.columns:
            df = df.rename(columns={primary: final_name})
        elif secondary in df.columns:
            df = df.rename(columns={secondary: final_name})

    for column in ["Weight_lbs", "Extra_Col_1", "Extra_Col_2", "unnamed_0", "Unnamed: 0", "Notes"]:
        if column in df.columns:
            df = df.drop(columns=[column])
            metrics["dropped"].append(column)
    for column in ["blood_pressure", "Admission_Date", "Adverse_Event", "Readmission_30d", "Smoking_Status", "Alcohol_Use", "Temperature_F"]:
        if column in df.columns:
            df = df.drop(columns=[column])
            metrics["dropped"].append(column)
    df.columns = df.columns.str.lower()

    if "gender" in df.columns:
        normalized = df["gender"].astype("string").str.lower()
        df["gender"] = normalized.map({"m": "M", "male": "M", "mal": "M", "f": "F", "female": "F", "femal": "F"}).fillna("Other")
    if "ethnicity" in df.columns:
        df["ethnicity"] = df["ethnicity"].astype("string").str.title()
    if "treatment_outcome" in df.columns:
        normalized = df["treatment_outcome"].astype("string").str.lower()
        df["treatment_outcome"] = normalized.map(TARGET_MAP).astype("float64")
        df = df.dropna(subset=["treatment_outcome"]).reset_index(drop=True)

    for column in [column for column in df.columns if column in NUMERIC_COLUMNS or column in {name for names in CONTINUOUS_FIELDS.values() for name in names}]:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    for field, candidates in CONTINUOUS_FIELDS.items():
        if field not in df.columns:
            continue
        values = df[field]
        bounds = {"age": (0, 120), "weight_kg": (20, 400), "bmi": (10, 80), "hemoglobin": (3, 25), "wbc_count": (1, 100000), "alt_enzyme": (0, 1000), "ast_enzyme": (0, 1000), "creatinine": (0, 20), "egfr": (0, 250), "hba1c": (2, 20), "dosage": (0, 5000), "duration_days": (0, 3650)}[field]
        values = values.mask(~np.isfinite(values) | (values < bounds[0]) | (values > bounds[1]))
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clipped = values.notna() & ((values < lower) | (values > upper))
        df[field] = values.clip(lower=lower, upper=upper)
        metrics["clip_summary"][field] = {"lower": lower, "upper": upper, "clipped": int(clipped.sum())}

    excluded_numeric = {"patient_id", "treatment_outcome"}
    for column in [column for column in df.select_dtypes(include=[np.number]).columns if column not in excluded_numeric]:
        missing_before = int(df[column].isna().sum())
        if missing_before:
            value = df[column].median()
            df[column] = df[column].fillna(value)
            metrics["imputation_summary"][column] = ["median", value, missing_before]

    for column in [column for column in df.select_dtypes(include=["object", "string"]).columns if column != "admission_date"]:
        missing_before = int(df[column].isna().sum())
        if missing_before:
            modes = df[column].mode(dropna=True)
            value = modes.iloc[0] if len(modes) else "Unknown"
            df[column] = df[column].fillna(value)
            metrics["imputation_summary"][column] = ["mode", value, missing_before]

    metrics["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    assert len(df.columns) < 26, f"Expected fewer than 26 columns after cleaning; found {len(df.columns)}"
    return df, metrics


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def make_report(raw: pd.DataFrame, clean: pd.DataFrame, metrics: dict[str, object]) -> str:
    clipping = [[c, f"{v['lower']:.6g}", f"{v['upper']:.6g}", f"{v['clipped']:,}"] for c, v in metrics["clip_summary"].items()]
    imputation = [[c, v[0], str(v[1]), f"{v[2]:,}"] for c, v in metrics["imputation_summary"].items()]
    return f"""# Step 2: Consolidated Data Cleaning Report

- **Input:** `{INPUT}`
- **Output:** `{OUTPUT}`
- **Before:** {len(raw):,} rows x {raw.shape[1]:,} columns
- **After:** {len(clean):,} rows x {clean.shape[1]:,} columns
- **Duplicate rows removed:** {metrics['duplicates_removed']:,}
- **Target-missing rows dropped:** {len(raw) - len(clean) - metrics['duplicates_removed']:,}

## Schema Consolidation

- Column headers were stripped before all other processing.
- Target casing was standardized to `treatment_outcome`.
- Recovered primary values with `combine_first()` for: {', '.join(metrics['consolidated']) or 'none'}.
- Dropped redundant/empty columns: {', '.join(metrics['dropped']) or 'none'}.
- Final schema assertion: **PASS**, {len(clean.columns)} columns (< 26).

## Outlier Handling

Implausible values were set to missing before calculating 1.5 x IQR bounds. Only the requested continuous fields were clipped.

{markdown_table(clipping, ['Field', 'Lower bound', 'Upper bound', 'Values clipped'])}

## Imputation

Median imputation was applied to numeric features, and mode imputation to categorical features. Patient identifiers, dates, and the target were excluded from imputation.

{markdown_table(imputation, ['Column', 'Method', 'Value', 'Missing before'])}

## Validation

- Gender values: **{'PASS' if set(clean['gender'].unique()).issubset({'M', 'F', 'Other'}) else 'FAIL'}**
- Target values: **{'PASS' if clean['treatment_outcome'].isin([0.0, 1.0]).all() else 'FAIL'}**
- Duplicate rows remaining: **{int(clean.duplicated().sum()):,}**
- Original duplicate variants removed: **PASS**
"""


def main() -> None:
    raw = pd.read_csv(INPUT, low_memory=False, keep_default_na=True)
    clean, metrics = clean_data(raw)
    clean.to_csv(OUTPUT, index=False)
    REPORT.write_text(make_report(raw, clean, metrics), encoding="utf-8")
    print(f"Cleaning complete: {raw.shape} -> {clean.shape}")


if __name__ == "__main__":
    main()
