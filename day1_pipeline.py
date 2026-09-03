from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REQUESTED_INPUT = ROOT / "dataset" / "clinical_data_raw.csv"
INPUT = REQUESTED_INPUT if REQUESTED_INPUT.exists() else ROOT / "clinical_data_raw.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Treatment_Outcome"
PLOT_FIELDS = [
    "age", "weight_kg", "bmi", "hemoglobin", "wbc_count", "alt_enzyme",
    "ast_enzyme", "creatinine", "egfr", "hba1c", "dosage", "duration_days",
]
SOURCE_MAP = {
    "age": ["Age", " age", "age"], "weight_kg": ["Weight_kg", "weight_kg"],
    "bmi": ["BMI", "bmi"], "hemoglobin": ["Hemoglobin", "hemoglobin"],
    "wbc_count": ["WBC_Count", "wbc_count"], "alt_enzyme": ["ALT_Enzyme", "alt_enzyme"],
    "ast_enzyme": ["AST_Enzyme", "ast_enzyme"], "creatinine": ["Creatinine", "creatinine"],
    "egfr": ["eGFR", "egfr"], "hba1c": ["HbA1c", "hba1c"], "dosage": ["Dosage", "dosage"],
    "duration_days": ["Duration_Days", "duration_days"],
}
NUMERIC_COLUMNS = {
    "Patient_ID", "patient_id", "Age", " age", "Weight_kg", "Weight_lbs", "Height_cm", "BMI",
    "Systolic_BP", "Diastolic_BP", "Heart_Rate", "Temperature_F", "Hemoglobin", "WBC_Count",
    "ALT_Enzyme", "AST_Enzyme", "Creatinine", "eGFR", "HbA1c", "Total_Cholesterol", "Dosage",
    "Duration_Days", "Concurrent_Drugs", "Readmission_30d",
}
VALID_CATEGORIES = {
    "Gender": {"m", "male", "f", "female", "other", "unknown", "0", "1"},
    "gender ": {"m", "male", "f", "female", "other", "unknown", "0", "1"},
    "Route": {"oral", "iv", "intravenous", "im", "intramuscular", "sc", "subcutaneous", "unknown"},
    "Treatment_Outcome": {"0", "1", "yes", "no", "effective", "ineffective", "true", "false"},
}
PLACEHOLDERS = {"na", "n/a", "none", "null", "-", "?", "unknown", "missing"}
CLINICAL_BOUNDS = {
    "age": (0, 120), "weight_kg": (20, 400), "bmi": (10, 80), "hemoglobin": (3, 25),
    "wbc_count": (1, 100), "alt_enzyme": (0, 1000), "ast_enzyme": (0, 1000), "creatinine": (0, 20),
    "egfr": (0, 250), "hba1c": (2, 20), "dosage": (0, 5000), "duration_days": (0, 3650),
    "concurrent_drugs": (0, 50),
}


def source_values(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = next((name for name in candidates if name in frame.columns), None)
    return frame[column] if column else pd.Series(index=frame.index, dtype="object")


def examples(values: pd.Series, limit: int = 6) -> str:
    return ", ".join(repr(str(value)) for value in values.dropna().astype(str).drop_duplicates().head(limit))


def audit(raw: pd.DataFrame) -> tuple[str, dict[str, int], dict[str, str]]:
    lines = ["RAW DATA PROBLEM AUDIT", f"Source: {INPUT}", f"Dimensions: {len(raw):,} rows x {raw.shape[1]:,} columns", "Audit scope: complete raw file; no sampling, cleaning, imputation, or outlier removal.", "", "[1] SCHEMA AND FORMATTING"]
    duplicate_headers = sorted({name for name in raw.columns if list(raw.columns).count(name) > 1})
    whitespace_headers = [name for name in raw.columns if name != name.strip()]
    if duplicate_headers:
        lines.append(f"- Duplicate column names: {duplicate_headers}")
    if whitespace_headers:
        lines.append(f"- Column names with leading/trailing whitespace: {whitespace_headers}")
    whitespace_found = bool(duplicate_headers or whitespace_headers)
    for column in raw.select_dtypes(include=["object", "string"]).columns:
        mask = raw[column].astype("string").str.match(r"^\s|\s$", na=False)
        if mask.any():
            whitespace_found = True
            lines.append(f"- {column!r}: {int(mask.sum()):,} values contain leading/trailing whitespace; examples: {examples(raw.loc[mask, column])}")
    if not whitespace_found:
        lines.append("- None detected.")

    lines.append("\n[2] MISSING VALUES AND PLACEHOLDERS")
    missing_counts = {}
    placeholder_found = False
    for column in raw.columns:
        values = raw[column].astype("string")
        missing = values.isna() | values.str.strip().eq("")
        count = int(missing.sum())
        missing_counts[column] = count
        if count:
            lines.append(f"- {column!r}: {count:,} blank/missing values ({count / len(raw):.2%})")
        placeholder = values.str.strip().str.lower().isin(PLACEHOLDERS).fillna(False)
        if placeholder.any():
            placeholder_found = True
            lines.append(f"- {column!r}: {int(placeholder.sum()):,} placeholder values ({examples(raw.loc[placeholder, column])})")
    if not any(missing_counts.values()) and not placeholder_found:
        lines.append("- None detected.")

    lines.append("\n[3] DATA TYPES AND CONVERSION FAILURES")
    dtype_map = {column: str(raw[column].dtype) for column in raw.columns}
    conversion_found = False
    for column in raw.columns:
        if column in NUMERIC_COLUMNS:
            numeric = pd.to_numeric(raw[column], errors="coerce")
            failures = raw[column].notna() & numeric.isna() & raw[column].astype("string").str.strip().ne("")
            if failures.any():
                conversion_found = True
                lines.append(f"- {column!r}: {int(failures.sum()):,} non-empty values fail numeric conversion; examples: {examples(raw.loc[failures, column])}")
    if "Admission_Date" in raw:
        parsed = pd.to_datetime(raw["Admission_Date"], errors="coerce", format="mixed")
        failures = raw["Admission_Date"].notna() & parsed.isna() & raw["Admission_Date"].astype("string").str.strip().ne("")
        if failures.any():
            conversion_found = True
            lines.append(f"- 'Admission_Date': {int(failures.sum()):,} non-empty values fail date parsing; examples: {examples(raw.loc[failures, 'Admission_Date'])}")
    if not conversion_found:
        lines.append("- None detected.")

    lines.append("\n[4] INVALID CATEGORICAL CODES AND SPELLING VARIATION")
    category_found = False
    for column, valid in VALID_CATEGORIES.items():
        if column not in raw:
            continue
        values = raw[column].astype("string").str.strip().str.lower()
        invalid = values.notna() & values.ne("") & ~values.isin(valid)
        spellings = raw[column].dropna().astype(str).str.strip()
        if invalid.any():
            category_found = True
            lines.append(f"- {column!r}: {int(invalid.sum()):,} values outside expected codes {sorted(valid)}; examples: {examples(raw.loc[invalid, column])}")
        if spellings.nunique() > 1:
            category_found = True
            lines.append(f"- {column!r}: {spellings.nunique():,} distinct raw spellings; examples: {examples(spellings)}")
    if not category_found:
        lines.append("- None detected.")

    lines.append("\n[5] EXTREME AND CLINICALLY IMPLAUSIBLE NUMERIC VALUES")
    outlier_found = False
    for field, (low, high) in CLINICAL_BOUNDS.items():
        values = pd.to_numeric(source_values(raw, SOURCE_MAP.get(field, [field])), errors="coerce")
        invalid = values.notna() & ((values < low) | (values > high))
        if invalid.any():
            outlier_found = True
            lines.append(f"- {field!r}: {int(invalid.sum()):,} values outside plausible range [{low}, {high}]; observed min={values.min():.4g}, max={values.max():.4g}; examples: {examples(values.loc[invalid])}")
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            iqr_outliers = values.notna() & ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr))
            if iqr_outliers.any():
                outlier_found = True
                lines.append(f"- {field!r}: {int(iqr_outliers.sum()):,} statistical IQR outliers; bounds [{q1 - 1.5 * iqr:.4g}, {q3 + 1.5 * iqr:.4g}]")
    if not outlier_found:
        lines.append("- None detected.")

    lines.append("\n[6] DUPLICATE ROWS")
    duplicate_count = int(raw.duplicated().sum())
    lines.append(f"- Exact duplicate rows: {duplicate_count:,} ({duplicate_count / len(raw):.2%})")
    lines.append("\n[7] TARGET DISTRIBUTION")
    if TARGET in raw:
        target = raw[TARGET].astype("string").str.strip().str.lower()
        for label, count in target.value_counts(dropna=False).items():
            lines.append(f"- treatment_outcome={label!r}: {int(count):,} ({count / len(raw):.2%})")
        binary = target[target.isin(VALID_CATEGORIES[TARGET])].map(lambda value: "1" if value in {"1", "yes", "effective", "true"} else "0").value_counts()
        if len(binary) == 2:
            lines.append(f"- Binary minority/majority ratio: {binary.min() / binary.max():.4f}")
    else:
        lines.append("- ERROR: treatment_outcome target column is absent.")
    return "\n".join(lines) + "\n", missing_counts, dtype_map


def plot_distributions(raw: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(3, 4, figsize=(16, 10))
    for axis, field in zip(axes.flat, PLOT_FIELDS):
        values = pd.to_numeric(source_values(raw, SOURCE_MAP[field]), errors="coerce")
        values = values[np.isfinite(values)].dropna()
        axis.hist(values, bins=40, color="#1d6f72", alpha=0.85)
        axis.set_title(field)
        axis.set_xlabel("Raw value")
        axis.set_ylabel("Count")
        axis.grid(alpha=0.2)
    figure.suptitle("Raw Continuous Variable Distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def main() -> None:
    raw = pd.read_csv(INPUT, low_memory=False, keep_default_na=True)
    audit_text, missing_counts, dtype_map = audit(raw)
    (OUTPUT_DIR / "raw_data_problems.txt").write_text(audit_text, encoding="utf-8")
    plot_distributions(raw, OUTPUT_DIR / "step1_raw_distributions.png")
    missing_rows = [[column, f"{count:,}", f"{count / len(raw):.2%}"] for column, count in missing_counts.items() if count]
    dtype_rows = [[column, dtype_map[column]] for column in raw.columns]
    target_rows = []
    if TARGET in raw:
        target = raw[TARGET].astype("string").str.strip().str.lower()
        target_rows = [[str(label), f"{int(count):,}", f"{count / len(raw):.2%}"] for label, count in target.value_counts(dropna=False).items()]
    report = f"""# Step 1: Raw Data Problem Audit and Initial Diagnostic EDA

## Dataset Overview

- **Input:** `{INPUT}`
- **Raw dimensions:** {len(raw):,} rows x {raw.shape[1]:,} columns
- **Exact duplicate rows:** {int(raw.duplicated().sum()):,}
- **Audit scope:** Complete raw file. No cleaning, sampling, imputation, or outlier removal was performed.

## Column Data Types

{markdown_table(dtype_rows, ['Column', 'Pandas dtype'])}

## Missing Values

{markdown_table(missing_rows or [['None detected', '0', '0.00%']], ['Column', 'Missing count', 'Missing rate'])}

## Target Distribution: `treatment_outcome`

{markdown_table(target_rows or [['Target absent', '0', '0.00%']], ['Raw label', 'Count', 'Rate'])}

## Outputs

- Complete line-by-line anomaly inventory: `raw_data_problems.txt`
- Raw distributions for age, weight, BMI, hemoglobin, WBC, liver enzymes, creatinine, eGFR, HbA1c, dosage, and duration: `step1_raw_distributions.png`

The anomaly inventory includes missing values, placeholders, formatting whitespace, duplicate headers, type-conversion failures, invalid categorical codes, clinical-bound violations, statistical IQR outliers, duplicate rows, and target imbalance.
"""
    (OUTPUT_DIR / "Step1_Initial_EDA_Report.md").write_text(report, encoding="utf-8")
    print(f"Step 1 complete: raw_shape=({len(raw)}, {raw.shape[1]}); outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
