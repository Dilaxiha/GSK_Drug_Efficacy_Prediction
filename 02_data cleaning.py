from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Path configurations
ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "dataset" / "clinical_data_raw.csv"
if not INPUT.exists():
    INPUT = ROOT / "clinical_data_raw.csv"
OUTPUT = ROOT / "dataset" / "clinical_data_cleaned.csv"
REPORT = ROOT / "output" / "Step2_Cleaning_Report.md"

OUTPUT.parent.mkdir(exist_ok=True)
REPORT.parent.mkdir(exist_ok=True)

PLACEHOLDERS = {"", "na", "n/a", "none", "null", "-", "?", "missing", "unknown"}
TARGET_MAP = {"0": 0, "0.0": 0, "no": 0, "ineffective": 0, "false": 0, "1": 1, "1.0": 1, "yes": 1, "effective": 1, "true": 1}

CONTINUOUS_FIELDS = [
    "age", "weight_kg", "bmi", "hemoglobin", "wbc_count", 
    "alt_enzyme", "ast_enzyme", "creatinine", "egfr", 
    "hba1c", "dosage", "duration_days"
]

TARGET = "treatment_outcome"


def mark_placeholders(value: object) -> object:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return np.nan if text.lower() in PLACEHOLDERS else text


def consolidate_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Automatically cleans column names, combines split values across duplicate pairs, and drops uninformative fields."""
    # 1. Clean leading/trailing whitespace from column headers
    df.columns = df.columns.str.strip()
    dropped_columns = []

    # 2. Merge duplicate variants case-insensitively, preserving the first value.
    consolidated: dict[str, pd.Series] = {}
    original_names: dict[str, list[str]] = {}
    for column in df.columns:
        key = column.lower()
        if key in consolidated:
            consolidated[key] = consolidated[key].combine_first(df[column])
            original_names[key].append(column)
        else:
            consolidated[key] = df[column]
            original_names[key] = [column]
    df = pd.DataFrame(consolidated, index=df.index)
    for key, names in original_names.items():
        if len(names) > 1:
            dropped_columns.extend(names[1:])

    # 3. Standardize column names to lowercase
    df.columns = df.columns.str.lower()

    # 4. Remove redundant unit features and empty placeholder fields
    cols_to_remove = ["weight_lbs", "extra_col_1", "extra_col_2", "unnamed_0", "unnamed: 0", "notes"]
    existing_junk = [col for col in cols_to_remove if col in df.columns]
    
    if existing_junk:
        df = df.drop(columns=existing_junk)
        dropped_columns.extend(existing_junk)

    return df, dropped_columns


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = raw.copy()
    metrics: dict[str, object] = {"clip_summary": {}, "imputation_summary": {}}

    # Step 1: Consolidate Schema and drop unneeded columns
    frame, dropped_cols = consolidate_schema(frame)
    metrics["dropped_columns"] = dropped_cols

    # Step 2: Convert Placeholder Strings to NaN
    for col in frame.columns:
        if frame[col].dtype == "object" or pd.api.types.is_string_dtype(frame[col]):
            frame[col] = frame[col].map(mark_placeholders)

    # Step 3: Categorical Normalization
    if "gender" in frame.columns:
        gender_norm = frame["gender"].astype("string").str.strip().str.lower()
        frame["gender"] = gender_norm.map({"m": "M", "male": "M", "f": "F", "female": "F", "other": "Other"}).fillna("Other")

    if "ethnicity" in frame.columns:
        frame["ethnicity"] = frame["ethnicity"].astype("string").str.strip().str.title()

    if TARGET in frame.columns:
        target_norm = frame[TARGET].astype("string").str.strip().str.lower()
        frame[TARGET] = target_norm.map(TARGET_MAP)
        # Remove records where target outcome is missing
        frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)

    # Step 4: Cast Continuous Fields to Float64
    for col in CONTINUOUS_FIELDS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    # Step 5: IQR Outlier Capping
    for col in CONTINUOUS_FIELDS:
        if col in frame.columns:
            vals = frame[col]
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            clipped = vals.notna() & ((vals < lower) | (vals > upper))
            frame[col] = vals.clip(lower=lower, upper=upper)
            metrics["clip_summary"][col] = {"lower": lower, "upper": upper, "clipped": int(clipped.sum())}

    # Step 6: Imputation
    exclude_imputation = {"patient_id", TARGET}
    
    # Numeric Imputation (Median)
    numeric_cols = [col for col in frame.select_dtypes(include=[np.number]).columns if col not in exclude_imputation]
    for col in numeric_cols:
        missing = int(frame[col].isna().sum())
        if missing:
            med_val = frame[col].median()
            frame[col] = frame[col].fillna(med_val)
            metrics["imputation_summary"][col] = {"method": "median", "value": med_val, "missing": missing}

    # Categorical Imputation (Mode)
    categorical_cols = [col for col in frame.select_dtypes(include=["object", "string"]).columns if col not in exclude_imputation]
    for col in categorical_cols:
        missing = int(frame[col].isna().sum())
        if missing:
            modes = frame[col].mode(dropna=True)
            fill_val = modes.iloc[0] if len(modes) else "Unknown"
            frame[col] = frame[col].fillna(fill_val)
            metrics["imputation_summary"][col] = {"method": "mode", "value": str(fill_val), "missing": missing}

    # Step 7: Remove Duplicate Rows
    dups = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    metrics["duplicates_removed"] = dups

    return frame, metrics


def write_report(raw: pd.DataFrame, clean: pd.DataFrame, metrics: dict[str, object]) -> None:
    clipping_rows = "\n".join(
        f"| {column} | {details['lower']:.6g} | {details['upper']:.6g} | {details['clipped']:,} |"
        for column, details in metrics["clip_summary"].items()
    ) or "| None | NA | NA | 0 |"
    imputation_rows = "\n".join(
        f"| {column} | {details['method']} | {details['value']} | {details['missing']:,} |"
        for column, details in metrics["imputation_summary"].items()
    ) or "| None | NA | NA | 0 |"
    dropped_rows = ", ".join(metrics.get("dropped_columns", [])) or "None"
    report = f"""# Step 2: Data Cleaning and Preprocessing Report

## Dataset Metrics

- **Input:** `{INPUT}`
- **Output:** `{OUTPUT}`
- **Before:** {len(raw):,} rows x {raw.shape[1]:,} columns
- **After:** {len(clean):,} rows x {clean.shape[1]:,} columns
- **Target-missing rows dropped:** {len(raw) - len(clean) - metrics['duplicates_removed']:,}
- **Duplicate rows removed:** {metrics['duplicates_removed']:,}
- **Dropped columns:** {dropped_rows}

## Processing Applied

1. Stripped column-header whitespace and consolidated duplicate variants case-insensitively using `combine_first`.
2. Standardized headers to lowercase and removed redundant, empty, and free-text columns.
3. Converted placeholders to missing values, normalized gender and ethnicity, and mapped `treatment_outcome` to binary values.
4. Coerced numeric fields to `float64`.
5. Calculated 1.5 x IQR bounds, clipped only the requested continuous fields, then imputed numeric medians and categorical modes.
6. Removed exact duplicate rows after consolidation.

## Outlier Clipping

| Column | Lower bound | Upper bound | Values clipped |
| --- | --- | --- | --- |
{clipping_rows}

## Imputation

| Column | Method | Value | Missing before |
| --- | --- | --- | --- |
{imputation_rows}

## Validation

- Target values restricted to `0` and `1`: **{'PASS' if clean[TARGET].isin([0, 1]).all() else 'FAIL'}**
- Gender values restricted to `M`, `F`, `Other`: **{'PASS' if set(clean['gender'].unique()).issubset({'M', 'F', 'Other'}) else 'FAIL'}**
- Duplicate rows remaining: **{int(clean.duplicated().sum()):,}**
- Remaining missing cells: **{int(clean.isna().sum().sum()):,}**
"""
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    raw = pd.read_csv(INPUT, low_memory=False, keep_default_na=True)
    clean, metrics = clean_data(raw)
    clean.to_csv(OUTPUT, index=False)
    write_report(raw, clean, metrics)
    print(f"Cleaning complete! Dimensions transformed from {raw.shape} to {clean.shape}")
    print(f"Clean dataset saved: {OUTPUT}")
    print(f"Cleaning report updated: {REPORT}")


if __name__ == "__main__":
    main()