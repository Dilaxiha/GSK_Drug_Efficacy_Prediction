"""
data_exploration.py
-------------------
Comprehensive Exploratory Data Analysis (EDA) & Data Quality Verification Pipeline.
Calculates summary statistics, VIF, target correlations, Mutual Information scores, 
and generates diagnostic visualizations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Global Visual Styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#eeeeee",
    "grid.linestyle": "--",
})

# Path Configurations
ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "dataset" / "feature_engineered_clinical_data.csv"
FALLBACK_PATH = ROOT / "data_cleaned.csv"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_PATH = OUTPUT_DIR / "Data_Exploration_Report.md"

TARGET = "treatment_outcome"


def log(message: str) -> None:
    print(f"[EDA Pipeline] {message}", flush=True)


def ensure_paths() -> None:
    """Create necessary output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load primary or fallback dataset safely."""
    if INPUT_PATH.exists():
        log(f"Loading primary dataset from: {INPUT_PATH}")
        return pd.read_csv(INPUT_PATH, low_memory=False)
    elif FALLBACK_PATH.exists():
        log(f"Primary dataset not found. Loading fallback from: {FALLBACK_PATH}")
        return pd.read_csv(FALLBACK_PATH, low_memory=False)
    else:
        raise FileNotFoundError(f"Neither {INPUT_PATH} nor {FALLBACK_PATH} was found.")


# ==============================================================================
# STATISTICAL & DATA AUDIT MODULES
# ==============================================================================

def run_integrity_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Check missing cells, target validity, near-zero variance, and separation."""
    if TARGET not in df.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")
    missing_cells = int(df.isna().sum().sum())
    target_vals = set(df[TARGET].unique())
    target_counts = df[TARGET].value_counts().sort_index()

    # Near-Zero Variance (NZV)
    nzv_list = []
    for col in df.columns:
        top_rate = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_rate > 0.99:
            nzv_list.append({"feature": col, "dominant_rate": float(top_rate)})

    return {
        "missing_cells": missing_cells,
        "target_values": target_vals,
        "target_counts": target_counts,
        "nzv_features": nzv_list,
    }


def compute_vif(df: pd.DataFrame, max_samples: int = 50_000) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) on numeric features to check redundancy."""
    num_df = df.select_dtypes(include=[np.number]).drop(columns=[TARGET, "patient_id"], errors="ignore")
    num_df = num_df.loc[:, num_df.nunique() > 1].dropna(axis=1, how="any")

    if num_df.empty:
        return pd.DataFrame(columns=["feature", "vif", "warning"])

    if len(num_df) > max_samples:
        num_df = num_df.sample(max_samples, random_state=42)

    scaled = (num_df - num_df.mean()) / num_df.std(ddof=0).replace(0, 1)
    
    vif_records = []
    for idx, col in enumerate(scaled.columns):
        try:
            vif_val = float(variance_inflation_factor(scaled.to_numpy(), idx))
        except Exception:
            vif_val = float("inf")
        
        vif_records.append({
            "feature": col,
            "vif": vif_val,
            "warning": vif_val > 10.0,
        })

    return pd.DataFrame(vif_records).sort_values("vif", ascending=False, ignore_index=True)


def compute_mutual_information(df: pd.DataFrame, max_samples: int = 25_000) -> pd.DataFrame:
    """Calculate Mutual Information to discover non-linear feature relationships."""
    valid_rows = df[TARGET].notna()
    X = df.loc[valid_rows].drop(columns=[TARGET, "patient_id"], errors="ignore")
    y = df.loc[valid_rows, TARGET]
    if len(X) > max_samples:
        sample_idx = X.sample(max_samples, random_state=42).index
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]

    # Limit high-cardinality text columns only for this diagnostic calculation.
    for column in X.select_dtypes(include=["object", "category"]).columns:
        top_values = X[column].value_counts(dropna=True).head(100).index
        X[column] = X[column].where(X[column].isin(top_values), "__Other__")

    X_encoded = pd.get_dummies(X, drop_first=True).replace([np.inf, -np.inf], np.nan)
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True)).fillna(0)

    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    return pd.DataFrame({
        "feature": X_encoded.columns,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False, ignore_index=True)


def compute_categorical_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze categorical feature strength using Chi-Square and Cramer's V."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if TARGET in cat_cols:
        cat_cols.remove(TARGET)

    records = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[TARGET])
        chi2, p_val, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0.0

        records.append({
            "feature": col,
            "chi2_stat": float(chi2),
            "p_value": float(p_val),
            "cramers_v": float(cramers_v),
        })

    return pd.DataFrame(records).sort_values("cramers_v", ascending=False, ignore_index=True)


# ==============================================================================
# VISUALIZATION MODULES
# ==============================================================================

def plot_target_imbalance(df: pd.DataFrame) -> None:
    """Generate a clear, styled bar plot showing target class imbalance."""
    counts = df[TARGET].value_counts().sort_index()
    total = len(df)
    
    labels = ["Ineffective (0)", "Effective (1)"] if set(counts.index).issubset({0, 1}) else [str(k) for k in counts.index]
    colors = ["#287271", "#b23a48"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="none")

    for bar, count in zip(bars, counts.values):
        pct = (count / total) * 100
        ax.annotate(
            f"{count:,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Target Class Distribution (treatment_outcome)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_class_imbalance.png", dpi=200)
    plt.close(fig)


def plot_mutual_information(mi_df: pd.DataFrame) -> None:
    """Plot top 15 non-linear predictors ranked by Mutual Information."""
    top_15 = mi_df.head(15).sort_values("mi_score", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_15["feature"], top_15["mi_score"], color="#287271", height=0.6)
    ax.set_xlabel("Mutual Information Score (Predictive Power)", fontsize=11)
    ax.set_title("Top 15 Features by Mutual Information Score", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mutual_information_scores.png", dpi=200)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot lower-triangle correlation heatmap across numeric variables."""
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["patient_id"], errors="ignore")
    if num_df.shape[1] < 2:
        return

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
        annot=num_df.shape[1] <= 15,
        fmt=".2f",
        annot_kws={"size": 7},
    )
    ax.set_title("Numeric Inter-Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

def main() -> None:
    ensure_paths()
    log("Starting Data Exploration & Diagnostics Pipeline...")
    
    # 1. Load Data
    df = load_dataset()
    input_path = INPUT_PATH if INPUT_PATH.exists() else FALLBACK_PATH
    log(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Integrity Audit
    log("Running integrity & missingness checks...")
    audit = run_integrity_audit(df)
    log(f"Missing Cells: {audit['missing_cells']:,}")

    # 3. Target Imbalance & Visualization
    log("Generating Target Imbalance plot...")
    plot_target_imbalance(df)

    # 4. Multicollinearity Diagnostic (VIF)
    log("Calculating Variance Inflation Factor (VIF)...")
    vif_df = compute_vif(df)

    # 5. Non-Linear Signal Extraction (Mutual Information)
    log("Computing Mutual Information scores...")
    mi_df = compute_mutual_information(df)
    plot_mutual_information(mi_df)

    # 6. Categorical Analysis
    log("Evaluating Categorical Associations (Cramer's V)...")
    cat_df = compute_categorical_associations(df)

    # 7. Heatmap Visuals
    log("Generating Correlation Heatmap...")
    plot_correlation_heatmap(df)

    # Output Summary
    log("=== EXPLORATION SUMMARY ===")
    print("\n--- TOP 5 MUTUAL INFORMATION PREDICTORS ---")
    print(mi_df.head(5).to_string(index=False))

    if not vif_df.empty and vif_df["warning"].any():
        print("\n--- MULTICOLLINEARITY WARNINGS (VIF > 10) ---")
        print(vif_df[vif_df["warning"]].head(5).to_string(index=False))

    if not cat_df.empty:
        print("\n--- TOP CATEGORICAL PREDICTORS (CRAMER'S V) ---")
        print(cat_df.head(5).to_string(index=False))

    report_lines = [
        "# Data Exploration and Verification Report",
        "",
        f"- **Input dataset:** `{input_path}`",
        f"- **Dimensions:** {df.shape[0]:,} rows x {df.shape[1]:,} columns",
        f"- **Missing cells:** {audit['missing_cells']:,}",
        f"- **Target values:** {sorted(str(value) for value in audit['target_values'])}",
        "",
        "## Target Distribution",
        "",
        audit["target_counts"].to_string(),
        "",
        "## Mutual Information",
        "",
        mi_df.head(15).to_string(index=False),
        "",
        "## Variance Inflation Factor",
        "",
        vif_df.head(15).to_string(index=False) if not vif_df.empty else "No numeric features available.",
        "",
        "## Categorical Associations",
        "",
        cat_df.head(15).to_string(index=False) if not cat_df.empty else "No categorical features available.",
        "",
        "## Generated Figures",
        "",
        "- `figures/target_class_imbalance.png`",
        "- `figures/mutual_information_scores.png`",
        "- `figures/correlation_heatmap.png`",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    log(f"EDA summary report saved to '{REPORT_PATH}'")

    log(f"Data exploration pipeline executed successfully! Visuals exported to '{FIGURES_DIR}/'")


if __name__ == "__main__":
    main()