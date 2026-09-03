"""Shared, leakage-safe utilities for the clinical classification pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "feature_engineered_clinical_data.csv"
METRICS_DIR = ROOT / "metrics_output"
REPORTS_DIR = ROOT / "reports"
TARGET = "treatment_outcome"
RANDOM_STATE = 42


def ensure_output_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class FrequencyCategoryEncoder:
    """Encode numeric fields directly and categoricals with train-only codes."""

    def __init__(self, max_categories: int = 100) -> None:
        self.max_categories = max_categories
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.medians: dict[str, float] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.feature_names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "FrequencyCategoryEncoder":
        for column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if pd.api.types.is_numeric_dtype(frame[column]) or converted.notna().mean() >= 0.95:
                self.numeric_columns.append(column)
                values = converted.replace([np.inf, -np.inf], np.nan)
                median = values.median()
                self.medians[column] = float(median) if pd.notna(median) else 0.0
            else:
                self.categorical_columns.append(column)
                values = frame[column].astype("string").fillna("__MISSING__")
                top = values.value_counts().head(self.max_categories).index
                self.category_maps[column] = {str(value): index for index, value in enumerate(top)}
        self.feature_names = self.numeric_columns + self.categorical_columns
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        for column in self.numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            output[column] = values.replace([np.inf, -np.inf], np.nan).fillna(self.medians[column]).astype("float32")
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__").astype(str)
            output[column] = values.map(self.category_maps[column]).fillna(-1).astype("int16")
        return output[self.feature_names]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")
    frame = frame.dropna(subset=[TARGET]).copy()
    y_numeric = pd.to_numeric(frame.pop(TARGET), errors="coerce")
    if y_numeric.isna().any():
        raise ValueError("treatment_outcome contains non-numeric values after missing-target removal.")
    y = y_numeric.astype("int8")
    if not set(y.unique()).issubset({0, 1}) or y.nunique() != 2:
        raise ValueError(f"treatment_outcome must contain exactly binary values 0 and 1; found {sorted(y.unique())}")
    return frame.drop(columns=["patient_id"], errors="ignore"), y


def get_train_val_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return encoded 60/20/20 partitions with preprocessing fit on training data only."""
    X_raw, y = load_data()
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    encoder = FrequencyCategoryEncoder().fit(X_train_raw)
    return (encoder.transform(X_train_raw), encoder.transform(X_val_raw),
            encoder.transform(X_test_raw), y_train, y_val, y_test)


def get_train_test_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Backward-compatible name for the six-partition data helper."""
    return get_train_val_test_data()


def find_best_threshold(y_true: pd.Series, y_probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(np.clip(thresholds[int(np.nanargmax(scores))], 0.01, 0.99))


def evaluate(y_true: pd.Series, probabilities: np.ndarray, threshold: float, model_name: str) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return {"model_name": model_name, "threshold": float(threshold),
            "roc_auc": float(roc_auc_score(y_true, probabilities)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "feature_names": [], "feature_importances": []}


def fit_evaluate_save(estimator: Any, model_name: str, short_name: str, imbalance_strategy: str) -> dict[str, Any]:
    ensure_output_dirs()
    started = time.perf_counter()
    X_raw, y = load_data()
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_raw, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    encoder = FrequencyCategoryEncoder().fit(X_train_raw)
    X_train, X_val, X_test = (encoder.transform(X_train_raw), encoder.transform(X_val_raw), encoder.transform(X_test_raw))
    estimator.fit(X_train, y_train)
    train_probs = estimator.predict_proba(X_train)[:, 1]
    val_probs = estimator.predict_proba(X_val)[:, 1]
    test_probs = estimator.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_val, val_probs)
    train_predictions = (train_probs >= threshold).astype("int8")
    test_predictions = (test_probs >= threshold).astype("int8")
    train_pr_auc = float(average_precision_score(y_train, train_probs))
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    train_f1 = float(f1_score(y_train, train_predictions, zero_division=0))
    test_f1 = float(f1_score(y_test, test_predictions, zero_division=0))
    train_roc_auc = float(roc_auc_score(y_train, train_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc_gap = train_pr_auc - test_pr_auc
    f1_gap = train_f1 - test_f1
    roc_auc_gap = train_roc_auc - test_roc_auc
    diagnosis = (
        "Well-Balanced" if pr_auc_gap <= 0.03
        else "Moderate Overfit" if pr_auc_gap <= 0.06
        else "Severe Overfit"
    )
    metrics = evaluate(y_test, test_probs, threshold, model_name)
    metrics.update({
        "optimal_threshold": float(threshold),
        "train_pr_auc": train_pr_auc, "test_pr_auc": test_pr_auc,
        "train_f1": train_f1, "test_f1": test_f1,
        "train_roc_auc": train_roc_auc,
        "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
        "test_roc_auc": test_roc_auc,
        "pr_auc_gap": float(pr_auc_gap), "f1_gap": float(f1_gap),
        "roc_auc_gap": float(roc_auc_gap), "overfit_gap_auc": float(roc_auc_gap),
        "overfitting_diagnosis": diagnosis,
        "test_recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_precision": float(precision_score(y_test, test_predictions, zero_division=0)),
    })
    metrics.update({"imbalance_strategy": imbalance_strategy, "execution_seconds": round(time.perf_counter() - started, 3), "feature_names": encoder.feature_names})
    if hasattr(estimator, "feature_importances_"):
        metrics["feature_importances"] = [float(value) for value in estimator.feature_importances_]
    else:
        importance_count = min(5000, len(X_test))
        importance_indices = np.linspace(0, len(X_test) - 1, importance_count, dtype=int)
        permutation = permutation_importance(
            estimator,
            X_test.iloc[importance_indices],
            y_test.iloc[importance_indices],
            scoring="average_precision",
            n_repeats=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        metrics["feature_importances"] = [float(value) for value in permutation.importances_mean]

    bundle = {"preprocessing_state": encoder, "estimator": estimator, "feature_names": encoder.feature_names, "selected_threshold": threshold}
    joblib.dump(bundle, METRICS_DIR / f"{short_name}.pkl")
    (METRICS_DIR / f"{short_name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "=" * 70,
        f"CLINICAL MODEL PERFORMANCE & OVERFITTING REPORT: {model_name}",
        "=" * 70,
        f"Optimal Decision Threshold: {threshold:.6f}",
        "",
        "--- MULTI-METRIC OVERFITTING & GENERALIZATION DIAGNOSIS ---",
        "Metric      | Train    | Test     | Gap (Train-Test) | Status",
        "----------------------------------------------------------------------",
        f"PR-AUC      | {train_pr_auc:.6f} | {test_pr_auc:.6f} | {pr_auc_gap:.6f}         | {diagnosis}",
        f"F1-Score    | {train_f1:.6f} | {test_f1:.6f} | {f1_gap:.6f}         | {diagnosis}",
        f"ROC-AUC     | {train_roc_auc:.6f} | {test_roc_auc:.6f} | {roc_auc_gap:.6f}         | {diagnosis}",
        "",
        "--- FINAL TEST SET METRICS ---",
        f"PR-AUC    : {test_pr_auc:.6f}",
        f"F1-Score  : {test_f1:.6f}",
        f"Recall    : {metrics['test_recall']:.6f}",
        f"Precision : {metrics['test_precision']:.6f}",
        f"ROC-AUC   : {test_roc_auc:.6f}",
        f"Accuracy  : {metrics['test_accuracy']:.6f}",
        "",
        f"Execution Time (seconds): {metrics['execution_seconds']}",
        f"Imbalance Strategy: {imbalance_strategy}",
        "",
        "=" * 70,
        "CONFUSION MATRIX (TN, FP, FN, TP)",
        f"{metrics['tn']}, {metrics['fp']}, {metrics['fn']}, {metrics['tp']}",
        "",
        "CLASSIFICATION REPORT",
        classification_report(y_test, test_predictions, target_names=["Ineffective (0)", "Effective (1)"], zero_division=0),
        "=" * 70,
    ]
    (METRICS_DIR / f"{short_name}_report.txt").write_text("\n".join(report), encoding="utf-8")
    return metrics


def save_model_charts(metrics: dict[str, Any], short_name: str) -> None:
    ensure_output_dirs()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(metrics["fpr"], metrics["tpr"], label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(title=f"{metrics['model_name']} ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(); figure.tight_layout(); figure.savefig(REPORTS_DIR / f"{short_name}_roc_curve.png", dpi=300); plt.close(figure)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    figure, axis = plt.subplots(figsize=(6, 5)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(title=f"{metrics['model_name']} Confusion Matrix", xlabel="Predicted", ylabel="Actual"); figure.tight_layout(); figure.savefig(REPORTS_DIR / f"{short_name}_confusion_matrix.png", dpi=300); plt.close(figure)
