"""Shared utilities for leakage-safe imbalanced classification experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "feature_engineered_clinical_data.csv"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "treatment_outcome"
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series, Path]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    frame = frame.dropna(subset=[TARGET]).copy()
    y = frame.pop(TARGET).astype(int)
    X = frame.drop(columns=["patient_id"], errors="ignore")
    if y.nunique() != 2:
        raise ValueError(f"Target must contain exactly two classes; found {sorted(y.unique())}")
    return X, y, DATA_PATH


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=100, sparse_output=True))]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def make_pipeline(X: pd.DataFrame, estimator: Any, use_smote: bool = True) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocess", make_preprocessor(X))]
    if use_smote:
        steps.append(("smote", SMOTE(sampling_strategy=0.75, random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def make_gradient_boosting_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    """Use ordinal categories because sklearn GradientBoosting needs dense input."""
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical))
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0)
    return Pipeline(steps=[("preprocess", preprocessor), ("smote", SMOTE(sampling_strategy=0.75, random_state=RANDOM_STATE)), ("model", estimator)])


def find_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.linspace(0.10, 0.50, 161)
    scores = [(f1_score(y_true, probabilities >= threshold, zero_division=0), threshold) for threshold in thresholds]
    return float(max(scores)[1])


def fit_with_threshold(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    X_fit, X_validation, y_fit, y_validation = train_test_split(X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE)
    pipeline.fit(X_fit, y_fit)
    validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
    threshold = find_threshold(y_validation, validation_probabilities)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return pipeline, threshold, predictions, probabilities


def evaluate_predictions(model_name: str, threshold: float, y_test: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced-Accuracy": balanced_accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def print_evaluation(metrics: dict[str, Any], y_test: pd.Series, predictions: np.ndarray) -> None:
    print("\n--- Evaluation ---")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nConfusion Matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Ineffective", "Effective"], zero_division=0))


def save_model_outputs(prefix: str, model: Pipeline, metrics: dict[str, Any], feature_names: list[str] | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / f"{prefix}_metrics.csv", index=False)
    joblib.dump(model, OUTPUT_DIR / f"{prefix}_model.pkl")
    estimator = model.named_steps["model"]
    if feature_names is not None and hasattr(estimator, "feature_importances_"):
        try:
            transformed_names = model.named_steps["preprocess"].get_feature_names_out()
        except (AttributeError, ValueError):
            transformed_names = np.asarray(feature_names, dtype=str)
        importance = pd.DataFrame({"Feature": transformed_names, "Importance": estimator.feature_importances_}).sort_values("Importance", ascending=False)
        importance.to_csv(OUTPUT_DIR / f"{prefix}_feature_importance.csv", index=False)
        top = importance.head(15).sort_values("Importance")
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.barplot(data=top, x="Importance", y="Feature", color="#287271", ax=axis)
        axis.set_title(f"{prefix.upper()} Top 15 Feature Importances")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"{prefix}_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def save_probability_charts(prefix: str, y_test: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> None:
    save_confusion_chart(prefix, y_test, predictions)
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, color="#287271", label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.4f}")
    axis.plot([0, 1], [0, 1], "--", color="#888888")
    axis.set(title=f"{prefix.upper()} ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{prefix}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_results_report(
    prefix: str,
    input_path: Path,
    metrics: dict[str, Any],
    y_test: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    best_params: dict[str, Any] | None = None,
) -> None:
    """Save a human-readable evaluation report for one fitted model."""
    report = [
        f"{prefix.upper()} MODEL RESULTS",
        "=" * (len(prefix) + 14),
        f"Input dataset: {input_path}",
        "",
    ]
    if best_params is not None:
        report.extend(["BEST PARAMETERS", "---------------", str(best_params), ""])
    report.extend(f"{key}: {value}" for key, value in metrics.items())
    report.extend([
        "",
        "CONFUSION MATRIX [TN FP; FN TP]",
        str(confusion_matrix(y_test, predictions)),
        "",
        "CLASSIFICATION REPORT",
        classification_report(y_test, predictions, target_names=["Ineffective", "Effective"], zero_division=0),
    ])
    (OUTPUT_DIR / f"{prefix}_results.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, color="#B23A48", label=f"PR-AUC = {average_precision_score(y_test, probabilities):.4f}")
    axis.set(title=f"{prefix.upper()} Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{prefix}_precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_confusion_chart(prefix: str, y_test: pd.Series, predictions: np.ndarray) -> None:
    matrix = confusion_matrix(y_test, predictions)
    labels = np.array([[f"{value}\n({value / matrix.sum():.1%})" for value in row] for row in matrix])
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues", cbar=False, square=True, xticklabels=["Ineffective", "Effective"], yticklabels=["Ineffective", "Effective"], ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"{prefix.upper()} Confusion Matrix")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"{prefix}_confusion_matrix.png", dpi=300)
    plt.close(figure)


def write_reflection_pdf(results: pd.DataFrame, pdf_path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=21, leading=25, textColor=colors.HexColor("#17324D"), alignment=1, spaceAfter=5 * mm))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#17324D"), spaceBefore=4 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9, leading=13, textColor=colors.HexColor("#263238"), spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.5, leading=9, textColor=colors.HexColor("#263238")))

    def P(text: str, style: str = "BodyCustom") -> Paragraph:
        return Paragraph(text, styles[style])

    best = results.iloc[0]
    rows = [[P(column, "Small") for column in ["Model", "PR-AUC", "F1-Score", "Recall", "ROC-AUC"]]]
    for _, row in results.iterrows():
        rows.append([P(str(row[column]), "Small") if column == "Model" else P(f"{row[column]:.4f}", "Small") for column in ["Model", "PR-AUC", "F1-Score", "Recall", "ROC-AUC"]])

    story = [P("Healthcare Treatment Outcome Model Summary", "ReportTitle"), P("Leakage-safe imbalance handling and model comparison", "BodyCustom"), P("Executive Summary", "Section"), P("The target is imbalanced, with class 1 representing the minority effective outcome. Accuracy is therefore treated as a secondary measure. The workflow uses stratified splitting, train-fold-only SMOTE inside imblearn pipelines, native model weighting where supported, and threshold tuning on an inner training validation split."), P("Leaderboard", "Section"), Table(rows, colWidths=[55 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CDD9DF")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]), ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])), P("Best Model Selection", "Section"), P(f"<b>{best['Model']}</b> ranks first by PR-AUC, the primary metric for this imbalanced problem. Its test-set PR-AUC is {best['PR-AUC']:.4f}, F1-Score is {best['F1-Score']:.4f}, recall is {best['Recall']:.4f}, and ROC-AUC is {best['ROC-AUC']:.4f}. At threshold {best['Threshold']:.3f}, the confusion-matrix counts are TN={int(best['TN'])}, FP={int(best['FP'])}, FN={int(best['FN'])}, and TP={int(best['TP'])}."), P("Reflection and Limitations", "Section"), P("SMOTE improves representation of the minority class during fitting, but synthetic observations are not new clinical evidence. Model-level weights and resampling can produce different operating points, so threshold choice must be tied to the cost of false negatives and false positives. The reported test results are appropriate for comparison, not clinical deployment. Further work should include calibration, repeated cross-validation, subgroup analysis, and external or temporal validation."), P("Artifacts", "Section"), P(f"Metrics, fitted pipelines, and supporting charts are stored in {OUTPUT_DIR}.", "Small")]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm, title="Model Summary Report").build(story)
