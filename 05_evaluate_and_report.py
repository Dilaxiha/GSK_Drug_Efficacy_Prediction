"""Create high-resolution comparison charts and a publication-ready PDF."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from utils import METRICS_DIR, REPORTS_DIR, ensure_output_dirs

FILES = ["gradient_boosting", "xgboost", "random_forest", "decision_tree", "keras", "lstm"]


def overfitting_status(gap: float) -> str:
    if gap <= 0.03:
        return "Well-Balanced"
    if gap <= 0.06:
        return "Moderate Overfit"
    return "High Overfit"


def main() -> None:
    ensure_output_dirs()
    missing = [
        name for name in FILES
        if not (METRICS_DIR / ("keras_metrics.json" if name == "keras" else "lstm_metrics.json" if name == "lstm" else f"{name}.json")).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing model metrics JSON files: {', '.join(missing)}")
    records = []
    for name in FILES:
        filename = "keras_metrics.json" if name == "keras" else "lstm_metrics.json" if name == "lstm" else f"{name}.json"
        records.append(json.loads((METRICS_DIR / filename).read_text(encoding="utf-8")))
    for record in records:
        record["train_roc_auc"] = float(record.get("train_roc_auc", record.get("roc_auc", record.get("test_roc_auc", 0.0))))
        record["test_roc_auc"] = float(record.get("test_roc_auc", record.get("roc_auc", 0.0)))
        record["overfit_gap_auc"] = float(record.get("overfit_gap_auc", record["train_roc_auc"] - record["test_roc_auc"]))
        record["overfitting_diagnosis"] = overfitting_status(record["overfit_gap_auc"])
    frame = pd.DataFrame(records).sort_values(["pr_auc", "f1_score", "recall"], ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["overfitting_status"] = frame["overfit_gap_auc"].map(overfitting_status)
    frame["overfitting_diagnosis"] = frame.apply(
        lambda row: f"Train={row['train_roc_auc']:.4f} | Test={row['test_roc_auc']:.4f} | Gap={row['overfit_gap_auc']:.4f} | {row['overfitting_status']}", axis=1
    )
    frame.to_json(METRICS_DIR / "model_comparison.json", orient="records", indent=2)
    frame[["rank", "model_name", "pr_auc", "f1_score", "recall", "roc_auc", "accuracy", "precision", "threshold", "overfitting_diagnosis"]].to_csv(METRICS_DIR / "model_comparison.txt", sep="\t", index=False)

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6))
    for record in records:
        axis.plot(record["fpr"], record["tpr"], label=f"{record['model_name']} (AUC={record['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(title="Multi-model ROC Curve Comparison", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(); figure.tight_layout(); roc_path = REPORTS_DIR / "multi_model_roc_comparison.png"; figure.savefig(roc_path, dpi=400); plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 13)); axes = axes.ravel()
    for axis, record in zip(axes, records):
        matrix = np.array([[record["tn"], record["fp"]], [record["fn"], record["tp"]]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=axis, xticklabels=["Ineffective", "Effective"], yticklabels=["Ineffective", "Effective"])
        axis.set_title(f"{record['model_name']} (threshold={record['threshold']:.3f})"); axis.set_xlabel("Predicted"); axis.set_ylabel("Actual")
    figure.suptitle("Confusion Matrices at Optimal Thresholds", fontsize=15); figure.tight_layout(); cm_path = REPORTS_DIR / "confusion_matrix_grid.png"; figure.savefig(cm_path, dpi=400, bbox_inches="tight"); plt.close(figure)

    best = frame.iloc[0].to_dict()
    importance = pd.Series(best["feature_importances"], index=best["feature_names"]).nlargest(10).sort_values()
    figure, axis = plt.subplots(figsize=(9, 6)); importance.plot.barh(ax=axis, color="#287271"); axis.set(title=f"Top 10 Feature Importances: {best['model_name']}", xlabel="Importance"); figure.tight_layout(); importance_path = REPORTS_DIR / "top_10_feature_importances.png"; figure.savefig(importance_path, dpi=400); plt.close(figure)

    styles = getSampleStyleSheet(); pdf_path = REPORTS_DIR / "Clinical_Model_Performance_Report.pdf"
    temporary_pdf_path = REPORTS_DIR / "Clinical_Model_Performance_Report.tmp.pdf"
    document = SimpleDocTemplate(str(temporary_pdf_path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm, title="Clinical Model Performance Report")
    headers = ["Rank", "Model", "PR-AUC", "F1-Score", "Recall", "ROC-AUC", "Accuracy", "Precision", "Threshold"]
    rows = [headers] + [[int(row.rank), row.model_name, f"{row.pr_auc:.4f}", f"{row.f1_score:.4f}", f"{row.recall:.4f}", f"{row.roc_auc:.4f}", f"{row.accuracy:.4f}", f"{row.precision:.4f}", f"{row.threshold:.4f}"] for row in frame.itertuples()]
    table = Table(rows, colWidths=[10 * mm, 32 * mm, 18 * mm, 19 * mm, 18 * mm, 20 * mm, 20 * mm, 20 * mm, 22 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F4")])]))
    diagnosis_rows = [["Model", "Train ROC-AUC", "Test ROC-AUC", "Gap", "Status"]] + [[row.model_name, f"{row.train_roc_auc:.4f}", f"{row.test_roc_auc:.4f}", f"{row.overfit_gap_auc:.4f}", row.overfitting_status] for row in frame.itertuples()]
    diagnosis_table = Table(diagnosis_rows, colWidths=[42 * mm, 29 * mm, 29 * mm, 22 * mm, 40 * mm])
    diagnosis_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C2D12")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF7ED")])]))
    story = [Paragraph("Clinical Model Performance Report", styles["Title"]), Paragraph(f"Executive summary: {best['model_name']} ranked first by PR-AUC, then F1-Score and Recall. The selected threshold was tuned only on an internal validation partition; the final test set remained untouched until evaluation.", styles["BodyText"]), Spacer(1, 8), Paragraph("Performance Benchmark", styles["Heading2"]), table, Spacer(1, 10), Paragraph("Overfitting Analysis", styles["Heading2"]), diagnosis_table, Spacer(1, 10), Image(str(roc_path), width=175 * mm, height=105 * mm), Spacer(1, 8), Image(str(cm_path), width=175 * mm, height=143 * mm), Spacer(1, 8), Image(str(importance_path), width=165 * mm, height=110 * mm)]
    document.build(story)
    try:
        temporary_pdf_path.replace(pdf_path)
    except PermissionError as error:
        fallback_path = REPORTS_DIR / f"Clinical_Model_Performance_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        try:
            temporary_pdf_path.replace(fallback_path)
        except PermissionError as fallback_error:
            raise PermissionError(
                f"Cannot write the report because {pdf_path} is locked and the fallback PDF could not be created. "
                "Close the PDF in any viewer and rerun the report."
            ) from fallback_error
        print(f"Warning: {pdf_path} is locked; new PDF saved to {fallback_path}")
    print(frame[["rank", "model_name", "pr_auc", "f1_score", "recall", "roc_auc", "accuracy", "precision", "threshold", "overfitting_diagnosis"]].to_string(index=False))
    print(f"Best model: {best['model_name']}; PDF: {pdf_path}")


if __name__ == "__main__":
    main()
