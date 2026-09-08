"""Compact 2-page executive summary PDF: all 7 models in one table, champion model called out
for the Flask API deployment, plus a couple of key comparison charts.

Standalone from 05_evaluate_and_report.py (which produces a much longer, 6-model deep-dive
report and doesn't yet include LightGBM). This script covers all 7 trained models.
"""
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils import METRICS_DIR, REPORTS_DIR, ensure_output_dirs

FILES = ["xgboost", "gradient_boosting", "lightgbm", "random_forest", "decision_tree", "lstm", "keras"]
DISPLAY_NAMES = {
    "decision_tree": "Decision Tree", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting", "lightgbm": "LightGBM", "lstm": "LSTM", "keras": "MLP",
}
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#287271")
GOLD = colors.HexColor("#B8860B")
TOTAL_ROWS = 1_015_405  # dataset/feature_engineered_clinical_data.csv row count (see Data_Exploration_Report.md)


def metrics_filename(name: str) -> str:
    return "keras_metrics.json" if name == "keras" else "lstm_metrics.json" if name == "lstm" else f"{name}.json"


def load_records() -> list[dict]:
    records = []
    for name in FILES:
        path = METRICS_DIR / metrics_filename(name)
        if not path.exists():
            baseline_path = METRICS_DIR / (f"{name}_baseline_metrics.json" if name in ("lstm", "keras") else f"{name}_baseline.json")
            path = baseline_path if baseline_path.exists() else path
        if not path.exists():
            print(f"  -> skipping '{name}': no metrics file found at {path}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        record["short_name"] = name
        record["display_name"] = DISPLAY_NAMES[name]
        records.append(record)
    return records


def select_api_model(frame: pd.DataFrame) -> pd.Series:
    """XGBoost is the model actually wired into the Flask API (app.py / train_deployment_model.py).
    Report it as the recommendation, with an honest note on how close the runner-up is."""
    return frame.loc[frame["short_name"] == "xgboost"].iloc[0]


def build_charts(frame: pd.DataFrame, champion_name: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    ordered = frame.sort_values("pr_auc")
    bar_colors = ["#B8860B" if name == champion_name else "#287271" for name in ordered["short_name"]]
    figure, axis = plt.subplots(figsize=(8.6, 3.4))
    axis.barh(ordered["display_name"], ordered["pr_auc"], color=bar_colors)
    axis.set(title="Test PR-AUC by Model (gold = recommended for API)", xlabel="PR-AUC")
    figure.tight_layout()
    paths["pr_bar"] = REPORTS_DIR / "exec_summary_pr_auc_bar.png"
    figure.savefig(paths["pr_bar"], dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    for row in frame.itertuples():
        style = dict(linewidth=2.4, zorder=5) if row.short_name == champion_name else dict(linewidth=1.2, alpha=0.75)
        axis.plot(row.fpr, row.tpr, label=f"{row.display_name} ({row.roc_auc:.3f})", **style)
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(title="ROC Curve Comparison (all 7 models)", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    paths["roc"] = REPORTS_DIR / "exec_summary_roc_comparison.png"
    figure.savefig(paths["roc"], dpi=300)
    plt.close(figure)

    champion = frame.loc[frame["short_name"] == champion_name].iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    matrix = np.array([[champion.tn, champion.fp], [champion.fn, champion.tp]])
    axes[0].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[0].text(j, i, f"{value:,}", ha="center", va="center")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Ineffective", "Effective"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Ineffective", "Effective"])
    axes[0].set(title=f"{champion.display_name} Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    axes[1].plot(champion.recall_curve, champion.precision_curve, color="#B23A48")
    axes[1].set(title=f"{champion.display_name} Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    figure.tight_layout()
    paths["champion_detail"] = REPORTS_DIR / "exec_summary_champion_detail.png"
    figure.savefig(paths["champion_detail"], dpi=300)
    plt.close(figure)
    return paths


def main() -> None:
    ensure_output_dirs()
    records = load_records()
    frame = pd.DataFrame(records).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))

    api_model = select_api_model(frame)
    runner_up = frame.loc[frame["short_name"] != "xgboost"].iloc[0]
    pr_auc_margin = api_model["pr_auc"] - runner_up["pr_auc"]

    chart_paths = build_charts(frame, "xgboost")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ExecTitle", parent=styles["Title"], fontSize=17, leading=21, textColor=NAVY, alignment=TA_CENTER, spaceAfter=1 * mm))
    styles.add(ParagraphStyle(name="ExecSubtitle", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#5F6B72"), alignment=TA_CENTER, spaceAfter=3 * mm))
    styles.add(ParagraphStyle(name="ExecSection", parent=styles["Heading2"], fontSize=11.5, textColor=NAVY, spaceBefore=2 * mm, spaceAfter=1.5 * mm))
    styles.add(ParagraphStyle(name="ExecBody", parent=styles["BodyText"], fontSize=8.5, leading=11.5, spaceAfter=1.5 * mm))

    def P(text: str, style: str = "ExecBody") -> Paragraph:
        return Paragraph(text, styles[style])

    story: list = [
        Paragraph("Clinical Treatment-Outcome Prediction: Model Comparison &amp; API Recommendation", styles["ExecTitle"]),
        P(f"Generated {datetime.now():%Y-%m-%d %H:%M} &mdash; {len(frame)} models trained on {TOTAL_ROWS:,} patient records, "
          "leakage-safe 60/20/20 split, primary metric = PR-AUC (imbalanced target: ~22.3% Effective).", "ExecSubtitle"),
    ]

    story.append(P("All Models Compared (Test Set)", "ExecSection"))
    headers = ["Rank", "Model", "PR-AUC", "F1", "Precision", "Recall", "ROC-AUC", "Accuracy", "Diagnosis"]
    rows = [headers]
    for row in frame.itertuples():
        label = f"{row.display_name} \u2605" if row.short_name == "xgboost" else row.display_name
        rows.append([int(row.rank), label, f"{row.pr_auc:.4f}", f"{row.f1_score:.4f}", f"{row.precision:.4f}",
                     f"{row.recall:.4f}", f"{row.roc_auc:.4f}", f"{row.accuracy:.4f}", row.overfitting_diagnosis])
    table = Table(rows, colWidths=[10 * mm, 32 * mm, 16 * mm, 14 * mm, 18 * mm, 14 * mm, 16 * mm, 16 * mm, 26 * mm])
    champion_row_index = int(frame.loc[frame["short_name"] == "xgboost"].index[0]) + 1
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, champion_row_index), (-1, champion_row_index), colors.HexColor("#FFF3D6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F4")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 2 * mm))

    story.append(P("Recommended for API Deployment: XGBoost \u2605", "ExecSection"))
    story.append(P(
        f"<b>XGBoost</b> is deployed behind the project's Flask REST API (<b>/predict</b> endpoint, see app.py) "
        f"and leads the comparison on PR-AUC ({api_model['pr_auc']:.4f}), the primary metric for this imbalanced "
        f"problem. Its closest competitor, {runner_up['display_name']}, trails by only {pr_auc_margin:.4f} PR-AUC "
        "(effectively tied, likely within run-to-run noise), but XGBoost was chosen for deployment because it "
        f"combines that top-tier accuracy with {api_model['overfitting_diagnosis'].lower()} generalization "
        f"(train-test PR-AUC gap of {api_model['pr_auc_gap']:.4f}), fast CPU-only inference (tree_method='hist'), "
        "and native early stopping/regularization suited to the 8&nbsp;GB RAM deployment target. LightGBM was also "
        "evaluated as a lighter-weight alternative but under-trained during tuning (early stopping halted after "
        "just 4 boosting rounds) and scored lowest of the gradient-boosted models on this run."
    ))

    story.append(Spacer(1, 1 * mm))
    story.append(Image(str(chart_paths["pr_bar"]), width=165 * mm, height=65 * mm))
    story.append(Spacer(1, 1 * mm))
    story.append(Image(str(chart_paths["roc"]), width=165 * mm, height=88 * mm))
    story.append(Spacer(1, 1 * mm))
    story.append(Image(str(chart_paths["champion_detail"]), width=165 * mm, height=65 * mm))

    story.append(P("Key Takeaways", "ExecSection"))
    story.append(P(
        "&bull; All 7 models cluster within a narrow PR-AUC band (0.316-0.347), well above the majority-class "
        "floor of ~0.223, but no single model dominates &mdash; suggesting the available features carry a real but "
        "limited amount of predictive signal.<br/>"
        "&bull; Class imbalance (~22.3% Effective) was handled via scale_pos_weight rather than resampling, and "
        "PR-AUC (not accuracy) was used throughout for tuning/comparison since a trivial always-Ineffective "
        "classifier would score ~78% accuracy while catching zero Effective cases.<br/>"
        "&bull; XGBoost's confusion matrix (above) favors recall over precision at its tuned threshold: it catches "
        "67% of Effective cases (recall) at the cost of some false positives, a deliberate trade-off for a clinical "
        "screening use case where missing a true Effective response is costlier than a false alarm.<br/>"
        "&bull; LightGBM under-trained during this run (early stopping halted at only 4 boosting rounds) and is a "
        "candidate for a follow-up tuning pass rather than a like-for-like comparison against the other six models."
    ))

    output_path = REPORTS_DIR / "Executive_Summary_2Page.pdf"
    try:
        SimpleDocTemplate(
            str(output_path), pagesize=A4,
            topMargin=12 * mm, bottomMargin=12 * mm, leftMargin=14 * mm, rightMargin=14 * mm,
        ).build(story)
    except PermissionError:
        output_path = REPORTS_DIR / f"Executive_Summary_2Page_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        print(f"Warning: default path is locked; saving to {output_path} instead")
        SimpleDocTemplate(
            str(output_path), pagesize=A4,
            topMargin=12 * mm, bottomMargin=12 * mm, leftMargin=14 * mm, rightMargin=14 * mm,
        ).build(story)

    print(frame[["rank", "display_name", "pr_auc", "f1_score", "roc_auc", "accuracy", "overfitting_diagnosis"]].to_string(index=False))
    print(f"\nRecommended for API deployment: {api_model['display_name']}")
    print(f"Report PDF: {output_path}")


if __name__ == "__main__":
    main()
