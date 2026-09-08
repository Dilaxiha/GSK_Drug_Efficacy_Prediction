"""Create a reflective PDF report for the model-fitting stage."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = ROOT / "metrics_output"
PDF_PATH = OUTPUT_DIR / "model_fitting_reflective_summary.pdf"

# Real per-model metrics files produced by 0X_fit_*.py, not a "04_model_comparison.py" CSV export
# (that script doesn't exist in this project).
MODEL_FILES = {
    "Decision Tree": ("decision_tree.json", "decision_tree_baseline.json"),
    "Random Forest": ("random_forest.json", "random_forest_baseline.json"),
    "XGBoost": ("xgboost.json", "xgboost_baseline.json"),
    "Gradient Boosting": ("gradient_boosting.json", "gradient_boosting_baseline.json"),
    "LightGBM": ("lightgbm.json", "lightgbm_baseline.json"),
    # LSTM/MLP: if the tuned search never beat the baseline, the final-model file is
    # intentionally never written (it would be an exact duplicate) -- fall back to baseline.
    "LSTM": ("lstm_metrics.json", "lstm_baseline_metrics.json"),
    "MLP": ("keras_metrics.json", "keras_baseline_metrics.json"),
}

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#287271")
PALE_BLUE = colors.HexColor("#EAF1F5")
PALE_TEAL = colors.HexColor("#E8F3F1")
TEXT = colors.HexColor("#263238")
MUTED = colors.HexColor("#5F6B72")

styles = getSampleStyleSheet()

# Custom Typography Styles
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=NAVY, alignment=TA_CENTER, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED, alignment=TA_CENTER, spaceAfter=5 * mm))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=NAVY, spaceBefore=4 * mm, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.0, leading=12.5, textColor=TEXT, spaceAfter=2 * mm))
styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.8, leading=10.5, textColor=MUTED))
styles.add(ParagraphStyle(name="TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.0, leading=10.5, textColor=colors.white, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.0, leading=10.5, textColor=TEXT, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=9.5, leading=13, textColor=NAVY, alignment=TA_CENTER))


def paragraph(text: str, style: str = "BodyCustom") -> Paragraph:
    return Paragraph(text, styles[style])


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E5"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "Model Fitting Reflective Summary")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def load_results() -> pd.DataFrame:
    rows = []
    for model_name, (filename, fallback_filename) in MODEL_FILES.items():
        path = METRICS_DIR / filename
        if not path.exists():
            path = METRICS_DIR / fallback_filename
        if not path.exists():
            continue
        metrics = pd.read_json(path, typ="series")
        rows.append({
            "Model": model_name, "Accuracy": metrics["accuracy"], "Precision": metrics["precision"],
            "Recall": metrics["recall"], "F1-Score": metrics["f1_score"], "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
        })
    if not rows:
        raise FileNotFoundError(
            f"No model metrics JSON files found in {METRICS_DIR}. Run the 0X_fit_*.py scripts first."
        )
    return pd.DataFrame(rows)


def build_pdf() -> None:
    results = load_results()
    # PR-AUC (not ROC-AUC) is this project's primary ranking metric: the target is imbalanced
    # (~22% positive class), and PR-AUC is far more sensitive to minority-class performance.
    results = results.sort_values(["PR-AUC", "F1-Score"], ascending=[False, False]).reset_index(drop=True)
    best = results.iloc[0]

    # Build Header Row with White Text
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
    metric_rows = [[paragraph(h, "TableHeader") for h in headers]]

    # Build Data Rows
    for _, row in results.iterrows():
        row_data = [paragraph(str(row["Model"]), "TableCell")]
        row_data.extend([
            paragraph(f"{row[column]:.4f}", "TableCell")
            for column in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
        ])
        metric_rows.append(row_data)

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=19 * mm,
        title="Model Fitting Reflective Summary",
        author="BCP Clinical Data Project",
    )

    story = [
        paragraph("Model Fitting & Model Comparison", "ReportTitle"),
        paragraph("Reflective summary of the model-fitting stage | Clinical treatment outcome classification", "Subtitle"),
        
        paragraph("Executive Reflection", "Section"),
        paragraph("The model-fitting stage translated the cleaned and feature-engineered clinical data into comparable predictive evidence. I used the same leakage-safe stratified 60/20/20 train/validation/test split for every model, tuning and selecting hyperparameters from training/validation data only and touching the test set exactly once per model. This made the comparison fair and kept the test set representative of the real class distribution."),
        
        # Summary KPI Box Table
        Table(
            [
                [paragraph("Models compared", "SmallCustom"), paragraph("Selection priority", "SmallCustom"), paragraph("Best current model", "SmallCustom")],
                [paragraph(str(len(results)), "Callout"), paragraph("PR-AUC, then F1-Score", "Callout"), paragraph(str(best["Model"]), "Callout")]
            ],
            colWidths=[58 * mm, 58 * mm, 58 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
                ("BACKGROUND", (0, 1), (-1, 1), PALE_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C7D6DE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E1E6")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        ),
        
        paragraph("Modelling Decisions", "Section"),
        paragraph("Class imbalance (~22% Effective) was handled via scale_pos_weight (tree-based models) or class_weight='balanced' rather than resampling (no SMOTE), so no synthetic rows were introduced. Because the engineered dataset contains categorical variables, a train-only-fitted frequency-rank encoder converted them to numeric codes (median imputation for numeric columns); the untouched validation/test partitions were transformed with those same fitted statistics. This sequence reduces the risk of target leakage."),
        paragraph("Seven models were intentionally compared using the same held-out test data: Decision Tree, Random Forest, XGBoost, Gradient Boosting, LightGBM, LSTM, and an MLP. The tree-based models were configured with constrained depth and minimum sample requirements where specified, balancing predictive flexibility against overfitting risk."),
        
        # Keep Table and Section Header Together to Avoid Orphan Headings
        KeepTogether([
            paragraph("Results", "Section"),
            Table(
                metric_rows,
                colWidths=[38 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CDD9DF")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ])
            )
        ]),
        
        paragraph("Interpretation", "Section"),
        paragraph(f"The current comparison ranks <b>{best['Model']}</b> first with PR-AUC {best['PR-AUC']:.4f} and F1-Score {best['F1-Score']:.4f}. PR-AUC (average precision) was used as the primary ranking metric because, unlike ROC-AUC or accuracy, it is far more sensitive to performance on the minority (Effective) class in an imbalanced dataset; F1-Score provides a useful secondary check on the precision/recall balance."),
        paragraph("This result is a model-selection signal rather than evidence that the model is clinically ready. A strong score on one fixed split can still be affected by sampling variation, feature leakage, preprocessing assumptions, or changes in the population. The model should therefore be stress-tested with repeated cross-validation, calibration analysis, threshold analysis, and an external or temporal validation set before deployment."),
        
        paragraph("Reflection on Limitations", "Section"),
        paragraph("Frequency-rank encoding is convenient for these tree/boosting models, but its numerical ordering is not inherently meaningful and neural nets (LSTM/MLP) additionally need feature scaling to train stably. Imputation can also make records appear more complete than the underlying data really are. These choices were appropriate for a consistent baseline comparison, but they should be documented and revisited during model review."),
        
        paragraph("Next Steps", "Section"),
        paragraph("The next stage should confirm that the selected model remains stable across folds and clinically meaningful subgroups. I would also inspect calibration, precision-recall trade-offs, false-negative cases, feature importance stability, and the effect of alternate decision thresholds. Only after those checks should a final model and operating threshold be considered for a controlled evaluation setting."),
        
        Spacer(1, 3 * mm),
        paragraph(f"Source: per-model metrics JSON files in {METRICS_DIR.name}/. Model artifacts (.pkl/.h5) are stored alongside them.", "SmallCustom"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(f"Created: {PDF_PATH}")