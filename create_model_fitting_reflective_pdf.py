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
COMPARISON_PATH = OUTPUT_DIR / "model_comparison_results.csv"
PDF_PATH = OUTPUT_DIR / "model_fitting_reflective_summary.pdf"

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


def build_pdf() -> None:
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            f"Comparison results not found: {COMPARISON_PATH}. Run 04_model_comparison.py first."
        )

    results = pd.read_csv(COMPARISON_PATH)
    required = {"Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Comparison file is missing columns: {', '.join(sorted(missing))}")

    results = results.sort_values(["ROC-AUC", "F1-Score"], ascending=[False, False]).reset_index(
        drop=True
    )
    best = results.iloc[0]

    # Build Header Row with White Text
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    metric_rows = [[paragraph(h, "TableHeader") for h in headers]]

    # Build Data Rows
    for _, row in results.iterrows():
        row_data = [paragraph(str(row["Model"]), "TableCell")]
        row_data.extend([
            paragraph(f"{row[column]:.4f}", "TableCell")
            for column in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
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
        paragraph("Reflective summary of Steps 4-7 | Clinical treatment outcome classification", "Subtitle"),
        
        paragraph("Executive Reflection", "Section"),
        paragraph("The model-fitting stage translated the cleaned and feature-engineered clinical data into comparable predictive evidence. I used the same stratified 80/20 split and evaluated every model on the original test set. This made the comparison fair and kept the test set representative of the real class distribution."),
        
        # Summary KPI Box Table
        Table(
            [
                [paragraph("Models compared", "SmallCustom"), paragraph("Selection priority", "SmallCustom"), paragraph("Best current model", "SmallCustom")],
                [paragraph(str(len(results)), "Callout"), paragraph("ROC-AUC, then F1-Score", "Callout"), paragraph(str(best["Model"]), "Callout")]
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
        paragraph("I applied SMOTE only after the train-test split and only to the training data. Because the engineered dataset contains categorical variables and missing values, numeric median imputation and categorical most-frequent imputation were fitted on the training partition. Categorical values were then ordinal-encoded, and the untouched test partition was transformed without resampling. This sequence reduces the risk of target leakage."),
        paragraph("The four fitted models were intentionally compared using the same held-out test data: Decision Tree, Random Forest, Gradient Boosting, and XGBoost. The tree-based models were configured with constrained depth and minimum sample requirements where specified, balancing predictive flexibility against overfitting risk."),
        
        # Keep Table and Section Header Together to Avoid Orphan Headings
        KeepTogether([
            paragraph("Results", "Section"),
            Table(
                metric_rows,
                colWidths=[44 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm],
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
        paragraph(f"The current comparison ranks <b>{best['Model']}</b> first with ROC-AUC {best['ROC-AUC']:.4f} and F1-Score {best['F1-Score']:.4f}. ROC-AUC was used as the primary ranking metric because it assesses discrimination across classification thresholds, while F1-Score provides a useful check on the balance between precision and recall for the minority outcome class."),
        paragraph("This result is a model-selection signal rather than evidence that the model is clinically ready. A strong score on one fixed split can still be affected by sampling variation, feature leakage, preprocessing assumptions, or changes in the population. The model should therefore be stress-tested with repeated cross-validation, calibration analysis, threshold analysis, and an external or temporal validation set before deployment."),
        
        paragraph("Reflection on Limitations", "Section"),
        paragraph("SMOTE changes the training distribution but does not create new clinical information. Ordinal encoding is convenient for these tree models, but its numerical ordering is not inherently meaningful. Imputation can also make records appear more complete than the underlying data really are. These choices were appropriate for a consistent baseline comparison, but they should be documented and revisited during model review."),
        
        paragraph("Next Steps", "Section"),
        paragraph("The next stage should confirm that the selected model remains stable across folds and clinically meaningful subgroups. I would also inspect calibration, precision-recall trade-offs, false-negative cases, feature importance stability, and the effect of alternate decision thresholds. Only after those checks should a final model and operating threshold be considered for a controlled evaluation setting."),
        
        Spacer(1, 3 * mm),
        paragraph(f"Source: {COMPARISON_PATH.name}. Model artifacts and individual metric files are stored in the outputs directory.", "SmallCustom"),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(f"Created: {PDF_PATH}")