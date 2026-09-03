from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "Day1_Data_Preparation_Reflective_Summary.pdf"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#287271")
RED = colors.HexColor("#B23A48")
PALE_BLUE = colors.HexColor("#EAF1F5")
PALE_TEAL = colors.HexColor("#E8F3F1")
TEXT = colors.HexColor("#263238")
MUTED = colors.HexColor("#5F6B72")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=24,
    leading=29,
    textColor=NAVY,
    alignment=TA_CENTER,
    spaceAfter=5 * mm,
))
styles.add(ParagraphStyle(
    name="Subtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10.5,
    leading=15,
    textColor=MUTED,
    alignment=TA_CENTER,
    spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=14,
    leading=18,
    textColor=NAVY,
    spaceBefore=5 * mm,
    spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    name="BodyTextCustom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=TEXT,
    spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    name="Small",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
    textColor=MUTED,
))
styles.add(ParagraphStyle(
    name="Callout",
    parent=styles["BodyText"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=14,
    textColor=NAVY,
    leftIndent=3 * mm,
    rightIndent=3 * mm,
    spaceBefore=2 * mm,
    spaceAfter=2 * mm,
))


def P(text: str, style: str = "BodyTextCustom") -> Paragraph:
    return Paragraph(text, styles[style])


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7E0E5"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "Day 1 Data Preparation Summary")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def build_pdf():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=19 * mm,
        title="Day 1 Data Preparation and Feature Engineering Summary",
        author="BCP Clinical Data Project",
    )

    story = [
        P("Day 1 Data Preparation & Feature Engineering", "ReportTitle"),
        P("End-of-day reflective summary | 2 September 2026", "Subtitle"),
        HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=6 * mm),
        P("Executive Reflection", "Section"),
        P(
            "Today I focused on making the clinical dataset trustworthy before any modelling work begins. "
            "The main lesson was that data preparation is not a mechanical pre-step: duplicated fields, inconsistent labels, "
            "missing values, implausible measurements, and target imbalance can all change the conclusions drawn from the data. "
            "By auditing the raw file first and validating each later transformation, I created a clearer and more defensible foundation for the next stage.",
        ),
        Table(
            [[P("Raw data", "Small"), P("Cleaned data", "Small"), P("Engineered data", "Small")],
             [P("1,155,000 rows<br/>41 columns", "Callout"), P("1,015,405 rows<br/>32 columns", "Callout"), P("1,015,405 rows<br/>39 columns", "Callout")]],
            colWidths=[58 * mm, 58 * mm, 58 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BACKGROUND", (0, 1), (-1, 1), PALE_TEAL),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C7D6DE")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E1E6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]),
        ),
        P("Work Completed", "Section"),
        P("<b>1. Raw-data audit.</b> I examined the complete source file rather than relying on a sample. The audit identified 70,017 exact duplicate rows, substantial missingness in several fields, duplicate or whitespace-variant columns, placeholder values, and mixed representations of the treatment outcome.",),
        P("<b>2. Cleaning and standardisation.</b> I consolidated duplicate schema variants, standardised column names and categorical labels, converted numeric fields, removed irrelevant columns, handled implausible values through IQR-based capping, imputed numeric and categorical gaps, removed duplicates, and retained only valid binary target records.",),
        P("<b>3. Verification and exploratory analysis.</b> I checked the cleaned dataset for missing cells, valid target values, near-zero-variance variables, quasi-complete separation, outliers, correlations, and multicollinearity. I also prepared focused charts for target balance, numeric distributions, target association, biomarker differences, and feature redundancy.",),
        P("<b>4. Feature engineering.</b> I created seven clinically motivated features: <i>kidney_stage</i>, <i>liver_risk</i>, <i>polypharmacy</i>, <i>bmi_category</i>, <i>age_group</i>, <i>elderly_high_dose</i>, and <i>de_ritis_ratio</i>. The final engineered dataset contains 39 columns.",),
        P("Key Findings", "Section"),
        Table(
            [[P("Area", "Small"), P("Finding", "Small"), P("Why it matters", "Small")],
             [P("Target", "Small"), P("788,599 failures (77.66%) and 226,806 successes (22.34%).", "Small"), P("The target is imbalanced and should be considered during modelling and evaluation.", "Small")],
             [P("Data quality", "Small"), P("The cleaned dataset passed with zero missing cells, binary target values, and zero duplicate rows remaining.", "Small"), P("The modelling input is structurally consistent and reproducible.", "Small")],
             [P("Predictive signal", "Small"), P("Age had the strongest baseline association with outcome (r = -0.179), followed by creatinine (r = -0.086).", "Small"), P("These variables deserve careful clinical and statistical review, but correlation alone does not establish causation.", "Small")],
             [P("Engineered output", "Small"), P("Seven new features were added; 2,461 missing values remain in the engineered output, associated with the ratio feature.", "Small"), P("The derived ratio needs explicit missing-value handling before it is used in a model.", "Small")]],
            colWidths=[28 * mm, 78 * mm, 68 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CDD9DF")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),
        P("Reflection on Decisions and Limitations", "Section"),
        P(
            "The most important decision was to preserve an auditable sequence: inspect first, clean second, verify third, and engineer features only after the base dataset was stable. This reduced the risk of hiding data problems inside transformations. I also learned that statistical significance is not the same as practical importance: with more than one million observations, very small correlations can appear significant, so I will prioritise effect size, clinical plausibility, and out-of-sample performance.",
        ),
        P(
            "There are still limitations. Median and mode imputation can reduce natural variability, IQR capping can affect extreme clinical cases, and the engineered de Ritis ratio has missing values when its source measurements are unavailable. The current findings are therefore appropriate for preparation and hypothesis generation, not for clinical interpretation or deployment.",
        ),
        P("Next-Day Focus", "Section"),
        P("I will next review the engineered features for leakage and clinical validity, define a modelling-ready feature matrix, document the train-validation-test split, and establish evaluation metrics that account for the imbalanced outcome. I will also test whether the observed age and creatinine relationships remain stable after appropriate validation.",),
        Spacer(1, 4 * mm),
        P("Supporting project outputs: Step1_Initial_EDA_Report.md, Step2_Cleaning_Report.md, Step3_Verification_Report.md, feature_engineering_report.txt, and the cleaned and engineered CSV datasets.", "Small"),
    ]
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(f"Created: {OUTPUT}")
