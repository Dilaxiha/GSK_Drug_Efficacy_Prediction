"""Step 5: final evaluation, cumulative model comparison, and the reflective project report.

Loads every model's final metrics JSON (+ its baseline-vs-tuned comparison JSON where present),
builds the cumulative leaderboard artifacts, generates comparison charts, and assembles a
professional multi-section PDF report covering methodology, per-model results, generalization,
champion selection, limitations, and recommendations.
"""
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
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from utils import METRICS_DIR, REPORTS_DIR, ensure_output_dirs

FILES = ["decision_tree", "random_forest", "xgboost", "gradient_boosting", "lstm", "keras"]
DISPLAY_NAMES = {
    "decision_tree": "Decision Tree", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting", "lstm": "LSTM", "keras": "MLP",
}

# Fixed, documented dataset facts (output/Data_Exploration_Report.md) -- avoids reloading the
# 250MB+ raw CSV just to recompute the same class counts every time this report is generated.
TOTAL_ROWS = 1_015_405
CLASS_0_COUNT = 788_599
CLASS_1_COUNT = 226_806
MAJORITY_ACCURACY = CLASS_0_COUNT / TOTAL_ROWS
MAJORITY_PR_AUC = CLASS_1_COUNT / TOTAL_ROWS


def overfitting_status(gap: float) -> str:
    if gap <= 0.03:
        return "Well-Balanced"
    if gap <= 0.06:
        return "Moderate Overfit"
    return "Severe Overfit"


def metrics_filename(name: str) -> str:
    return "keras_metrics.json" if name == "keras" else "lstm_metrics.json" if name == "lstm" else f"{name}.json"


def load_records() -> list[dict]:
    records = []
    for name in FILES:
        metrics_path = METRICS_DIR / metrics_filename(name)
        comparison_path = METRICS_DIR / f"{name}_comparison.json"
        comparison = json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else None
        if not metrics_path.exists():
            # When the tuned search never beats the baseline (e.g. LSTM), the final-model file is
            # intentionally never written (it would be an exact duplicate) -- fall back to the
            # baseline metrics file in that specific case rather than treating it as missing data.
            baseline_path = METRICS_DIR / f"{name}_baseline_metrics.json" if name in ("lstm", "keras") else METRICS_DIR / f"{name}_baseline.json"
            if comparison and comparison.get("selected_configuration") == "Baseline" and baseline_path.exists():
                metrics_path = baseline_path
            else:
                raise FileNotFoundError(f"Missing model metrics JSON file for '{name}': {metrics_path}")
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        record["short_name"] = name
        record["display_name"] = DISPLAY_NAMES[name]
        record["train_roc_auc"] = float(record.get("train_roc_auc", record.get("roc_auc", 0.0)))
        record["test_roc_auc"] = float(record.get("test_roc_auc", record.get("roc_auc", 0.0)))
        record["train_pr_auc"] = float(record.get("train_pr_auc", record.get("pr_auc", 0.0)))
        record["test_pr_auc"] = float(record.get("test_pr_auc", record.get("pr_auc", 0.0)))
        record["train_f1"] = float(record.get("train_f1", record.get("f1_score", 0.0)))
        record["test_f1"] = float(record.get("test_f1", record.get("f1_score", 0.0)))
        record["val_pr_auc"] = record.get("val_pr_auc")
        record["pr_auc_gap"] = float(record.get("pr_auc_gap", record["train_pr_auc"] - record["test_pr_auc"]))
        record["f1_gap"] = float(record.get("f1_gap", record["train_f1"] - record["test_f1"]))
        record["roc_auc_gap"] = float(record.get("roc_auc_gap", record["train_roc_auc"] - record["test_roc_auc"]))
        record["overfitting_diagnosis"] = record.get("overfitting_diagnosis") or overfitting_status(record["pr_auc_gap"])
        record["execution_seconds"] = record.get("execution_seconds", 0.0)
        record["comparison"] = comparison
        records.append(record)
    return records


def normalize_comparison(record: dict) -> tuple[float | None, float | None, dict | None, str]:
    """Normalize the three baseline-vs-tuned JSON schemas used across the six model scripts:
    (a) Decision Tree / Random Forest / Gradient Boosting: single-stage CV search.
    (b) XGBoost: two-stage (broad + refined) CV search.
    (c) LSTM / MLP: validation-selected architecture search (no k-fold CV for neural nets)."""
    comparison = record["comparison"]
    if comparison is None:
        return None, None, None, "N/A"
    if "broad_search_cv_pr_auc" in comparison:
        baseline_cv = comparison.get("baseline_cv_pr_auc")
        tuned_cv = max(comparison.get("broad_search_cv_pr_auc", 0.0), comparison.get("refined_search_cv_pr_auc", 0.0))
        best_params = comparison.get("final_params_used")
    elif "baseline_validation_pr_auc" in comparison:
        baseline_cv = comparison.get("baseline_validation_pr_auc")
        tuned_cv = comparison.get("best_validation_pr_auc")
        best_params = comparison.get("best_config")
    else:
        baseline_cv = comparison.get("baseline_cv_pr_auc")
        tuned_cv = comparison.get("tuned_cv_pr_auc")
        best_params = comparison.get("best_params_from_search") or comparison.get("final_params_used")
    return baseline_cv, tuned_cv, best_params, comparison.get("selected_configuration", "N/A")


def build_charts(records: list[dict], frame: pd.DataFrame) -> dict[str, Path]:
    sns.set_theme(style="whitegrid")
    paths: dict[str, Path] = {}

    figure, axis = plt.subplots(figsize=(10, 6))
    for record in records:
        axis.plot(record["fpr"], record["tpr"], label=f"{record['display_name']} (AUC={record['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(title="Multi-model ROC Curve Comparison", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(); figure.tight_layout()
    paths["roc"] = REPORTS_DIR / "multi_model_roc_comparison.png"; figure.savefig(paths["roc"], dpi=400); plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 13)); axes = axes.ravel()
    for axis, record in zip(axes, records):
        matrix = np.array([[record["tn"], record["fp"]], [record["fn"], record["tp"]]])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=axis,
                    xticklabels=["Ineffective", "Effective"], yticklabels=["Ineffective", "Effective"])
        axis.set_title(f"{record['display_name']} (threshold={record['threshold']:.3f})")
        axis.set_xlabel("Predicted"); axis.set_ylabel("Actual")
    figure.suptitle("Confusion Matrices at Optimal Thresholds", fontsize=15); figure.tight_layout()
    paths["confusion"] = REPORTS_DIR / "confusion_matrix_grid.png"; figure.savefig(paths["confusion"], dpi=400, bbox_inches="tight"); plt.close(figure)

    ordered = frame.sort_values("pr_auc")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["display_name"], ordered["pr_auc"], color="#B23A48")
    axis.axvline(MAJORITY_PR_AUC, color="#888888", linestyle="--", label=f"Majority-class baseline ({MAJORITY_PR_AUC:.4f})")
    axis.set(title="PR-AUC Comparison (Primary Metric)", xlabel="Test PR-AUC")
    axis.legend(loc="lower right"); figure.tight_layout()
    paths["pr_bar"] = REPORTS_DIR / "pr_auc_comparison_bar.png"; figure.savefig(paths["pr_bar"], dpi=400); plt.close(figure)

    width = 0.35
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width / 2, frame["train_pr_auc"], width, label="Train PR-AUC", color="#287271")
    axis.bar(x + width / 2, frame["test_pr_auc"], width, label="Test PR-AUC", color="#B23A48")
    axis.set_xticks(x); axis.set_xticklabels(frame["display_name"], rotation=20, ha="right")
    axis.set(title="Train vs Test PR-AUC by Model (Generalization)", ylabel="PR-AUC")
    axis.legend(); figure.tight_layout()
    paths["gap"] = REPORTS_DIR / "train_test_pr_auc_comparison.png"; figure.savefig(paths["gap"], dpi=400); plt.close(figure)

    tree_records = [r for r in records if r.get("feature_importances") and any(r["feature_importances"])]
    if tree_records:
        best_tree = max(tree_records, key=lambda r: r["pr_auc"])
        importance = pd.Series(best_tree["feature_importances"], index=best_tree["feature_names"]).nlargest(10).sort_values()
        figure, axis = plt.subplots(figsize=(9, 6)); importance.plot.barh(ax=axis, color="#287271")
        axis.set(title=f"Top 10 Feature Importances: {best_tree['display_name']}", xlabel="Importance")
        figure.tight_layout()
        paths["importance"] = REPORTS_DIR / "top_10_feature_importances.png"; figure.savefig(paths["importance"], dpi=400); plt.close(figure)
    return paths


def select_champion(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    """Champion selection is NOT simply 'highest PR-AUC'. If the top-two PR-AUC scores are
    within a small, likely-noise margin, prefer whichever has the better F1/precision-recall
    balance and equally strong generalization -- matching the project's explicit selection rule."""
    ranked = frame.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    top, runner_up = ranked.iloc[0], ranked.iloc[1]
    margin = top["pr_auc"] - runner_up["pr_auc"]
    if margin < 0.002 and runner_up["overfitting_diagnosis"] == "Well-Balanced" and runner_up["f1_score"] >= top["f1_score"]:
        rationale = (
            f"{top['display_name']} and {runner_up['display_name']} are statistically tied on PR-AUC "
            f"(margin {margin:.6f}, likely within run-to-run noise). {runner_up['display_name']} is selected "
            f"as champion because it has an equal-or-better F1 ({runner_up['f1_score']:.4f} vs {top['f1_score']:.4f}) "
            f"and comparable generalization ({runner_up['overfitting_diagnosis']}), giving a slightly better "
            "precision-recall balance for identifying Effective cases."
        )
        return runner_up, rationale
    rationale = (
        f"{top['display_name']} leads on PR-AUC (the primary metric for this imbalanced problem) by a "
        f"non-trivial margin ({margin:.6f}) over the runner-up, {runner_up['display_name']}, while maintaining "
        f"{top['overfitting_diagnosis']} generalization."
    )
    return top, rationale


def main() -> None:
    ensure_output_dirs()
    records = load_records()
    frame = pd.DataFrame(records).sort_values(["pr_auc", "f1_score", "recall"], ascending=False).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    frame["overfitting_status"] = frame["overfitting_diagnosis"]

    frame.drop(columns=["comparison"]).to_json(METRICS_DIR / "model_comparison.json", orient="records", indent=2)
    frame[["rank", "display_name", "pr_auc", "f1_score", "recall", "precision", "roc_auc", "accuracy",
          "threshold", "train_pr_auc", "test_pr_auc", "pr_auc_gap", "train_f1", "test_f1", "f1_gap",
          "train_roc_auc", "test_roc_auc", "roc_auc_gap", "overfitting_diagnosis", "execution_seconds"]].rename(
        columns={"display_name": "model_name"}
    ).to_csv(METRICS_DIR / "model_comparison.txt", sep="\t", index=False)

    chart_paths = build_charts(records, frame)
    champion, champion_rationale = select_champion(frame)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#17324D")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#17324D"), spaceBefore=6 * mm, spaceAfter=2 * mm))
    styles.add(ParagraphStyle(name="SubSection", parent=styles["Heading3"], fontSize=11, textColor=colors.HexColor("#17324D"), spaceBefore=3 * mm))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=2 * mm))

    def P(text: str, style: str = "Body") -> Paragraph:
        return Paragraph(text, styles[style])

    story: list = [
        Paragraph("Clinical Treatment-Outcome Model Comparison &amp; Reflective Report", styles["ReportTitle"]),
        P(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Dataset: {TOTAL_ROWS:,} rows, 39 columns, target treatment_outcome."),
        Spacer(1, 3 * mm),
    ]

    # 1. Executive summary
    story += [P("1. Executive Summary", "Section"), P(
        f"Six models (Decision Tree, Random Forest, XGBoost, Gradient Boosting, LSTM, MLP) were tuned and evaluated "
        f"under a single leakage-safe 60/20/20 stratified split (random_state=42) to predict treatment_outcome "
        f"(0=Ineffective, 1=Effective). Because Effective cases represent only {CLASS_1_COUNT/TOTAL_ROWS:.2%} of the "
        f"data, PR-AUC (average precision) was used as the primary tuning and comparison metric throughout, rather than accuracy. "
        f"All six final models achieved PR-AUC between {frame['pr_auc'].min():.4f} and {frame['pr_auc'].max():.4f}, "
        f"well above the majority-class baseline of {MAJORITY_PR_AUC:.4f} but clustered tightly together, indicating "
        f"the dataset's feature set carries limited predictive signal (see Section 3). "
        f"The selected champion model is <b>{champion['display_name']}</b>."
    )]

    # 2. Dataset overview
    story += [P("2. Dataset Overview", "Section"), P(
        f"Source: dataset/feature_engineered_clinical_data.csv &mdash; {TOTAL_ROWS:,} rows, 39 columns, 2,461 missing cells. "
        "Features span demographics, vitals, lab values, medication/dosage details, and engineered clinical flags "
        "(kidney_stage, liver_risk, polypharmacy, bmi_category, age_group, elderly_high_dose, de_ritis_ratio). "
        "patient_id was dropped from modeling features. Preprocessing (median imputation for numeric columns, "
        "frequency-rank encoding for categoricals, and -- for the two neural networks only -- StandardScaler) was "
        "fit strictly on the training partition and applied unchanged to validation/test."
    )]

    # 3. Target imbalance / why accuracy is not primary
    story += [P("3. Target Imbalance and Why Accuracy Is Not the Primary Metric", "Section"), P(
        f"Class 0 (Ineffective) = {CLASS_0_COUNT:,} ({CLASS_0_COUNT/TOTAL_ROWS:.2%}); Class 1 (Effective) = "
        f"{CLASS_1_COUNT:,} ({CLASS_1_COUNT/TOTAL_ROWS:.2%}). A trivial classifier that always predicts 'Ineffective' "
        f"achieves {MAJORITY_ACCURACY:.2%} accuracy while identifying zero Effective cases (Recall=0, Precision=0, "
        f"PR-AUC&asymp;{MAJORITY_PR_AUC:.4f}, the class prevalence). This demonstrates why accuracy alone is misleading "
        "here: every model in this report deliberately trades some accuracy for substantially higher recall/precision "
        "on the clinically important minority class."
    )]

    # 4. Methodology
    story += [P("4. Modeling Methodology", "Section"), P(
        "Split: stratified 60% train / 20% validation / 20% test, random_state=42, identical across all six models "
        "so comparisons are fair. The test set was touched exactly once per model, for final evaluation only. "
        "Hyperparameter tuning used RandomizedSearchCV with StratifiedKFold (3 folds) scoring on average_precision "
        "for the four tree-based models; the two neural networks (LSTM, MLP) instead used a compact architecture "
        "search selected by held-out validation PR-AUC (k-fold CV was judged too computationally expensive to repeat "
        "per candidate on this machine's 8&nbsp;GB RAM budget). In every case the baseline-vs-tuned decision was made "
        "from CV/validation performance only, never from the test set."
    )]

    # 5. Baseline / majority-class results
    story += [P("5. Majority-Class Baseline", "Section"), P(
        f"Accuracy={MAJORITY_ACCURACY:.4f}, Precision=0, Recall=0, F1=0, PR-AUC&asymp;{MAJORITY_PR_AUC:.4f}. "
        "Every tuned model in this report substantially exceeds this PR-AUC floor."
    )]

    # 6-11: Per-model results
    story += [P("6. Per-Model Results (Baseline vs Tuned)", "Section")]
    for name in FILES:
        record = next(r for r in records if r["short_name"] == name)
        baseline_cv, tuned_cv, best_params, selected = normalize_comparison(record)
        story.append(P(f"{DISPLAY_NAMES[name]}", "SubSection"))
        if baseline_cv is not None:
            story.append(P(
                f"Selection basis: {baseline_cv:.6f} (baseline) vs {tuned_cv:.6f} (search) &rarr; <b>{selected}</b> retained. "
                f"Final test: PR-AUC={record['pr_auc']:.4f}, Precision={record['precision']:.4f}, Recall={record['recall']:.4f}, "
                f"F1={record['f1_score']:.4f}, ROC-AUC={record['roc_auc']:.4f}, Accuracy={record['accuracy']:.4f}, "
                f"Threshold={record['threshold']:.4f}. Generalization: Train PR-AUC={record['train_pr_auc']:.4f} / "
                f"Test PR-AUC={record['test_pr_auc']:.4f} (gap {record['pr_auc_gap']:.4f}, {record['overfitting_diagnosis']}). "
                f"Execution time: {record['execution_seconds']:.1f}s. Hyperparameters: {best_params}."
            ))
        else:
            story.append(P(
                f"Final test: PR-AUC={record['pr_auc']:.4f}, Precision={record['precision']:.4f}, Recall={record['recall']:.4f}, "
                f"F1={record['f1_score']:.4f}, ROC-AUC={record['roc_auc']:.4f}. No baseline-vs-tuned comparison file found."
            ))
        if name == "lstm":
            story.append(P(
                "<i>Sequence-structure limitation:</i> 97.0% of unique patients have exactly one row in this dataset "
                "(only ~2.95% have repeat visits, max 4). No genuine multi-step temporal sequence exists, so seq_len=1 "
                "was used deliberately rather than fabricating artificial sequences. Note: an initial LSTM run was "
                "degenerate (PR-AUC&asymp;0.222, ROC-AUC&asymp;0.50) due to missing input feature scaling, which saturated "
                "the recurrent gates; this was diagnosed and fixed by adding a train-only-fitted StandardScaler."
            ))

    story.append(PageBreak())

    # 12. Threshold optimization methodology
    story += [P("7. Threshold Optimization Methodology", "Section"), P(
        "For every model, probabilities were generated on an inner validation split never seen by the fitted "
        "estimator's parameters, and the decision threshold was chosen by maximizing F1 across the precision-recall "
        "curve on that validation split (find_best_threshold in utils.py). The chosen threshold was then applied "
        "exactly once to the untouched test set. The test set was never used to choose a threshold."
    )]

    # 13/14. PR-AUC & ROC-AUC analysis
    story += [P("8. Precision-Recall and ROC-AUC Analysis", "Section"),
              Image(str(chart_paths["pr_bar"]), width=160 * mm, height=90 * mm), Spacer(1, 3 * mm),
              Image(str(chart_paths["roc"]), width=165 * mm, height=99 * mm)]

    # 15. Confusion matrix comparison
    story += [P("9. Confusion Matrix Comparison", "Section"), Image(str(chart_paths["confusion"]), width=165 * mm, height=135 * mm)]

    story.append(PageBreak())

    # 16. Overfitting / generalization comparison
    story += [P("10. Overfitting and Generalization Comparison", "Section"),
              Image(str(chart_paths["gap"]), width=160 * mm, height=90 * mm)]
    gen_rows = [["Model", "Train PR-AUC", "Test PR-AUC", "PR-AUC Gap", "Train F1", "Test F1", "F1 Gap", "Diagnosis"]]
    for row in frame.itertuples():
        gen_rows.append([row.display_name, f"{row.train_pr_auc:.4f}", f"{row.test_pr_auc:.4f}", f"{row.pr_auc_gap:.4f}",
                         f"{row.train_f1:.4f}", f"{row.test_f1:.4f}", f"{row.f1_gap:.4f}", row.overfitting_diagnosis])
    gen_table = Table(gen_rows, colWidths=[28 * mm, 22 * mm, 22 * mm, 20 * mm, 20 * mm, 20 * mm, 18 * mm, 26 * mm])
    gen_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C2D12")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF7ED")]),
    ]))
    story += [Spacer(1, 3 * mm), gen_table]

    # 17. Final comparison table
    story += [P("11. Final Model Comparison Table", "Section")]
    headers = ["Rank", "Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "Accuracy", "Threshold", "Diagnosis"]
    rows = [headers] + [[int(row.rank), row.display_name, f"{row.pr_auc:.4f}", f"{row.roc_auc:.4f}", f"{row.precision:.4f}",
                        f"{row.recall:.4f}", f"{row.f1_score:.4f}", f"{row.accuracy:.4f}", f"{row.threshold:.4f}",
                        row.overfitting_diagnosis] for row in frame.itertuples()]
    table = Table(rows, colWidths=[10 * mm, 26 * mm, 16 * mm, 17 * mm, 18 * mm, 15 * mm, 14 * mm, 16 * mm, 16 * mm, 24 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F4")]),
    ]))
    story.append(table)

    # 18/19. Champion model + reason
    story += [P("12. Champion Model Selection", "Section"), P(
        f"<b>Champion: {champion['display_name']}</b>. {champion_rationale} Selection deliberately does NOT default "
        "to the highest accuracy, highest recall, highest ROC-AUC, or highest training score alone; PR-AUC, "
        "Class-1 precision/recall balance, F1, generalization gap, and computational practicality were all weighed."
    )]

    # 20. Feature importance
    if "importance" in chart_paths:
        story += [P("13. Feature Importance", "Section"), Image(str(chart_paths["importance"]), width=155 * mm, height=103 * mm)]

    # 21. Limitations
    story += [P("14. Limitations", "Section"), P(
        "(1) All six models cluster within a narrow PR-AUC band (~0.33-0.35), consistent with weak individual "
        "feature signal observed during data exploration (maximum mutual-information score &asymp;0.011 for 'age'); "
        "this likely reflects a ceiling in the available features rather than a modeling deficiency. "
        "(2) The LSTM has no genuine sequential structure to exploit (97% single-record patients); its recurrent "
        "architecture is not meaningfully advantaged over the MLP here. "
        "(3) Neural-network hyperparameter selection used a single validation split rather than k-fold CV, for "
        "computational reasons on an 8&nbsp;GB RAM machine; tree-model tuning used only 3-fold CV and a compact "
        "search budget for the same reason, so the explored hyperparameter space, while reasonable, was not exhaustive. "
        "(4) A subset of raw columns (e.g. adverse_event, readmission_30d) have unclear causal timing relative to the "
        "target and were retained as-is; genuine leakage was not confirmed but cannot be fully ruled out from column "
        "names alone."
    )]

    # 22. Recommendations
    story += [P("15. Recommendations for Future Work", "Section"), P(
        "Investigate richer feature engineering (interaction terms, non-linear transforms of lab values) given the "
        "weak individual mutual-information scores; obtain genuinely longitudinal per-patient records if repeat-visit "
        "sequences become available, to make an LSTM/sequence model worthwhile; consider probability calibration "
        "(e.g., Platt scaling/isotonic regression) before any clinical use; run a wider hyperparameter search with "
        "more compute if available; and perform subgroup/fairness analysis across demographic segments before deployment."
    )]

    # 23. Reproducibility
    story += [P("16. Reproducibility", "Section"), P(
        "random_state=42 throughout (data split, model fitting, CV folds). Split: 60/20/20 stratified on "
        "treatment_outcome. Preprocessing: FrequencyCategoryEncoder (median/frequency-rank, train-only fit), plus "
        "StandardScaler (train-only fit) for LSTM/MLP. Threshold method: F1-maximization on validation predictions. "
        "CV strategy: StratifiedKFold(3) for tree models scoring average_precision; held-out validation PR-AUC for "
        "neural networks. All hyperparameters, execution times, and model versions are recorded in "
        "metrics_output/*_comparison.json and *_metrics.json / *.json alongside this report."
    )]

    pdf_path = REPORTS_DIR / "Clinical_Model_Performance_Report.pdf"
    temporary_pdf_path = REPORTS_DIR / "Clinical_Model_Performance_Report.tmp.pdf"
    document = SimpleDocTemplate(str(temporary_pdf_path), pagesize=A4, rightMargin=15 * mm, leftMargin=15 * mm,
                                 topMargin=15 * mm, bottomMargin=15 * mm, title="Clinical Model Performance Report")
    document.build(story)
    try:
        temporary_pdf_path.replace(pdf_path)
    except PermissionError:
        fallback_path = REPORTS_DIR / f"Clinical_Model_Performance_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        try:
            temporary_pdf_path.replace(fallback_path)
        except PermissionError as fallback_error:
            raise PermissionError(
                f"Cannot write the report because {pdf_path} is locked and the fallback PDF could not be created. "
                "Close the PDF in any viewer and rerun the report."
            ) from fallback_error
        print(f"Warning: {pdf_path} is locked; new PDF saved to {fallback_path}")

    print(frame[["rank", "display_name", "pr_auc", "f1_score", "recall", "roc_auc", "accuracy", "precision",
                "threshold", "overfitting_diagnosis"]].to_string(index=False))
    print(f"Champion model: {champion['display_name']}")
    print(f"Report PDF: {pdf_path}")


if __name__ == "__main__":
    main()
