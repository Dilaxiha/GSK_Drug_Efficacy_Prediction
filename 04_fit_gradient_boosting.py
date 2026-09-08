"""Fit and evaluate the leakage-safe HistGradientBoosting model: baseline vs tuned."""
import json

from scipy.stats import randint
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import (
    METRICS_DIR,
    cross_validate_average_precision,
    ensure_output_dirs,
    fit_evaluate_save,
    save_model_charts,
    tune_hyperparameters,
)

# HistGradientBoostingClassifier has no n_jobs argument (its own internal threading is fixed);
# RandomizedSearchCV/cross_val_score parallelism is still capped via SEARCH_N_JOBS for 8 GB safety.
SEARCH_N_JOBS = 2

# early_stopping=True carves its own internal validation slice out of whatever training data it
# is given (train-only in every call below), so overfitting is bounded automatically without
# needing an external eval_set -- unlike XGBoost/RandomForest, no extra plumbing is required.
BASELINE_PARAMS = dict(class_weight="balanced", early_stopping=True, random_state=42)

# Compact search (n_iter=8, cv=3 -> 24 fits). max_iter is a ceiling only: early stopping usually
# halts well before reaching it, so this stays memory/time-bounded despite the wide upper range.
PARAM_DISTRIBUTIONS = {
    "max_iter": randint(100, 301),
    "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "max_leaf_nodes": randint(15, 64),
    "max_depth": [None, 3, 5, 7, 9],
    "min_samples_leaf": randint(15, 100),
    "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    "max_bins": [127, 255],
    "class_weight": ["balanced", None],
}


def main() -> None:
    ensure_output_dirs()

    print("=== [1/6] Baseline Gradient Boosting: training ===")
    baseline_metrics = fit_evaluate_save(
        HistGradientBoostingClassifier(**BASELINE_PARAMS),
        "Gradient Boosting (Baseline)", "gradient_boosting_baseline",
        "class_weight='balanced' with early_stopping=True (fixed baseline hyperparameters)",
    )
    save_model_charts(baseline_metrics, "gradient_boosting_baseline")

    print("=== [2/6] Baseline Gradient Boosting: cross-validated PR-AUC (train-only) ===")
    baseline_cv_pr_auc = cross_validate_average_precision(
        HistGradientBoostingClassifier(**BASELINE_PARAMS), cv_splits=3, search_n_jobs=SEARCH_N_JOBS,
    )
    print(f"Baseline CV PR-AUC: {baseline_cv_pr_auc:.6f}")

    print("=== [3/6] Hyperparameter search (train-only StratifiedKFold CV, scoring=average_precision) ===")
    best_params, tuned_cv_pr_auc = tune_hyperparameters(
        HistGradientBoostingClassifier(early_stopping=True, random_state=42),
        PARAM_DISTRIBUTIONS, n_iter=8, cv_splits=3, search_n_jobs=SEARCH_N_JOBS, verbose=2,
    )
    print(f"Tuned best CV PR-AUC: {tuned_cv_pr_auc:.6f}")
    print(f"Best params: {best_params}")

    print("=== [4/6] Selecting baseline vs tuned configuration (decision based on CV, not test) ===")
    if tuned_cv_pr_auc > baseline_cv_pr_auc:
        final_params = {**best_params, "early_stopping": True, "random_state": 42}
        selected = "Tuned"
        strategy_note = f"tuned via RandomizedSearchCV (CV PR-AUC={tuned_cv_pr_auc:.4f} vs baseline {baseline_cv_pr_auc:.4f})"
    else:
        final_params = BASELINE_PARAMS
        selected = "Baseline"
        strategy_note = f"baseline retained (CV PR-AUC={baseline_cv_pr_auc:.4f} >= tuned {tuned_cv_pr_auc:.4f})"
    print(f"Selected configuration: {selected}")

    print(f"=== [5/6] Final Gradient Boosting ({selected}): fitting and evaluating on the untouched test set ===")
    final_metrics = fit_evaluate_save(
        HistGradientBoostingClassifier(**final_params),
        "Gradient Boosting", "gradient_boosting",
        f"class_weight='{final_params.get('class_weight')}' with early_stopping=True; {strategy_note}",
    )
    save_model_charts(final_metrics, "gradient_boosting")

    print("=== [6/6] Saving baseline-vs-tuned comparison + best hyperparameters ===")
    comparison = {
        "selected_configuration": selected,
        "baseline_cv_pr_auc": baseline_cv_pr_auc,
        "tuned_cv_pr_auc": tuned_cv_pr_auc,
        "best_params_from_search": best_params,
        "final_params_used": final_params,
        "baseline_test": {key: baseline_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "final_test": {key: final_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "generalization": {
            "train_pr_auc": final_metrics["train_pr_auc"], "val_pr_auc": final_metrics["val_pr_auc"], "test_pr_auc": final_metrics["test_pr_auc"],
            "train_roc_auc": final_metrics["train_roc_auc"], "val_roc_auc": final_metrics["val_roc_auc"], "test_roc_auc": final_metrics["test_roc_auc"],
            "pr_auc_gap": final_metrics["pr_auc_gap"], "f1_gap": final_metrics["f1_gap"], "roc_auc_gap": final_metrics["roc_auc_gap"],
            "overfitting_diagnosis": final_metrics["overfitting_diagnosis"],
        },
    }
    (METRICS_DIR / "gradient_boosting_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (METRICS_DIR / "gradient_boosting_best_params.json").write_text(json.dumps({"selected_configuration": selected, "params": final_params}, indent=2), encoding="utf-8")

    lines = [
        "GRADIENT BOOSTING: BASELINE VS TUNED COMPARISON",
        "=" * 48,
        f"Selected configuration: {selected}",
        f"Baseline CV PR-AUC (train-only, 3-fold): {baseline_cv_pr_auc:.6f}",
        f"Tuned CV PR-AUC (train-only, 3-fold):    {tuned_cv_pr_auc:.6f}",
        f"Best params from search: {best_params}",
        "",
        f"{'Metric':<12}{'Baseline (Test)':<18}{'Final (Test)':<18}{'Change':<12}",
    ]
    for label, key in (("PR-AUC", "pr_auc"), ("F1", "f1_score"), ("Precision", "precision"), ("Recall", "recall"), ("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Threshold", "threshold")):
        change = final_metrics[key] - baseline_metrics[key]
        lines.append(f"{label:<12}{baseline_metrics[key]:<18.6f}{final_metrics[key]:<18.6f}{change:+.6f}")

    lines.extend([
        "",
        "GENERALIZATION (final model)",
        f"{'Split':<12}{'PR-AUC':<12}{'ROC-AUC':<12}",
        f"{'Train':<12}{final_metrics['train_pr_auc']:<12.6f}{final_metrics['train_roc_auc']:<12.6f}",
        f"{'Validation':<12}{final_metrics['val_pr_auc']:<12.6f}{final_metrics['val_roc_auc']:<12.6f}",
        f"{'Test':<12}{final_metrics['test_pr_auc']:<12.6f}{final_metrics['test_roc_auc']:<12.6f}",
        f"Train-Test PR-AUC gap: {final_metrics['pr_auc_gap']:.6f} | Diagnosis: {final_metrics['overfitting_diagnosis']}",
    ])
    (METRICS_DIR / "gradient_boosting_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nGradient Boosting training complete.")


if __name__ == "__main__":
    main()
