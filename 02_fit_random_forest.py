"""Fit and evaluate the leakage-safe Random Forest model: baseline vs tuned."""
import json

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

from utils import (
    METRICS_DIR,
    cross_validate_average_precision,
    ensure_output_dirs,
    fit_evaluate_save,
    save_model_charts,
    tune_hyperparameters,
)

# n_jobs capped at 2 (not -1) throughout: an 8 GB machine cannot safely hold many parallel
# copies of the ~600k-row training fold plus large forests at once.
RF_N_JOBS = 2
# Search-level parallelism (RandomizedSearchCV/cross_val_score). Estimators used inside the
# search always use n_jobs=1 so the two levels of parallelism never compete for the same cores/RAM.
SEARCH_N_JOBS = 2

BASELINE_PARAMS = dict(
    n_estimators=80, max_depth=12, class_weight="balanced", n_jobs=RF_N_JOBS, random_state=42,
)

# Reduced search space (n_iter=6, cv=3 -> 18 fits, down from 36) to fit an 8 GB machine:
# smaller forests, shallower trees, larger leaves, and cheaper split criteria/feature subsets.
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(50, 101),
    "max_depth": randint(6, 16),
    "min_samples_split": randint(20, 151),
    "min_samples_leaf": randint(20, 101),
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini"],
    "class_weight": ["balanced", "balanced_subsample"],
}


def main() -> None:
    ensure_output_dirs()

    print("=== [1/6] Baseline Random Forest: training ===")
    baseline_metrics = fit_evaluate_save(
        RandomForestClassifier(**BASELINE_PARAMS),
        "Random Forest (Baseline)", "random_forest_baseline",
        "class_weight='balanced' (fixed baseline hyperparameters)",
    )
    save_model_charts(baseline_metrics, "random_forest_baseline")

    print("=== [2/6] Baseline Random Forest: cross-validated PR-AUC (train-only) ===")
    baseline_cv_pr_auc = cross_validate_average_precision(
        RandomForestClassifier(**{**BASELINE_PARAMS, "n_jobs": 1}), cv_splits=3, search_n_jobs=SEARCH_N_JOBS,
    )
    print(f"Baseline CV PR-AUC: {baseline_cv_pr_auc:.6f}")

    print("=== [3/6] Hyperparameter search (train-only StratifiedKFold CV, scoring=average_precision) ===")
    best_params, tuned_cv_pr_auc = tune_hyperparameters(
        RandomForestClassifier(random_state=42, n_jobs=1), PARAM_DISTRIBUTIONS,
        n_iter=6, cv_splits=3, search_n_jobs=SEARCH_N_JOBS, verbose=2,
    )
    print(f"Tuned best CV PR-AUC: {tuned_cv_pr_auc:.6f}")
    print(f"Best params: {best_params}")

    print("=== [4/6] Selecting baseline vs tuned configuration (decision based on CV, not test) ===")
    if tuned_cv_pr_auc > baseline_cv_pr_auc:
        final_params = {**best_params, "random_state": 42, "n_jobs": RF_N_JOBS}
        selected = "Tuned"
        strategy_note = f"tuned via RandomizedSearchCV (CV PR-AUC={tuned_cv_pr_auc:.4f} vs baseline {baseline_cv_pr_auc:.4f})"
    else:
        final_params = BASELINE_PARAMS
        selected = "Baseline"
        strategy_note = f"baseline retained (CV PR-AUC={baseline_cv_pr_auc:.4f} >= tuned {tuned_cv_pr_auc:.4f})"
    print(f"Selected configuration: {selected}")

    print(f"=== [5/6] Final Random Forest ({selected}): fitting and evaluating on the untouched test set ===")
    final_metrics = fit_evaluate_save(
        RandomForestClassifier(**final_params),
        "Random Forest", "random_forest",
        f"class_weight='{final_params.get('class_weight')}'; {strategy_note}",
    )
    save_model_charts(final_metrics, "random_forest")

    print("=== [6/6] Saving baseline-vs-tuned comparison report ===")
    comparison = {
        "selected_configuration": selected,
        "baseline_cv_pr_auc": baseline_cv_pr_auc,
        "tuned_cv_pr_auc": tuned_cv_pr_auc,
        "best_params_from_search": best_params,
        "final_params_used": final_params,
        "baseline_test": {key: baseline_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "final_test": {key: final_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
    }
    (METRICS_DIR / "random_forest_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "RANDOM FOREST: BASELINE VS TUNED COMPARISON",
        "=" * 44,
        f"Selected configuration: {selected}",
        f"Baseline CV PR-AUC (train-only, 3-fold): {baseline_cv_pr_auc:.6f}",
        f"Tuned CV PR-AUC (train-only, 3-fold):    {tuned_cv_pr_auc:.6f}",
        f"Best params from search: {best_params}",
        "",
        f"{'Metric':<12}{'Baseline (Test)':<18}{'Final (Test)':<18}",
        f"{'PR-AUC':<12}{baseline_metrics['pr_auc']:<18.6f}{final_metrics['pr_auc']:<18.6f}",
        f"{'F1':<12}{baseline_metrics['f1_score']:<18.6f}{final_metrics['f1_score']:<18.6f}",
        f"{'Precision':<12}{baseline_metrics['precision']:<18.6f}{final_metrics['precision']:<18.6f}",
        f"{'Recall':<12}{baseline_metrics['recall']:<18.6f}{final_metrics['recall']:<18.6f}",
        f"{'ROC-AUC':<12}{baseline_metrics['roc_auc']:<18.6f}{final_metrics['roc_auc']:<18.6f}",
        f"{'Accuracy':<12}{baseline_metrics['accuracy']:<18.6f}{final_metrics['accuracy']:<18.6f}",
        f"{'Threshold':<12}{baseline_metrics['threshold']:<18.6f}{final_metrics['threshold']:<18.6f}",
    ]
    (METRICS_DIR / "random_forest_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nRandom Forest training complete.")


if __name__ == "__main__":
    main()
