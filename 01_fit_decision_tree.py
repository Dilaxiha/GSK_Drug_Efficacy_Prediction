"""Fit and evaluate the leakage-safe Decision Tree model: baseline vs tuned."""
import json

from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier

from utils import (
    METRICS_DIR,
    cross_validate_average_precision,
    ensure_output_dirs,
    fit_evaluate_save,
    save_model_charts,
    tune_hyperparameters,
)

BASELINE_PARAMS = dict(
    max_depth=6, min_samples_leaf=75, min_samples_split=150,
    class_weight="balanced", random_state=42,
)

PARAM_DISTRIBUTIONS = {
    "criterion": ["gini", "entropy"],
    "max_depth": randint(4, 26),
    "min_samples_split": randint(20, 400),
    "min_samples_leaf": randint(10, 200),
    "max_features": [None, "sqrt", "log2", 0.5, 0.75],
    "class_weight": ["balanced", None],
}


def main() -> None:
    ensure_output_dirs()

    print("=== Baseline Decision Tree ===")
    baseline_metrics = fit_evaluate_save(
        DecisionTreeClassifier(**BASELINE_PARAMS),
        "Decision Tree (Baseline)", "decision_tree_baseline",
        "class_weight='balanced', criterion='gini' (fixed baseline hyperparameters)",
    )
    save_model_charts(baseline_metrics, "decision_tree_baseline")
    baseline_cv_pr_auc = cross_validate_average_precision(DecisionTreeClassifier(**BASELINE_PARAMS))

    print("=== Hyperparameter search (train-only StratifiedKFold CV, scoring=average_precision) ===")
    best_params, tuned_cv_pr_auc = tune_hyperparameters(
        DecisionTreeClassifier(random_state=42), PARAM_DISTRIBUTIONS, n_iter=25, cv_splits=3,
    )
    print(f"Baseline CV PR-AUC: {baseline_cv_pr_auc:.6f}")
    print(f"Tuned best CV PR-AUC: {tuned_cv_pr_auc:.6f}")
    print(f"Best params: {best_params}")

    if tuned_cv_pr_auc > baseline_cv_pr_auc:
        final_params = {**best_params, "random_state": 42}
        selected = "Tuned"
        strategy_note = f"tuned via RandomizedSearchCV (CV PR-AUC={tuned_cv_pr_auc:.4f} vs baseline {baseline_cv_pr_auc:.4f})"
    else:
        final_params = BASELINE_PARAMS
        selected = "Baseline"
        strategy_note = f"baseline retained (CV PR-AUC={baseline_cv_pr_auc:.4f} >= tuned {tuned_cv_pr_auc:.4f})"

    print(f"=== Final Decision Tree ({selected}) ===")
    final_metrics = fit_evaluate_save(
        DecisionTreeClassifier(**final_params),
        "Decision Tree", "decision_tree",
        f"class_weight='{final_params.get('class_weight')}'; {strategy_note}",
    )
    save_model_charts(final_metrics, "decision_tree")

    comparison = {
        "selected_configuration": selected,
        "baseline_cv_pr_auc": baseline_cv_pr_auc,
        "tuned_cv_pr_auc": tuned_cv_pr_auc,
        "best_params_from_search": best_params,
        "final_params_used": final_params,
        "baseline_test": {key: baseline_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "final_test": {key: final_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
    }
    (METRICS_DIR / "decision_tree_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "DECISION TREE: BASELINE VS TUNED COMPARISON",
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
    (METRICS_DIR / "decision_tree_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nDecision Tree training complete.")


if __name__ == "__main__":
    main()
