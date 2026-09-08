"""Fit and evaluate the leakage-safe XGBoost model: baseline vs tuned."""
import json

import numpy as np
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils import (
    METRICS_DIR,
    RANDOM_STATE,
    cross_validate_average_precision,
    ensure_output_dirs,
    fit_evaluate_save,
    load_data,
    save_model_charts,
    tune_hyperparameters,
)

# n_jobs capped at 2 (not -1), consistent with the Random Forest script, so the estimator's own
# threading never competes with RandomizedSearchCV's own parallelism on this 8 GB machine.
XGB_N_JOBS = 2
SEARCH_N_JOBS = 2

# Leakage fix: the previous script computed scale_pos_weight from the FULL dataset via
# load_data(), before any split. It must reflect the training partition only, so we reproduce
# the exact same 60/20/20 split used inside fit_evaluate_save/tune_hyperparameters here.
_X_raw, _y = load_data()
_, _, _y_train, _ = train_test_split(_X_raw, _y, test_size=0.40, stratify=_y, random_state=RANDOM_STATE)
_negative_count, _positive_count = np.bincount(_y_train.astype(int))
SCALE_POS_WEIGHT = float(_negative_count / _positive_count)

BASELINE_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=SCALE_POS_WEIGHT, eval_metric="logloss",
    n_jobs=XGB_N_JOBS, random_state=42, tree_method="hist",
)

# Compact search (n_iter=8, cv=3 -> 24 fits). No early stopping: fit_evaluate_save has no
# eval_set hook, so n_estimators is bounded by the search itself and overfitting is instead
# controlled via subsample/colsample_bytree/gamma/reg_alpha/reg_lambda and shallower max_depth.
PARAM_DISTRIBUTIONS = {
    "n_estimators": randint(100, 251),
    "max_depth": randint(3, 8),
    "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "min_child_weight": randint(1, 11),
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.1, 0.3, 0.5, 1.0],
    "reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
}


def _refine_param_distributions(center: dict) -> dict:
    """Build a narrower search grid centered on a prior best result (cheaper fits since the
    winning region so far uses shallow trees: max_depth<=4, modest n_estimators)."""
    depth = int(center["max_depth"])
    return {
        "n_estimators": randint(max(80, center["n_estimators"] - 60), center["n_estimators"] + 61),
        "max_depth": [max(2, depth - 1), depth, depth + 1],
        "learning_rate": sorted({round(center["learning_rate"] * factor, 4) for factor in (0.6, 0.8, 1.0, 1.4, 1.8)}),
        "min_child_weight": randint(max(1, center["min_child_weight"] - 5), center["min_child_weight"] + 6),
        "subsample": sorted({round(min(1.0, max(0.5, center["subsample"] + delta)), 2) for delta in (-0.2, -0.1, 0.0, 0.1)}),
        "colsample_bytree": sorted({round(min(1.0, max(0.5, center["colsample_bytree"] + delta)), 2) for delta in (-0.2, -0.1, 0.0)}),
        "gamma": sorted({round(max(0.0, center["gamma"] + delta), 3) for delta in (-0.05, 0.0, 0.1, 0.2)}),
        "reg_alpha": sorted({round(max(0.0, center["reg_alpha"] + delta), 3) for delta in (-0.005, 0.0, 0.05, 0.1)}),
        "reg_lambda": sorted({round(max(0.1, center["reg_lambda"] + delta), 2) for delta in (-0.5, 0.0, 0.5, 1.0)}),
    }


def main() -> None:
    ensure_output_dirs()

    print("=== [1/8] Baseline XGBoost: training ===")
    baseline_metrics = fit_evaluate_save(
        XGBClassifier(**BASELINE_PARAMS),
        "XGBoost (Baseline)", "xgboost_baseline",
        f"scale_pos_weight={SCALE_POS_WEIGHT:.4f} (computed from training partition only)",
    )
    save_model_charts(baseline_metrics, "xgboost_baseline")

    print("=== [2/8] Baseline XGBoost: cross-validated PR-AUC (train-only) ===")
    baseline_cv_pr_auc = cross_validate_average_precision(
        XGBClassifier(**{**BASELINE_PARAMS, "n_jobs": 1}), cv_splits=3, search_n_jobs=SEARCH_N_JOBS,
    )
    print(f"Baseline CV PR-AUC: {baseline_cv_pr_auc:.6f}")

    print("=== [3/8] Broad hyperparameter search (train-only StratifiedKFold CV, scoring=average_precision) ===")
    broad_params, broad_cv_pr_auc = tune_hyperparameters(
        XGBClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT, eval_metric="logloss",
            n_jobs=1, random_state=42, tree_method="hist",
        ),
        PARAM_DISTRIBUTIONS, n_iter=8, cv_splits=3, search_n_jobs=SEARCH_N_JOBS, verbose=2,
    )
    print(f"Broad search best CV PR-AUC: {broad_cv_pr_auc:.6f}")
    print(f"Broad search best params: {broad_params}")

    print("=== [4/8] Refined hyperparameter search (narrow grid centered on broad-search winner) ===")
    refined_distributions = _refine_param_distributions(broad_params)
    refined_params, refined_cv_pr_auc = tune_hyperparameters(
        XGBClassifier(
            scale_pos_weight=SCALE_POS_WEIGHT, eval_metric="logloss",
            n_jobs=1, random_state=42, tree_method="hist",
        ),
        refined_distributions, n_iter=8, cv_splits=3, search_n_jobs=SEARCH_N_JOBS, verbose=2,
    )
    print(f"Refined search best CV PR-AUC: {refined_cv_pr_auc:.6f}")
    print(f"Refined search best params: {refined_params}")

    print("=== [5/8] Selecting baseline vs broad vs refined configuration (decision based on CV, not test) ===")
    candidates = [
        ("Baseline", baseline_cv_pr_auc, BASELINE_PARAMS),
        ("Tuned (broad search)", broad_cv_pr_auc, {**broad_params, "scale_pos_weight": SCALE_POS_WEIGHT, "eval_metric": "logloss", "n_jobs": XGB_N_JOBS, "random_state": 42, "tree_method": "hist"}),
        ("Tuned (refined search)", refined_cv_pr_auc, {**refined_params, "scale_pos_weight": SCALE_POS_WEIGHT, "eval_metric": "logloss", "n_jobs": XGB_N_JOBS, "random_state": 42, "tree_method": "hist"}),
    ]
    selected, tuned_cv_pr_auc, final_params = max(candidates, key=lambda candidate: candidate[1])
    best_params = broad_params if selected == "Tuned (broad search)" else refined_params if selected == "Tuned (refined search)" else {}
    strategy_note = f"selected='{selected}' (CV PR-AUC: baseline={baseline_cv_pr_auc:.4f}, broad={broad_cv_pr_auc:.4f}, refined={refined_cv_pr_auc:.4f})"
    print(f"Selected configuration: {selected}")

    print(f"=== [6/8] Final XGBoost ({selected}): fitting and evaluating on the untouched test set ===")
    final_metrics = fit_evaluate_save(
        XGBClassifier(**final_params),
        "XGBoost", "xgboost",
        f"scale_pos_weight={SCALE_POS_WEIGHT:.4f} (train-only); {strategy_note}",
    )
    save_model_charts(final_metrics, "xgboost")

    print("=== [7/8] Saving baseline-vs-tuned comparison + best hyperparameters ===")
    comparison = {
        "selected_configuration": selected,
        "baseline_cv_pr_auc": baseline_cv_pr_auc,
        "broad_search_cv_pr_auc": broad_cv_pr_auc,
        "refined_search_cv_pr_auc": refined_cv_pr_auc,
        "broad_search_best_params": broad_params,
        "refined_search_best_params": refined_params,
        "final_params_used": final_params,
        "baseline_test": {key: baseline_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "final_test": {key: final_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "generalization": {
            "train_pr_auc": final_metrics["train_pr_auc"], "val_pr_auc": final_metrics["val_pr_auc"], "test_pr_auc": final_metrics["test_pr_auc"],
            "train_roc_auc": final_metrics["train_roc_auc"], "val_roc_auc": final_metrics["val_roc_auc"], "test_roc_auc": final_metrics["test_roc_auc"],
            "train_val_pr_auc_gap": final_metrics["train_pr_auc"] - final_metrics["val_pr_auc"],
            "pr_auc_gap": final_metrics["pr_auc_gap"], "f1_gap": final_metrics["f1_gap"], "roc_auc_gap": final_metrics["roc_auc_gap"],
            "overfitting_diagnosis": final_metrics["overfitting_diagnosis"],
        },
    }
    (METRICS_DIR / "xgboost_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (METRICS_DIR / "xgboost_best_params.json").write_text(json.dumps({"selected_configuration": selected, "params": final_params}, indent=2), encoding="utf-8")

    lines = [
        "XGBOOST: BASELINE VS TUNED COMPARISON",
        "=" * 40,
        f"Selected configuration: {selected}",
        f"Baseline CV PR-AUC (train-only, 3-fold):        {baseline_cv_pr_auc:.6f}",
        f"Broad search best CV PR-AUC (train-only, 3-fold):  {broad_cv_pr_auc:.6f}",
        f"Refined search best CV PR-AUC (train-only, 3-fold): {refined_cv_pr_auc:.6f}",
        f"Broad search best params: {broad_params}",
        f"Refined search best params: {refined_params}",
        "",
        f"{'Metric':<12}{'Baseline (Test)':<18}{'Final (Test)':<18}{'Change':<12}",
    ]
    print("=== [8/8] Done ===")
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
    (METRICS_DIR / "xgboost_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nXGBoost training complete.")


if __name__ == "__main__":
    main()
