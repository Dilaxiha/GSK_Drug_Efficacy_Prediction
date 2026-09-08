"""Fit and evaluate a single, non-ensembled-with-anything-else LightGBM model (LGBMClassifier)
on the leakage-safe clinical split, tuned for an 8 GB machine, then compare it against the
existing tuned XGBoost baseline in metrics_output/xgboost.json.

"Single model, no stacking": this trains exactly one LGBMClassifier (LightGBM's own internal
boosting is not ensembling/stacking with other model families) and reports it on its own merits
against the XGBoost run -- it does not average/stack predictions across models.

Memory/speed design (mirrors 03_fit_xgboost.py):
- n_jobs capped at 2 everywhere so the estimator's own threading never competes with the
  search's parallelism for the same 8 GB.
- The shared FrequencyCategoryEncoder (utils.py) already emits float32 for numeric columns and
  int16 for categoricals on every transform, so no extra float32 cast is needed before fit.
- Early stopping (early_stopping_rounds=15, eval_metric="average_precision" -- LightGBM's native
  PR-AUC metric name; XGBoost calls the same thing "aucpr") with a high n_estimators cap (1000)
  is used only for the two FINAL fits (baseline + tuned, via fit_evaluate_save's
  eval_set_split=True) so tree building stops automatically once validation PR-AUC plateaus.
  It is NOT used inside cross_validate_average_precision/the hyperparameter search: those loop
  over CV folds internally and don't expose a per-fold, leakage-free eval_set, so a small fixed
  n_estimators is used there instead (SEARCH_N_ESTIMATORS).
- Hyperparameter search is a single HalvingRandomSearchCV pass over the training partition:
  candidates start on a small row subset and only the strongest are promoted to progressively
  larger subsets (factor=3), keeping total fit count and peak memory low.
- subsample_freq is fixed >=1 in every LightGBM instance: LightGBM silently IGNORES the
  `subsample` (bagging fraction) parameter unless subsample_freq > 0, which would otherwise make
  the tuned `subsample` a no-op.
"""
import json

import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import randint, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401 (registers HalvingRandomSearchCV)
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold, train_test_split

from utils import (
    METRICS_DIR,
    RANDOM_STATE,
    FrequencyCategoryEncoder,
    cross_validate_average_precision,
    ensure_output_dirs,
    fit_evaluate_save,
    load_data,
    save_model_charts,
)

LGBM_N_JOBS = 2
SEARCH_N_JOBS = 2
SEARCH_N_ESTIMATORS = 300  # fixed cap used only inside CV/search (no per-fold early stopping)
FINAL_N_ESTIMATORS = 1000  # high safety cap for the two final fits; early stopping halts sooner
EARLY_STOPPING_ROUNDS = 15
EVAL_METRIC = "average_precision"  # LightGBM's native PR-AUC metric (XGBoost calls it "aucpr")

# Leakage fix: scale_pos_weight must reflect the training partition only, so we reproduce the
# exact same 60/20/20 split used inside fit_evaluate_save/the search helpers here.
_X_raw, _y = load_data()
_, _, _y_train, _ = train_test_split(_X_raw, _y, test_size=0.40, stratify=_y, random_state=RANDOM_STATE)
_negative_count, _positive_count = np.bincount(_y_train.astype(int))
SCALE_POS_WEIGHT = float(_negative_count / _positive_count)

BASELINE_HYPERPARAMS = dict(max_depth=4, num_leaves=31, learning_rate=0.1, scale_pos_weight=SCALE_POS_WEIGHT)

# Focused, shallow-tree search space: max_depth 3-6 and num_leaves 15-31 both curb overfitting on
# the imbalanced target; scale_pos_weight is swept around the train-only computed weight.
PARAM_DISTRIBUTIONS = {
    "max_depth": randint(3, 7),
    "num_leaves": randint(15, 32),
    "learning_rate": uniform(0.02, 0.08),  # sampled from [0.02, 0.10)
    "min_child_samples": randint(10, 101),
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "scale_pos_weight": [round(SCALE_POS_WEIGHT * factor, 3) for factor in (0.5, 0.75, 1.0, 1.25, 1.5)],
}


def _final_params(hyperparams: dict) -> dict:
    """Attach the final-fit-only settings (high n_estimators cap; early stopping is wired via
    fit_kwargs, not here) to a set of tuned hyperparameters."""
    return {
        **hyperparams,
        "n_estimators": FINAL_N_ESTIMATORS,
        "subsample_freq": 1,
        "n_jobs": LGBM_N_JOBS,
        "random_state": 42,
        "verbose": -1,
    }


def _halving_search(param_distributions: dict) -> tuple[dict, float]:
    """Single-pass successive-halving search on the train-only partition: cheap candidates are
    screened on a small row subset first, and only the top third are promoted each round to a
    3x larger subset."""
    print("  -> preparing training partition for halving search...")
    X_raw, y = load_data()
    X_train_raw, _, y_train, _ = train_test_split(X_raw, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE)
    encoder = FrequencyCategoryEncoder().fit(X_train_raw)
    X_train = encoder.transform(X_train_raw)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = HalvingRandomSearchCV(
        LGBMClassifier(
            n_estimators=SEARCH_N_ESTIMATORS, subsample_freq=1,
            n_jobs=1, random_state=42, verbose=-1,
        ),
        param_distributions,
        n_candidates=40, factor=3, resource="n_samples", min_resources=5000,
        scoring="average_precision", cv=cv, random_state=RANDOM_STATE,
        n_jobs=SEARCH_N_JOBS, refit=False, verbose=2,
    )
    search.fit(X_train, y_train)
    return dict(search.best_params_), float(search.best_score_)


def main() -> None:
    ensure_output_dirs()
    early_stopping_fit_kwargs = {
        "eval_metric": EVAL_METRIC,
        "callbacks": [early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False), log_evaluation(period=0)],
    }

    print("=== [1/6] Baseline LightGBM: training (early stopping, n_estimators cap=1000) ===")
    baseline_metrics = fit_evaluate_save(
        LGBMClassifier(**_final_params(BASELINE_HYPERPARAMS)),
        "LightGBM (Baseline)", "lightgbm_baseline",
        f"scale_pos_weight={SCALE_POS_WEIGHT:.4f} (train partition only); "
        f"early stopping (rounds={EARLY_STOPPING_ROUNDS}, metric={EVAL_METRIC}, cap={FINAL_N_ESTIMATORS})",
        eval_set_split=True, fit_kwargs=early_stopping_fit_kwargs,
    )
    save_model_charts(baseline_metrics, "lightgbm_baseline")

    print("=== [2/6] Baseline LightGBM: cross-validated PR-AUC (train-only, fixed n_estimators, no early stopping) ===")
    baseline_cv_pr_auc = cross_validate_average_precision(
        LGBMClassifier(**BASELINE_HYPERPARAMS, n_estimators=SEARCH_N_ESTIMATORS, subsample_freq=1,
                        n_jobs=1, random_state=42, verbose=-1),
        cv_splits=3, search_n_jobs=SEARCH_N_JOBS,
    )
    print(f"Baseline CV PR-AUC: {baseline_cv_pr_auc:.6f}")

    print("=== [3/6] Single-pass HalvingRandomSearchCV (train-only StratifiedKFold CV, scoring=average_precision) ===")
    best_params, tuned_cv_pr_auc = _halving_search(PARAM_DISTRIBUTIONS)
    print(f"Halving search best CV PR-AUC: {tuned_cv_pr_auc:.6f}")
    print(f"Halving search best params: {best_params}")

    print("=== [4/6] Selecting baseline vs tuned configuration (decision based on CV, not test) ===")
    if tuned_cv_pr_auc > baseline_cv_pr_auc:
        final_params = _final_params(best_params)
        selected = "Tuned"
        strategy_note = f"tuned via HalvingRandomSearchCV (CV PR-AUC={tuned_cv_pr_auc:.4f} vs baseline {baseline_cv_pr_auc:.4f})"
    else:
        final_params = _final_params(BASELINE_HYPERPARAMS)
        selected = "Baseline"
        strategy_note = f"baseline retained (CV PR-AUC={baseline_cv_pr_auc:.4f} >= tuned {tuned_cv_pr_auc:.4f})"
    print(f"Selected configuration: {selected}")

    print(f"=== [5/6] Final LightGBM ({selected}): fitting and evaluating on the untouched test set ===")
    final_metrics = fit_evaluate_save(
        LGBMClassifier(**final_params),
        "LightGBM", "lightgbm",
        f"scale_pos_weight={final_params['scale_pos_weight']:.4f} (train-only); {strategy_note}; "
        f"early stopping (rounds={EARLY_STOPPING_ROUNDS}, metric={EVAL_METRIC}, cap={FINAL_N_ESTIMATORS})",
        eval_set_split=True, fit_kwargs=early_stopping_fit_kwargs,
    )
    save_model_charts(final_metrics, "lightgbm")
    final_bundle = joblib.load(METRICS_DIR / "lightgbm.pkl")
    trees_used = getattr(final_bundle["estimator"], "best_iteration_", None)
    if trees_used:
        print(f"Early stopping halted at tree {trees_used} of {FINAL_N_ESTIMATORS} cap")

    print("=== [6/6] Saving baseline-vs-tuned comparison + best hyperparameters ===")
    comparison = {
        "selected_configuration": selected,
        "baseline_cv_pr_auc": baseline_cv_pr_auc,
        "tuned_cv_pr_auc": tuned_cv_pr_auc,
        "best_params_from_search": best_params,
        "final_params_used": final_params,
        "final_trees_used": trees_used,
        "baseline_test": {key: baseline_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "final_test": {key: final_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "generalization": {
            "train_pr_auc": final_metrics["train_pr_auc"], "val_pr_auc": final_metrics["val_pr_auc"], "test_pr_auc": final_metrics["test_pr_auc"],
            "train_roc_auc": final_metrics["train_roc_auc"], "val_roc_auc": final_metrics["val_roc_auc"], "test_roc_auc": final_metrics["test_roc_auc"],
            "pr_auc_gap": final_metrics["pr_auc_gap"], "f1_gap": final_metrics["f1_gap"], "roc_auc_gap": final_metrics["roc_auc_gap"],
            "overfitting_diagnosis": final_metrics["overfitting_diagnosis"],
        },
    }
    (METRICS_DIR / "lightgbm_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (METRICS_DIR / "lightgbm_best_params.json").write_text(json.dumps({"selected_configuration": selected, "params": final_params}, indent=2), encoding="utf-8")

    lines = [
        "LIGHTGBM: BASELINE VS TUNED COMPARISON",
        "=" * 40,
        f"Selected configuration: {selected}",
        f"Baseline CV PR-AUC (train-only, 3-fold, fixed n_estimators={SEARCH_N_ESTIMATORS}): {baseline_cv_pr_auc:.6f}",
        f"Halving search best CV PR-AUC (train-only, 3-fold):                           {tuned_cv_pr_auc:.6f}",
        f"Halving search best params: {best_params}",
        f"Final model trees used: {trees_used or 'N/A'} of {FINAL_N_ESTIMATORS} cap (early_stopping_rounds={EARLY_STOPPING_ROUNDS})",
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

    xgboost_path = METRICS_DIR / "xgboost.json"
    if xgboost_path.exists():
        xgb_metrics = json.loads(xgboost_path.read_text(encoding="utf-8"))
        lines.extend([
            "",
            "LIGHTGBM (FINAL) VS EXISTING TUNED XGBOOST BASELINE (both on the untouched test set)",
            f"{'Metric':<12}{'XGBoost':<18}{'LightGBM':<18}{'Change':<12}",
        ])
        for label, key in (("PR-AUC", "pr_auc"), ("F1", "f1_score"), ("Precision", "precision"), ("Recall", "recall"), ("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy")):
            change = final_metrics[key] - xgb_metrics[key]
            lines.append(f"{label:<12}{xgb_metrics[key]:<18.6f}{final_metrics[key]:<18.6f}{change:+.6f}")
    else:
        lines.append("\n(No metrics_output/xgboost.json found -- run 03_fit_xgboost.py first to enable the comparison.)")

    (METRICS_DIR / "lightgbm_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nLightGBM training complete.")


if __name__ == "__main__":
    main()
