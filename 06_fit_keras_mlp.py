"""Step 7: fit and evaluate a Keras MLP: baseline vs a compact architecture search."""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tensorflow import keras
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from utils import (METRICS_DIR, REPORTS_DIR, ensure_output_dirs,
                   find_best_threshold, get_train_val_test_data, save_model_charts)

EPOCHS = 100
PATIENCE = 5

# Baseline is CANDIDATES[0] (the previous fixed architecture); the rest are a compact,
# resource-aware search over depth/width/dropout/L2/learning-rate/batch-size. All candidates get
# the SAME epoch budget as the baseline for a fair comparison. No k-fold CV here (unlike the tree
# models): retraining a neural net 3-5x per candidate is too costly for an 8 GB machine, so
# selection uses the held-out validation split only (never test) -- standard practice for NNs.
CANDIDATES = [
    dict(label="baseline", hidden_units=[128, 64], dropout=[0.3, 0.2], l2=0.0, learning_rate=1e-3, batch_size=2048),
    dict(label="l2_regularized", hidden_units=[128, 64], dropout=[0.3, 0.2], l2=1e-4, learning_rate=1e-3, batch_size=2048),
    dict(label="smaller_more_l2", hidden_units=[64, 32], dropout=[0.3, 0.2], l2=1e-3, learning_rate=1e-3, batch_size=2048),
    dict(label="deeper", hidden_units=[128, 64, 32], dropout=[0.3, 0.3, 0.2], l2=1e-4, learning_rate=1e-3, batch_size=2048),
    dict(label="lower_lr_smaller_batch", hidden_units=[128, 64], dropout=[0.3, 0.2], l2=1e-4, learning_rate=5e-4, batch_size=1024),
]


def overfitting_status(gap: float) -> str:
    if gap <= 0.03:
        return "Well-Balanced"
    if gap <= 0.06:
        return "Moderate Overfit"
    return "Severe Overfit"


def build_model(n_features: int, config: dict) -> Sequential:
    regularizer = regularizers.l2(config["l2"]) if config["l2"] > 0 else None
    layers: list = [Dense(config["hidden_units"][0], activation="relu", kernel_regularizer=regularizer, input_shape=(n_features,)),
                    BatchNormalization(), Dropout(config["dropout"][0])]
    for units, dropout in zip(config["hidden_units"][1:], config["dropout"][1:]):
        layers += [Dense(units, activation="relu", kernel_regularizer=regularizer), BatchNormalization(), Dropout(dropout)]
    layers.append(Dense(1, activation="sigmoid"))
    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]), loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def train_candidate(config, X_train, y_train, X_val, y_val, class_weight):
    print(f"  -> training candidate '{config['label']}': {config}")
    model = build_model(X_train.shape[1], config)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=config["batch_size"], class_weight=class_weight,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, PATIENCE // 2), min_lr=1e-6),
        ],
        verbose=0,
    )
    val_probs = model.predict(X_val, batch_size=config["batch_size"], verbose=0).ravel()
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    print(f"     epochs run: {len(history.history['loss'])}, validation PR-AUC: {val_pr_auc:.6f}")
    return model, history, val_pr_auc


def evaluate_and_save(model, config, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, model_name, short_name):
    started = time.perf_counter()
    batch_size = config["batch_size"]
    train_probs = model.predict(X_train, batch_size=batch_size, verbose=0).ravel()
    val_probs = model.predict(X_val, batch_size=batch_size, verbose=0).ravel()
    threshold = find_best_threshold(y_val, val_probs)
    test_probs = model.predict(X_test, batch_size=batch_size, verbose=0).ravel()
    predictions = (test_probs >= threshold).astype(int)
    train_predictions = (train_probs >= threshold).astype(int)

    train_pr_auc = float(average_precision_score(y_train, train_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    train_roc_auc = float(roc_auc_score(y_train, train_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    train_f1 = float(f1_score(y_train, train_predictions, zero_division=0))
    test_f1 = float(f1_score(y_test, predictions, zero_division=0))
    pr_auc_gap = train_pr_auc - test_pr_auc
    f1_gap = train_f1 - test_f1
    roc_auc_gap = train_roc_auc - test_roc_auc
    diagnosis = overfitting_status(pr_auc_gap)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, test_probs)

    metrics = {
        "model_name": model_name, "threshold": float(threshold), "optimal_threshold": float(threshold),
        "pr_auc": test_pr_auc, "roc_auc": test_roc_auc, "f1_score": test_f1,
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "precision_curve": precision_curve.tolist(), "recall_curve": recall_curve.tolist(), "pr_thresholds": pr_thresholds.tolist(),
        "feature_names": feature_names, "feature_importances": [0.0] * len(feature_names),
        "imbalance_strategy": "dynamic balanced class weights (train-only)",
        "architecture": {key: value for key, value in config.items() if key != "label"},
        "execution_seconds": round(time.perf_counter() - started, 3),
        "train_pr_auc": train_pr_auc, "val_pr_auc": val_pr_auc, "test_pr_auc": test_pr_auc,
        "train_roc_auc": train_roc_auc, "val_roc_auc": val_roc_auc, "test_roc_auc": test_roc_auc,
        "train_f1": train_f1, "test_f1": test_f1,
        "pr_auc_gap": pr_auc_gap, "f1_gap": f1_gap, "roc_auc_gap": roc_auc_gap, "overfit_gap_auc": roc_auc_gap,
        "overfitting_diagnosis": diagnosis,
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
    }
    model.save(METRICS_DIR / f"{short_name}.h5")
    (METRICS_DIR / f"{short_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        f"Model Name: {model_name}", f"Architecture: {config}",
        f"Execution Time (seconds): {metrics['execution_seconds']}", f"Optimal Decision Threshold: {threshold:.6f}", "",
        "--- GENERALIZATION ---",
        f"PR-AUC  | Train={train_pr_auc:.6f} Val={val_pr_auc:.6f} Test={test_pr_auc:.6f} Gap={pr_auc_gap:.6f} {diagnosis}",
        f"ROC-AUC | Train={train_roc_auc:.6f} Val={val_roc_auc:.6f} Test={test_roc_auc:.6f} Gap={roc_auc_gap:.6f}",
        f"F1      | Train={train_f1:.6f} Test={test_f1:.6f} Gap={f1_gap:.6f}",
        "", "Metrics:",
    ]
    report.extend(f"{name}: {metrics[name]:.6f}" for name in ["pr_auc", "f1_score", "recall", "roc_auc", "accuracy", "precision"])
    report.extend(["", f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}", "", "Classification Report:", classification_report(y_test, predictions, target_names=["Ineffective (0)", "Effective (1)"], zero_division=0), ""])
    (METRICS_DIR / f"{short_name}_report.txt").write_text("\n".join(report), encoding="utf-8")
    save_model_charts(metrics, short_name)
    return metrics


def save_training_curves(history, short_name: str, model_name: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history.history["loss"], label="Train Loss")
    axis.plot(history.history["val_loss"], label="Validation Loss")
    axis.set(title=f"{model_name} Training vs Validation Loss", xlabel="Epoch", ylabel="Binary Cross-Entropy Loss")
    axis.legend(); figure.tight_layout(); figure.savefig(REPORTS_DIR / f"{short_name}_loss_curve.png", dpi=300); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history.history["pr_auc"], label="Train PR-AUC")
    axis.plot(history.history["val_pr_auc"], label="Validation PR-AUC")
    axis.set(title=f"{model_name} Training vs Validation PR-AUC", xlabel="Epoch", ylabel="PR-AUC")
    axis.legend(); figure.tight_layout(); figure.savefig(REPORTS_DIR / f"{short_name}_pr_auc_curve.png", dpi=300); plt.close(figure)


def main() -> None:
    ensure_output_dirs()
    print("=== [1/6] Loading data (leakage-safe 60/20/20 split, encoder+scaler fit on train only) ===")
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_data(scale=True)
    feature_names = list(X_train.columns)
    negative_count, positive_count = np.bincount(y_train.astype(int))
    class_weight = {0: 1.0, 1: float(negative_count / positive_count)}

    print(f"=== [2/6] Training {len(CANDIDATES)} candidates ({EPOCHS} max epochs, patience={PATIENCE}, validation-selected) ===")
    trained = []
    for config in CANDIDATES:
        model, history, val_pr_auc = train_candidate(config, X_train, y_train, X_val, y_val, class_weight)
        trained.append((model, history, val_pr_auc, config))

    baseline_model, baseline_history, baseline_val_pr_auc, baseline_config = trained[0]
    best_model, best_history, best_val_pr_auc, best_config = max(trained, key=lambda item: item[2])
    selected = "Baseline" if best_config is baseline_config else f"Tuned ({best_config['label']})"
    print(f"=== [3/6] Selected configuration: {selected} (validation PR-AUC {best_val_pr_auc:.6f} vs baseline {baseline_val_pr_auc:.6f}) ===")

    print("=== [4/6] Evaluating baseline on the untouched test set (for reporting comparison) ===")
    baseline_metrics = evaluate_and_save(baseline_model, baseline_config, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, "Keras MLP (Baseline)", "keras_baseline")
    save_training_curves(baseline_history, "keras_baseline", "Keras MLP (Baseline)")

    print(f"=== [5/6] Evaluating final selected model ({selected}) on the untouched test set ===")
    if best_config is baseline_config:
        final_metrics = baseline_metrics
    else:
        final_metrics = evaluate_and_save(best_model, best_config, X_train, y_train, X_val, y_val, X_test, y_test, feature_names, "Keras MLP", "keras")
        save_training_curves(best_history, "keras", "Keras MLP")

    print("=== [6/6] Saving baseline-vs-tuned comparison ===")
    comparison = {
        "selected_configuration": selected,
        "baseline_validation_pr_auc": baseline_val_pr_auc,
        "best_validation_pr_auc": best_val_pr_auc,
        "best_config": {key: value for key, value in best_config.items() if key != "label"},
        "baseline_test": {key: baseline_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "final_test": {key: final_metrics[key] for key in ("pr_auc", "f1_score", "precision", "recall", "roc_auc", "accuracy", "threshold")},
        "generalization": {
            "train_pr_auc": final_metrics["train_pr_auc"], "val_pr_auc": final_metrics["val_pr_auc"], "test_pr_auc": final_metrics["test_pr_auc"],
            "pr_auc_gap": final_metrics["pr_auc_gap"], "f1_gap": final_metrics["f1_gap"], "roc_auc_gap": final_metrics["roc_auc_gap"],
            "overfitting_diagnosis": final_metrics["overfitting_diagnosis"],
        },
    }
    (METRICS_DIR / "keras_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "KERAS MLP: BASELINE VS TUNED COMPARISON",
        "=" * 40,
        f"Selected configuration: {selected}",
        f"Baseline validation PR-AUC: {baseline_val_pr_auc:.6f}",
        f"Best validation PR-AUC:     {best_val_pr_auc:.6f}",
        f"Best config: {best_config}",
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
    (METRICS_DIR / "keras_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nKeras MLP training complete.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
