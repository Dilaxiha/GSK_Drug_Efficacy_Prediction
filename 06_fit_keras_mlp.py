"""Step 7: fit and evaluate a Keras MLP."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, f1_score)
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

from utils import METRICS_DIR, get_train_val_test_data, find_best_threshold, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    started = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_data()
    negative_count, positive_count = np.bincount(y_train.astype(int))
    class_weight = {0: 1.0, 1: float(negative_count / positive_count)}

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(), Dropout(0.3),
        Dense(64, activation="relu"), BatchNormalization(), Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=100, batch_size=2048, class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=1,
    )
    train_probs = model.predict(X_train, batch_size=2048, verbose=0).ravel()
    val_probs = model.predict(X_val, batch_size=2048, verbose=0).ravel()
    threshold = find_best_threshold(y_val, val_probs)
    probabilities = model.predict(X_test, batch_size=2048, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    metrics = {
        "model_name": "Keras MLP", "threshold": float(threshold), "optimal_threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "feature_names": list(X_train.columns),
        "feature_importances": [0.0] * X_train.shape[1],
        "imbalance_strategy": "dynamic balanced class weights",
        "execution_seconds": round(time.perf_counter() - started, 3),
        "train_roc_auc": float(roc_auc_score(y_train, train_probs)),
        "val_roc_auc": float(roc_auc_score(y_val, val_probs)),
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "overfit_gap_auc": float(roc_auc_score(y_train, train_probs) - roc_auc_score(y_test, probabilities)),
        "test_pr_auc": float(average_precision_score(y_test, probabilities)), "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)), "test_accuracy": float(accuracy_score(y_test, predictions)), "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
    }
    model.save(METRICS_DIR / "keras_mlp.h5")
    (METRICS_DIR / "keras_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = ["Model Name: Keras MLP", f"Execution Time (seconds): {metrics['execution_seconds']}", f"Optimal Decision Threshold: {threshold:.6f}", "", "Metrics:"]
    report.extend(f"{name}: {metrics[name]:.6f}" for name in ["pr_auc", "f1_score", "recall", "roc_auc", "accuracy", "precision"])
    report.extend(["", f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}", "", "Classification Report:", classification_report(y_test, predictions, target_names=["Ineffective (0)", "Effective (1)"], zero_division=0), ""])
    (METRICS_DIR / "keras_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Keras MLP complete: PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
