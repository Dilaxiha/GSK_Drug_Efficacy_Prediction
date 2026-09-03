"""Step 8: fit and evaluate an LSTM on one-step tabular sequences."""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM

from utils import METRICS_DIR, ensure_output_dirs, find_best_threshold, get_train_val_test_data


def main() -> None:
    ensure_output_dirs()
    started = time.perf_counter()
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_data()
    X_fit = X_train.to_numpy(dtype="float32").reshape((-1, 1, X_train.shape[1]))
    X_validation = X_val.to_numpy(dtype="float32").reshape((-1, 1, X_train.shape[1]))
    X_test_3d = X_test.to_numpy(dtype="float32").reshape((-1, 1, X_train.shape[1]))
    negative_count, positive_count = np.bincount(y_train.astype(int))
    class_weight = {0: 1.0, 1: float(negative_count / positive_count)}

    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(1, X_train.shape[1])),
        BatchNormalization(), Dropout(0.3),
        Dense(32, activation="relu"), Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        X_fit, y_train, validation_data=(X_validation, y_val),
        epochs=100, batch_size=2048, class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=1,
    )
    train_probs = model.predict(X_fit, batch_size=2048, verbose=0).ravel()
    val_probs = model.predict(X_validation, batch_size=2048, verbose=0).ravel()
    threshold = find_best_threshold(y_val, val_probs)
    probabilities = model.predict(X_test_3d, batch_size=2048, verbose=0).ravel()
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    metrics = {
        "model_name": "LSTM Network", "threshold": float(threshold), "optimal_threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr.tolist(), "tpr": tpr.tolist(),
        "feature_names": list(X_train.columns), "feature_importances": [0.0] * X_train.shape[1],
        "imbalance_strategy": "dynamic balanced class weights",
        "execution_seconds": round(time.perf_counter() - started, 3),
        "train_roc_auc": float(roc_auc_score(y_train, train_probs)), "val_roc_auc": float(roc_auc_score(y_val, val_probs)), "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "overfit_gap_auc": float(roc_auc_score(y_train, train_probs) - roc_auc_score(y_test, probabilities)),
        "test_pr_auc": float(average_precision_score(y_test, probabilities)), "test_f1": float(f1_score(y_test, predictions, zero_division=0)), "test_recall": float(recall_score(y_test, predictions, zero_division=0)), "test_accuracy": float(accuracy_score(y_test, predictions)), "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
    }
    model.save(METRICS_DIR / "lstm_model.h5")
    (METRICS_DIR / "lstm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = ["Model Name: LSTM Network", f"Execution Time (seconds): {metrics['execution_seconds']}", f"Optimal Decision Threshold: {threshold:.6f}", "", "Metrics:"]
    report.extend(f"{name}: {metrics[name]:.6f}" for name in ["pr_auc", "f1_score", "recall", "roc_auc", "accuracy", "precision"])
    report.extend(["", f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}", "", "Classification Report:", classification_report(y_test, predictions, target_names=["Ineffective (0)", "Effective (1)"], zero_division=0), ""])
    (METRICS_DIR / "lstm_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"LSTM complete: PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
